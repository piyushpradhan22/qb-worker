import argparse
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import qbittorrentapi
import requests
from guessit import guessit
from huggingface_hub import HfApi
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from torrent_metadata import parse_prefixed_torrent_name


CREDENTIALS_URL = (
    "https://raw.githubusercontent.com/piyushpradhan22/credentials/refs/heads/main/credentials.json"
)
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}


@dataclass
class VideoFile:
    file_path: str
    file_hash: str
    size: int


@dataclass
class ParseResult:
    imdb_id: str
    is_series: bool
    season: int | None
    episode: int | None


def load_credentials(dry_run: bool = False) -> dict[str, Any]:
    credentials_file = os.getenv("CREDENTIALS_FILE", "credentials.json")
    local_path = Path(credentials_file)

    data: dict[str, Any] | None = None
    token = os.getenv("token")
    if token:
        response = requests.get(
            CREDENTIALS_URL,
            headers={"Authorization": f"token {token}"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    elif local_path.exists():
        data = json.loads(local_path.read_text(encoding="utf-8"))
    else:
        raise RuntimeError(
            "Missing credentials. Set env var token or provide local credentials file via CREDENTIALS_FILE"
        )

    required = ["username", "password", "repo_id"]
    if not dry_run:
        required.extend(["postgres_url", "hf_token"])
    missing = [key for key in required if key not in data or not data[key]]
    if missing:
        raise RuntimeError(f"Missing keys in credentials payload: {missing}")
    return data


def get_video_files(content_path: str, torrent_hash: str) -> list[VideoFile]:
    path = os.path.abspath(content_path)
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return [VideoFile(file_path=path, file_hash=torrent_hash, size=os.path.getsize(path))]
        return []

    results: list[VideoFile] = []
    counter = 1
    for root, _, files in os.walk(path):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                continue
            full_path = os.path.join(root, file_name)
            results.append(
                VideoFile(
                    file_path=full_path,
                    file_hash=f"{torrent_hash}_{counter}",
                    size=os.path.getsize(full_path),
                )
            )
            counter += 1
    return results


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _regex_episode(file_name: str) -> tuple[int | None, int | None]:
    match = re.search(r"[Ss](\d{1,2})[Ee](\d{1,2})", file_name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_title(title: str) -> str:
    cleaned = title.lower()
    if cleaned.startswith("the "):
        cleaned = cleaned[4:]
    # Replace dots and hyphens to unify titles like "K.G.F" or "Hanu-Man"
    cleaned = cleaned.replace(".", "").replace("-", "")
    # Merge single characters separated by spaces (e.g., "k g f" -> "kgf")
    cleaned = re.sub(r"\b([a-zA-Z])\s+(?=[a-zA-Z]\b)", r"\1", cleaned)
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def digits_match(target: str, candidate: str) -> bool:
    """Ensure digits/numbers in the titles match exactly to prevent sequel/parts collisions.
    Only enforce if the target title contains digits."""
    target_digits = re.findall(r"\d+", target)
    if not target_digits:
        return True  # Allow any candidate if target has no digits
    candidate_digits = re.findall(r"\d+", candidate)
    return target_digits == candidate_digits

def fix_guessit_title(parsed, original_name) -> str | None:
    title = parsed.get("title")
    container = parsed.get("container")
    if container and title:
        containers = [container] if isinstance(container, str) else container
        for c in containers:
            if not original_name.lower().endswith("." + c.lower()):
                # It was a false container! Check if the title is missing this word
                pattern = rf"^{c}\b"
                if re.search(pattern, original_name, re.IGNORECASE):
                    title = f"{c} {title}"
                    title = re.sub(r"\s+", " ", title).strip()
    return title

def is_bonus_extra(name: str) -> bool:
    extra_keywords = [
        "making of", "featurette", "deleted", "trailer", "extra", 
        "bonus", "behind the scenes", "scene", "video musicale", 
        "music video", "interview", "promo", "clip"
    ]
    name_lower = name.lower()
    return any(kw in name_lower for kw in extra_keywords)

def resolve_via_tvmaze(title: str) -> str | None:
    try:
        url = "https://api.tvmaze.com/singlesearch/shows"
        resp = requests.get(url, params={"q": title}, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            imdb_id = data.get("externals", {}).get("imdb")
            if imdb_id:
                return imdb_id
    except Exception:
        pass
    return None

def resolve_via_imdb_suggestion(title: str, year: int | None = None, is_series: bool = False) -> str | None:
    cleaned_target_title = clean_title(title)
    if not cleaned_target_title:
        return None
    
    first_char = cleaned_target_title[0] if cleaned_target_title[0].isalnum() else "x"
    encoded_query = urllib.parse.quote(cleaned_target_title)
    url = f"https://v3.sg.media-imdb.com/suggestion/{first_char}/{encoded_query}.json"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        results = data.get("d", [])
        if not results:
            return None
        
        best_candidate = None
        highest_score = -9999
        
        for idx, result in enumerate(results):
            imdb_id = result.get("id", "")
            if not imdb_id.startswith("tt"):
                continue
            
            result_title = result.get("l", "")
            result_year = result.get("y")
            result_kind = result.get("q", "").lower()
            
            cleaned_result_title = clean_title(result_title)
            
            if not digits_match(cleaned_target_title, cleaned_result_title):
                continue
            
            title_score = 0
            if cleaned_target_title == cleaned_result_title:
                title_score = 150
            elif cleaned_result_title.startswith(cleaned_target_title):
                title_score = 80
            elif cleaned_target_title in cleaned_result_title:
                if len(cleaned_target_title) > 5 and " " in cleaned_target_title:
                    title_score = 40
            
            if title_score == 0:
                continue
            
            year_score = 0
            if year is not None:
                if result_year is not None:
                    diff = abs(year - result_year)
                    if diff == 0:
                        year_score = 100
                    elif diff == 1:
                        if title_score == 150:
                            year_score = 50
                        else:
                            year_score = -300
                    else:
                        year_score = -300
                else:
                    year_score = -15
            
            kind_score = 0
            is_result_series = "series" in result_kind or "mini-series" in result_kind
            is_result_movie = "feature" in result_kind or "movie" in result_kind or result_kind == "documentary"
            
            if is_series == is_result_series:
                kind_score = 50
            elif is_series and is_result_movie:
                kind_score = -50
            elif not is_series and is_result_series:
                kind_score = -50
                
            rank_score = -idx * 5
            total_score = title_score + year_score + kind_score + rank_score
            
            if total_score > highest_score:
                highest_score = total_score
                best_candidate = imdb_id
                
        if highest_score > 0:
            return best_candidate
            
    except Exception:
        pass
    return None

def get_top_imdb_suggestion_fallback(title: str, year: int | None = None) -> str | None:
    cleaned_target_title = clean_title(title)
    if not cleaned_target_title:
        return None
    
    first_char = cleaned_target_title[0] if cleaned_target_title[0].isalnum() else "x"
    encoded_query = urllib.parse.quote(cleaned_target_title)
    url = f"https://v3.sg.media-imdb.com/suggestion/{first_char}/{encoded_query}.json"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            results = resp.json().get("d", [])
            for result in results[:2]:
                imdb_id = result.get("id", "")
                if imdb_id.startswith("tt"):
                    result_title = result.get("l", "")
                    cleaned_result_title = clean_title(result_title)
                    
                    if not digits_match(cleaned_target_title, cleaned_result_title):
                        continue
                        
                    # Fallback length ratio check
                    len1 = len(cleaned_target_title)
                    len2 = len(cleaned_result_title)
                    if min(len1, len2) / max(len1, len2) < 0.4:
                        continue
                        
                    result_year = result.get("y")
                    if year is not None and result_year is not None:
                        if abs(year - result_year) > 3:
                            continue
                    return imdb_id
    except Exception:
        pass
    return None

def parse_torrent_metadata(torrent_name: str, torrent_file: str) -> ParseResult:
    parsed_file = guessit(torrent_file)
    parsed_tor = guessit(torrent_name)
    
    title_tor = fix_guessit_title(parsed_tor, torrent_name) or parsed_tor.get("title")
    title_file = fix_guessit_title(parsed_file, torrent_file) or parsed_file.get("title")
    
    if title_file and is_bonus_extra(torrent_file):
        title = title_tor or title_file
    elif title_tor and title_file:
        if len(title_file) > len(title_tor) and clean_title(title_tor) in clean_title(title_file):
            title = title_file
        elif len(title_tor) > len(title_file) and clean_title(title_file) in clean_title(title_tor):
            title = title_tor
        else:
            title = title_tor
    else:
        title = title_tor or title_file
    
    year = parsed_tor.get("year") or parsed_file.get("year")
    season = parsed_file.get("season") or parsed_tor.get("season")
    episode = parsed_file.get("episode") or parsed_tor.get("episode")
    
    if season is not None and season > 100:
        season = None

    is_series = (parsed_file.get("type") == "episode" or 
                 parsed_tor.get("type") == "episode" or 
                 season is not None or 
                 episode is not None)

    if not title:
        raise ValueError("Could not parse title from torrent name or file name")

    imdb_id = None
    if is_series:
        imdb_id = resolve_via_tvmaze(title)
        
    if not imdb_id:
        imdb_id = resolve_via_imdb_suggestion(title, year, is_series)
        

    if not imdb_id and title_tor and title_tor != title:
        imdb_id = resolve_via_imdb_suggestion(title_tor, year, is_series)
    if not imdb_id and title_file and title_file != title:
        imdb_id = resolve_via_imdb_suggestion(title_file, year, is_series)
        
    if not imdb_id and len(clean_title(title)) < 15:
        stripped_title = clean_title(title).replace(" ", "")
        if stripped_title != clean_title(title):
            imdb_id = resolve_via_imdb_suggestion(stripped_title, year, is_series)
            
    if not imdb_id:
        imdb_id = get_top_imdb_suggestion_fallback(title, year)

    return ParseResult(
        imdb_id=imdb_id or "",
        is_series=is_series,
        season=season,
        episode=episode,
    )


def build_imdb_id(base_imdb_id: str, is_series: bool, season: int | None, episode: int | None) -> str:
    imdb_base = str(base_imdb_id or "").strip()
    if not imdb_base:
        return ""
    if is_series and season is not None and episode is not None:
        return f"{imdb_base}:{season}:{episode}"
    return imdb_base


def process(dry_run: bool = False) -> None:
    creds = load_credentials(dry_run=dry_run)

    qbt_client = qbittorrentapi.Client(
        host="localhost",
        port=7860,
        username=creds["username"],
        password=creds["password"],
    )
    qbt_client.auth_log_in()

    postgres_engine = None
    hf_api = None
    if not dry_run:
        postgres_engine = create_engine(creds["postgres_url"], poolclass=NullPool)
        hf_api = HfApi(token=creds["hf_token"])
    repo_id = creds["repo_id"]

    torrents = [tor for tor in qbt_client.torrents_info() if tor.progress == 1 and tor.state != "pausedUP"]
    print("Received torrents:", [tor.name for tor in torrents])

    if torrents:
        if dry_run:
            print(f"[DRY RUN] Would pause {len(torrents)} completed torrents")
        else:
            qbt_client.torrents_pause([tor.hash for tor in torrents])

    for tor in torrents:
        forced_kind, explicit_imdb_id, clean_torrent_name = parse_prefixed_torrent_name(tor.name)
        explicit_imdb_id = str(explicit_imdb_id or "").strip()

        video_files = get_video_files(tor.content_path, tor.hash)
        if not video_files:
            print(f"No video files found for {tor.name}")
            if dry_run:
                print(f"[DRY RUN] Would delete torrent and files: {tor.hash}")
            else:
                qbt_client.torrents_delete(delete_files=True, torrent_hashes=tor.hash)
            continue

        for video in video_files:
            file_name = os.path.basename(video.file_path)
            if "sample" in file_name.lower() or "trailer" in file_name.lower() or "teaser" in file_name.lower() or "preview" in file_name.lower():
                continue

            if dry_run:
                print(
                    "[DRY RUN] Would upload",
                    {"path": video.file_path, "path_in_repo": video.file_hash, "repo_id": repo_id},
                )
            else:
                upload_result = hf_api.upload_file(
                    path_or_fileobj=video.file_path,
                    path_in_repo=video.file_hash,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(upload_result)

            parse_result: ParseResult | None = None
            try:
                parse_result = parse_torrent_metadata(clean_torrent_name, file_name)
            except Exception as exc:
                print(f"Parse API failed for {file_name}: {exc}")

            is_series = forced_kind == "series" or bool(parse_result and parse_result.is_series)
            season = parse_result.season if parse_result else None
            episode = parse_result.episode if parse_result else None

            if is_series and (season is None or episode is None):
                regex_season, regex_episode = _regex_episode(file_name)
                season = season if season is not None else regex_season
                episode = episode if episode is not None else regex_episode

            parsed_imdb_id = parse_result.imdb_id if parse_result else ""
            base_imdb_id = explicit_imdb_id or parsed_imdb_id
            final_imdb_id = build_imdb_id(base_imdb_id, is_series, season, episode)

            server_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{video.file_hash}?download=true"

            row = {
                "imdb_id": final_imdb_id,
                "name": clean_torrent_name,
                "file_name": file_name,
                "url": server_url,
                "size": video.size,
                "time": time.time(),
                "hash": tor.hash,
            }
            if dry_run:
                print("[DRY RUN] Computed row:", row)
            else:
                try:
                    pd.DataFrame([row]).to_sql(name="hftor", con=postgres_engine, if_exists="append", index=False)
                except Exception as exc:
                    print(f"DB insert failed for {file_name}: {exc}")

        if dry_run:
            print(f"[DRY RUN] Would delete torrent and files: {tor.hash}")
        else:
            qbt_client.torrents_delete(delete_files=True, torrent_hashes=tor.hash)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload completed torrents and resolve metadata via keyless resolver")
    parser.add_argument("--dry-run", action="store_true", help="Run without upload, DB insert, pause, or delete")
    parser.add_argument("--test", action="store_true", help="Run self-tests on the metadata resolver")
    parser.add_argument("--torrent", type=str, help="Custom torrent name to test resolve")
    parser.add_argument("--file", type=str, help="Custom file name to test resolve")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.torrent or args.file:
        torrent_val = args.torrent or ""
        file_val = args.file or ""
        print("Testing custom metadata resolution:")
        print(f"  Torrent Name: '{torrent_val}'")
        print(f"  File Name   : '{file_val}'")
        try:
            res = parse_torrent_metadata(torrent_val, file_val)
            final_id = build_imdb_id(res.imdb_id, res.is_series, res.season, res.episode)
            print(f"\nResolved ID : {final_id}")
            print(f"Details     : is_series={res.is_series}, season={res.season}, episode={res.episode}")
        except Exception as e:
            print(f"\nResolution Error: {e}")
    elif args.test:
        print("Running parser and resolver self-tests...")
        scenarios = [
            ("Partner (2007) 1080p bluray", "Partner.2007.1080p.BluRay.x264.AAC5.1-[YTS.MX].mp4", "tt0807758"),
            ("Sandeep Aur Pinky Faraar (2021) 1080p web", "Sandeep.Aur.Pinky.Faraar.2021.1080p.WEBRip.x264.AAC5.1-[YTS.MX].mp4", "tt7094488"),
            ("Stranger Things S01 (2016) Season 1 BluRay 1080p 10bit HEVC [Hindi DDP 5.1 - English AAC 5.1] x265 -RONIN", "Stranger Things S01E03 Chapter Three - Holly, Jolly.mkv", "tt4574334:1:3"),
            ("Twisters.2024.2160p.WEB-DL.DV.HDR10.PLUS.ENG.LATINO.HINDI.DDP5.1.Atmos.H265.MKV-BEN.THE.ME...", "Twisters.2024.2160p.WEB-DL.DV.HDR10.PLUS.ENG.LATINO.HINDI.DDP5.1.Atmos.H265.MKV-BEN.THE.MEN.mkv", "tt12584954"),
            ("Almost Pyaar with DJ Mohabbat (2022) 1080p web", "Almost.Pyaar.With.DJ.Mohabbat.2023.1080p.WEBRip.x264.AAC5.1-[YTS.MX].mp4", "tt23472806"),
            ("Mirzapur.2024.S03.1080p.AMZN.WEB-DL.HEVC.DDP5.1.Esub-KIN", "Mirzapur_S03E10_Pratibimbh.mkv", "tt6473300:3:10")
        ]
        passed = 0
        for name, filename, expected in scenarios:
            print(f"\nTesting: '{name}'")
            try:
                res = parse_torrent_metadata(name, filename)
                resolved = build_imdb_id(res.imdb_id, res.is_series, res.season, res.episode)
                print(f"Result: {resolved} (Expected: {expected})")
                if resolved == expected:
                    print("Status: PASSED")
                    passed += 1
                else:
                    print("Status: FAILED")
            except Exception as e:
                print(f"Error: {e}")
                print("Status: FAILED")
        print(f"\nSelf-tests result: {passed}/{len(scenarios)} passed.")
    else:
        if args.dry_run:
            print("Running in DRY RUN mode: no upload, DB writes, pause, or delete will be performed.")
        process(dry_run=args.dry_run)