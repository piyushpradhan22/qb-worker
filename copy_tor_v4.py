import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import pandas as pd
import qbittorrentapi
import requests
from guessit import guessit
from huggingface_hub import HfApi
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

CREDENTIALS_URL = (
    "https://raw.githubusercontent.com/piyushpradhan22/credentials/refs/heads/main/credentials.json"
)
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# List of noise words/tags to remove from torrent names (from hf_torrent_push.py)
NOISE_PATTERN = [
    "tamilmv",
    "todaypk",
    "1tamilmv",
    "extramovies",
    "bolly4u",
    "cinevood",
]


@dataclass
class VideoFile:
    file_path: Path
    file_name: str
    size: int


# ==============================================================================
# Helper Functions from hf_torrent_push.py
# ==============================================================================

def clean_torrent_name(value: str) -> str:
    candidate = (value or "").strip()
    pattern = NOISE_PATTERN
    if not candidate or not pattern:
        return candidate

    if isinstance(pattern, (list, tuple, set)):
        pattern_str = "|".join(re.escape(p) for p in pattern if p)
    else:
        pattern_str = str(pattern).strip()

    if not pattern_str:
        return candidate

    if re.search(pattern_str, candidate, re.IGNORECASE):
        # Strip prefix containing noise tag up to ' - ' (or other standard dash variants)
        cleaned = re.sub(rf"^.*?(?:{pattern_str}).*?\s*[-–—]\s*", "", candidate, flags=re.IGNORECASE)
        # Fallback to general '^.*? - ' if pattern matched anywhere in prefix
        if cleaned == candidate:
            cleaned = re.sub(r"^.*? - \s*", "", candidate, flags=re.IGNORECASE)
        # Strip leading bracketed or loose tag at start if no dash separator
        if cleaned == candidate:
            cleaned = re.sub(rf"^\[?.*?(?:{pattern_str}).*?\]?\s*[-–—]?\s*", "", candidate, flags=re.IGNORECASE)
        return cleaned.strip() or candidate

    return candidate


def clean_title(value: str) -> str:
    if not value:
        return ""
    cleaned = value.strip()
    cleaned = re.sub(r"[._-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned)
    return cleaned.strip().lower()


def title_variants(title: str):
    variants = []
    seen = set()
    for candidate in [title, title.strip(" ._-")]:
        if not candidate:
            continue
        normalized = re.sub(r"\s+", " ", candidate).strip()
        if normalized and normalized not in seen:
            variants.append(normalized)
            seen.add(normalized)
    return variants


def digits_match(target: str, candidate: str) -> bool:
    target_digits = re.findall(r"\d+", target)
    if not target_digits:
        return True
    candidate_digits = re.findall(r"\d+", candidate)
    return target_digits == candidate_digits


def extract_infohash(value: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return ""

    match = re.search(r"btih:([0-9a-fA-F]{40}|[A-Z2-7]{32})", candidate, re.IGNORECASE)
    if match:
        digest = match.group(1).lower()
        if len(digest) == 32:
            pad = "=" * ((8 - len(digest) % 8) % 8)
            try:
                digest = base64.b32decode(digest + pad).hex()
            except Exception:
                pass
        return digest

    if re.fullmatch(r"[0-9a-fA-F]{40}", candidate):
        return candidate.lower()

    if re.fullmatch(r"[A-Z2-7]{32}", candidate):
        pad = "=" * ((8 - len(candidate) % 8) % 8)
        try:
            return base64.b32decode(candidate + pad).hex()
        except Exception:
            return candidate.lower()

    return ""


def extract_torrent_title(value: str) -> str:
    source = (value or "").strip()
    if source.lower().startswith("magnet:"):
        query = parse_qs(urlsplit(source).query)
        display_name = query.get("dn", [""])[0]
        if display_name:
            return clean_torrent_name(unquote(display_name).strip())
    return clean_torrent_name(source)


def resolve_imdb_id(title: str, year: int | None = None, is_series: bool = False) -> str:
    query = (title or "").strip()
    if not query:
        return ""

    variants = title_variants(query)
    for candidate in variants:
        cleaned = clean_title(candidate)
        if not cleaned:
            continue

        first = cleaned[0] if cleaned[0].isalnum() else "x"
        encoded = requests.utils.quote(candidate, safe="")
        url = f"https://v3.sg.media-imdb.com/suggestion/{first}/{encoded}.json"
        for attempt in range(1, 4):
            try:
                response = requests.get(url, headers=HEADERS, timeout=20)
                if response.status_code != 200:
                    continue
                payload = response.json()
                results = payload.get("d", [])
                if not results:
                    continue

                best = None
                best_score = -9999
                for idx, result in enumerate(results):
                    imdb_id = result.get("id", "")
                    if not imdb_id or not imdb_id.startswith("tt"):
                        continue

                    result_title = result.get("l", "")
                    result_year = result.get("y")
                    result_kind = str(result.get("q", "")).lower()
                    result_clean = clean_title(result_title)

                    if not digits_match(cleaned, result_clean):
                        continue

                    score = 0
                    if cleaned == result_clean:
                        score += 150
                    elif cleaned in result_clean or result_clean in cleaned:
                        score += 50

                    if year is not None and result_year is not None:
                        diff = abs(int(year) - int(result_year))
                        if diff == 0:
                            score += 80
                        elif diff == 1:
                            score += 20
                        else:
                            score -= 200

                    is_result_series = "series" in result_kind or "mini-series" in result_kind
                    if is_series == is_result_series:
                        score += 30
                    else:
                        score -= 50

                    score -= idx * 3
                    if score > best_score:
                        best_score = score
                        best = imdb_id

                if best and best_score > 0:
                    return best
            except Exception:
                if attempt == 3:
                    continue

    return ""


def build_imdb_key(imdb_id: str | None, is_series: bool, season: int | None, episode: int | None) -> str:
    imdb_value = (imdb_id or "").strip()
    if not imdb_value:
        return ""
    if is_series and season is not None and episode is not None:
        return f"{imdb_value}:{season}:{episode}"
    return imdb_value


def parse_torrent_metadata(torrent_name: str, file_name: str, imdb_override: str | None = None):
    torrent_name = torrent_name or ""
    file_name = file_name or ""
    torrent_title = extract_torrent_title(torrent_name)
    if torrent_title.lower().startswith("magnet:"):
        torrent_title = clean_torrent_name(file_name)

    parsed_torrent = guessit(torrent_title)
    parsed_file = guessit(clean_torrent_name(file_name))

    title = parsed_torrent.get("title") or parsed_file.get("title") or torrent_title
    year = parsed_torrent.get("year") or parsed_file.get("year")
    season = parsed_torrent.get("season") or parsed_file.get("season")
    episode = parsed_torrent.get("episode") or parsed_file.get("episode")

    if season is None or episode is None:
        match = re.search(r"[Ss](\d{1,2})[Ee](\d{1,2})", file_name)
        if match:
            season = season if season is not None else int(match.group(1))
            episode = episode if episode is not None else int(match.group(2))

    is_series = bool(
        parsed_torrent.get("type") == "episode"
        or parsed_file.get("type") == "episode"
        or season is not None
        or episode is not None
    )
    if isinstance(title, (list, tuple)):
        title = title[0]

    alt_title = parsed_torrent.get("alternative_title") or parsed_file.get("alternative_title")

    imdb_id = (imdb_override or "").strip()
    if not imdb_id:
        if is_series:
            # For TV series, try without year first since torrent year is the season broadcast year, not show debut year
            imdb_id = resolve_imdb_id(str(title), None, is_series=True)
            if not imdb_id and alt_title:
                imdb_id = resolve_imdb_id(str(alt_title), None, is_series=True)
            if not imdb_id and year is not None:
                imdb_id = resolve_imdb_id(str(title), int(year), is_series=True)
        else:
            imdb_id = resolve_imdb_id(str(title), int(year) if year is not None else None, is_series=False)
            # If title is very short (e.g. acronym like GOAT) or alt_title exists, try alternative_title
            if (not imdb_id or len(clean_title(str(title))) <= 4) and alt_title:
                alt_imdb = resolve_imdb_id(str(alt_title), int(year) if year is not None else None, is_series=False)
                if alt_imdb:
                    imdb_id = alt_imdb
            if not imdb_id and title:
                imdb_id = resolve_imdb_id(str(title), None, is_series=False)

    return {
        "title": str(title),
        "year": int(year) if year is not None else None,
        "season": int(season) if season is not None else None,
        "episode": int(episode) if episode is not None else None,
        "is_series": is_series,
        "imdb_id": imdb_id,
    }


def find_video_files(root_dir: Path):
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if "sample" in path.name.lower():
            continue
        if "trailer" in path.name.lower() or "teaser" in path.name.lower() or "preview" in path.name.lower():
            continue
        yield path


def ensure_hftor_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS hftor (
                    imdb_id TEXT,
                    name TEXT,
                    file_name TEXT,
                    url TEXT,
                    size BIGINT,
                    time DOUBLE PRECISION,
                    hash TEXT
                )
                """
            )
        )


# ==============================================================================
# Credentials & Worker Logic
# ==============================================================================

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


def parse_prefixed_torrent_name(name: str) -> tuple[str | None, str | None, str]:
    if not (name.startswith("imdbm:") or name.startswith("imdbs:")):
        return None, None, name

    parts = name.split(":", 2)
    if len(parts) < 3:
        return None, None, name

    type_indicator = "movie" if parts[0] == "imdbm" else "series"
    explicit_imdb_id = parts[1] if parts[1].startswith("tt") else None
    clean_name = parts[2]
    return type_indicator, explicit_imdb_id, clean_name


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
        ensure_hftor_table(postgres_engine)
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
        forced_kind, explicit_imdb_id, raw_torrent_name = parse_prefixed_torrent_name(tor.name)
        torrent_title = extract_torrent_title(raw_torrent_name)
        content_path = Path(tor.content_path)

        if content_path.is_file():
            video_files = [content_path] if content_path.suffix.lower() in VIDEO_EXTENSIONS else []
        elif content_path.is_dir():
            video_files = list(find_video_files(content_path))
        else:
            video_files = []

        if not video_files:
            print(f"No video files found for {tor.name}")
            if dry_run:
                print(f"[DRY RUN] Would delete torrent and files: {tor.hash}")
            else:
                qbt_client.torrents_delete(delete_files=True, torrent_hashes=tor.hash)
            continue

        infohash = extract_infohash(tor.hash) or tor.hash

        for index, video_file in enumerate(video_files, start=1):
            file_name = video_file.name
            remote_name = f"{infohash}_{index}"
            metadata = parse_torrent_metadata(torrent_title, file_name, imdb_override=explicit_imdb_id)
            imdb_id = metadata["imdb_id"]
            imdb_key = build_imdb_key(imdb_id, metadata["is_series"], metadata["season"], metadata["episode"])
            cleaned_name = clean_torrent_name(torrent_title or file_name)

            if not imdb_key:
                print(f"IMDb ID not resolved for {cleaned_name or file_name}; uploading with blank imdb_id")

            if dry_run:
                print(
                    "[DRY RUN] Would upload",
                    {"path": str(video_file), "path_in_repo": remote_name, "repo_id": repo_id},
                )
            else:
                upload_error = None
                for attempt in range(1, 4):
                    try:
                        print(f"Uploading {file_name} ({attempt}/3)")
                        hf_api.upload_file(
                            path_or_fileobj=str(video_file),
                            path_in_repo=remote_name,
                            repo_id=repo_id,
                            repo_type="dataset",
                        )
                        upload_error = None
                        break
                    except Exception as exc:
                        upload_error = exc
                        if attempt < 3:
                            print(f"Upload attempt {attempt} failed; retrying in {10 * attempt}s...")
                            time.sleep(10 * attempt)

                if upload_error is not None:
                    print(f"Upload failed for {file_name}: {upload_error}")
                    continue

            clean_file_name = clean_torrent_name(file_name)
            server_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{remote_name}?download=true"
            row = {
                "imdb_id": imdb_key,
                "name": cleaned_name or clean_file_name,
                "file_name": clean_file_name,
                "url": server_url,
                "size": video_file.stat().st_size,
                "time": time.time(),
                "hash": infohash,
            }

            print(f"Resolved IMDb key: {imdb_key}")

            if not dry_run and postgres_engine is not None:
                try:
                    pd.DataFrame([row]).to_sql(name="hftor", con=postgres_engine, if_exists="append", index=False)
                    print(f"Inserted metadata row for {file_name} into Postgres")
                except Exception as exc:
                    print(f"Postgres insert failed for {file_name}: {exc}")

        if dry_run:
            print(f"[DRY RUN] Would delete torrent and files: {tor.hash}")
        else:
            qbt_client.torrents_delete(delete_files=True, torrent_hashes=tor.hash)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload completed torrents using new hf_torrent_push approach")
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
        res = parse_torrent_metadata(torrent_val, file_val)
        final_id = build_imdb_key(res["imdb_id"], res["is_series"], res["season"], res["episode"])
        print(f"\nResolved ID : {final_id}")
        print(f"Details     : is_series={res['is_series']}, season={res['season']}, episode={res['episode']}")
    elif args.test:
        print("Running parser and resolver self-tests...")
        scenarios = [
            ("Partner (2007) 1080p bluray", "Partner.2007.1080p.BluRay.x264.AAC5.1-[YTS.MX].mp4", "tt0807758"),
            ("Sandeep Aur Pinky Faraar (2021) 1080p web", "Sandeep.Aur.Pinky.Faraar.2021.1080p.WEBRip.x264.AAC5.1-[YTS.MX].mp4", "tt7094488"),
            ("Stranger Things S01 (2016) Season 1 BluRay 1080p 10bit HEVC [Hindi DDP 5.1 - English AAC 5.1] x265 -RONIN", "Stranger Things S01E03 Chapter Three - Holly, Jolly.mkv", "tt4574334:1:3"),
            ("Twisters.2024.2160p.WEB-DL.DV.HDR10.PLUS.ENG.LATINO.HINDI.DDP5.1.Atmos.H265.MKV-BEN.THE.ME...", "Twisters.2024.2160p.WEB-DL.DV.HDR10.PLUS.ENG.LATINO.HINDI.DDP5.1.Atmos.H265.MKV-BEN.THE.MEN.mkv", "tt12584954"),
            ("Almost Pyaar with DJ Mohabbat (2022) 1080p web", "Almost.Pyaar.With.DJ.Mohabbat.2023.1080p.WEBRip.x264.AAC5.1-[YTS.MX].mp4", "tt23472806"),
            ("Mirzapur.2024.S03.1080p.AMZN.WEB-DL.HEVC.DDP5.1.Esub-KIN", "Mirzapur_S03E10_Pratibimbh.mkv", "tt6473300:3:10"),
            ("www.1TamilMV.yt - GOAT - The Greatest Of All Time (2024) [Hindi - Tamil - Telugu] 1080p", "GOAT (2024).mkv", "tt27487934"),
        ]
        passed = 0
        for name, filename, expected in scenarios:
            print(f"\nTesting: '{name}'")
            res = parse_torrent_metadata(name, filename)
            resolved = build_imdb_key(res["imdb_id"], res["is_series"], res["season"], res["episode"])
            print(f"Result: {resolved} (Expected: {expected})")
            if resolved == expected:
                print("Status: PASSED")
                passed += 1
            else:
                print("Status: FAILED")
        print(f"\nSelf-tests result: {passed}/{len(scenarios)} passed.")
    else:
        if args.dry_run:
            print("Running in DRY RUN mode: no upload, DB writes, pause, or delete will be performed.")
        process(dry_run=args.dry_run)
