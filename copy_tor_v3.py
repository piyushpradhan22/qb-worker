import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import qbittorrentapi
import requests
from huggingface_hub import HfApi
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from torrent_metadata import parse_prefixed_torrent_name


CREDENTIALS_URL = (
    "https://raw.githubusercontent.com/piyushpradhan22/credentials/refs/heads/main/credentials.json"
)
PARSE_API_BASE = os.getenv(
    "PARSE_API_BASE",
    "https://guzowskilynottmvvy49011733757114-ptn.hf.space",
).rstrip("/")
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


def parse_torrent_metadata(torrent_name: str, torrent_file: str) -> ParseResult:
    payload = {"torrent_name": torrent_name, "torrent_file": torrent_file}
    response = requests.post(f"{PARSE_API_BASE}/api/parse", json=payload, timeout=30)
    response.raise_for_status()

    body = response.json()
    if body.get("status") != "success":
        raise RuntimeError(body.get("error") or "parse API returned non-success status")

    data = body.get("data") or {}
    episode = data.get("episode") or {}
    title_type = str(data.get("title_type") or "").lower()

    return ParseResult(
        imdb_id=str(data.get("imdb_id") or "").strip(),
        is_series=bool(data.get("is_series")) or title_type in {"tvseries", "tvmini", "tvminiseries"},
        season=_to_int(episode.get("season")),
        episode=_to_int(episode.get("episode")),
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
            if "sample" in file_name.lower():
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
    parser = argparse.ArgumentParser(description="Upload completed torrents and resolve metadata via parse API")
    parser.add_argument("--dry-run", action="store_true", help="Run without upload, DB insert, pause, or delete")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        print("Running in DRY RUN mode: no upload, DB writes, pause, or delete will be performed.")
    process(dry_run=args.dry_run)