import argparse
import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import pandas as pd
import qbittorrentapi
import requests
from torrent_metadata import parse_prefixed_torrent_name, resolve_media_identity, build_imdb_id, MediaMetadata
from huggingface_hub import HfApi
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool


CREDENTIALS_URL = (
    "https://raw.githubusercontent.com/piyushpradhan22/credentials/refs/heads/main/credentials.json"
)
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}


@dataclass
class VideoFile:
    file_path: str
    file_hash: str
    size: int


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
        video_files = get_video_files(tor.content_path, tor.hash)
        if not video_files:
            print(f"No video files found for {tor.name}")
            if dry_run:
                print(f"[DRY RUN] Would delete torrent and files: {tor.hash}")
            else:
                qbt_client.torrents_delete(delete_files=True, torrent_hashes=tor.hash)
            continue

        # Resolve metadata once per torrent (AI + Wikipedia lookup)
        first_file_name = os.path.basename(video_files[0].file_path)
        base_imdb_id, tor_metadata = resolve_media_identity(
            torrent_name=clean_torrent_name,
            file_name=first_file_name,
            explicit_imdb_id=explicit_imdb_id,
            forced_kind=forced_kind,
        )
        # Strip any season:episode suffix from the base ID for series
        base_imdb_id_only = base_imdb_id.split(":")[0] if base_imdb_id else ""

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

            # For series: extract per-file season/episode from filename via regex
            if tor_metadata.kind == "series":
                se_match = re.search(r"[Ss](\d{1,2})[Ee](\d{1,2})", file_name)
                if se_match:
                    file_metadata = MediaMetadata(
                        title=tor_metadata.title,
                        kind="series",
                        year=tor_metadata.year,
                        season=int(se_match.group(1)),
                        episode=int(se_match.group(2)),
                        confidence=tor_metadata.confidence,
                    )
                    final_imdb_id = build_imdb_id(base_imdb_id_only, file_metadata)
                else:
                    final_imdb_id = base_imdb_id_only
            else:
                final_imdb_id = base_imdb_id_only
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
    parser = argparse.ArgumentParser(description="Upload completed torrents and map IMDb metadata")
    parser.add_argument("--dry-run", action="store_true", help="Run without upload, DB insert, pause, or delete")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        print("Running in DRY RUN mode: no upload, DB writes, pause, or delete will be performed.")
    process(dry_run=args.dry_run)