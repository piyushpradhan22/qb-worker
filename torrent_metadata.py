import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests


IMDB_ID_RE = re.compile(r"^tt\d{7,9}$", flags=re.IGNORECASE)


@dataclass
class MediaMetadata:
    title: str
    kind: str | None = None
    year: int | None = None
    season: int | None = None
    episode: int | None = None
    confidence: float | None = None


def search_imdb_api(title: str, year: int | None = None, kind: str | None = None) -> str:
    """Search IMDb suggestion API for an IMDb ID.
    Returns the IMDb ID (e.g. tt1375666) or empty string if not found."""
    if not title or not title.strip():
        return ""

    headers = {"User-Agent": "qb-worker/1.0 (torrent metadata resolver; python-requests)"}
    
    try:
        query = title.strip().lower()
        first_letter = query[0] if query else "x"
        url = f"https://v3.sg.media-imdb.com/suggestion/{first_letter}/{query}.json"
        
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        results = data.get("d", [])
        if not results:
            return ""
        
        # Filter by year and kind if provided
        for result in results:
            result_year = result.get("y")
            result_kind = result.get("q", "").lower()
            
            # Match year if specified
            if year and result_year != year:
                continue
            
            # Match kind if specified
            if kind:
                if kind == "movie" and result_kind not in ["feature", "tv movie"]:
                    continue
                if kind == "series" and result_kind not in ["tv series", "tv mini-series"]:
                    continue
            
            imdb_id = result.get("id", "")
            if imdb_id and IMDB_ID_RE.match(imdb_id):
                return imdb_id
        
        return ""
    except Exception:
        return ""


def web_search_imdb(query: str) -> str:
    """Search for an IMDb ID via Wikipedia and IMDb API.
    Query format: 'Title year film' or 'Title TV series'.
    Returns the IMDb ID (e.g. tt1375666) or a descriptive failure string."""
    if not query or not query.strip():
        return "No results found"

    headers = {"User-Agent": "qb-worker/1.0 (torrent metadata resolver; python-requests)"}
    wiki_api_url = "https://en.wikipedia.org/w/api.php"

    try:
        # Step 1: Try Wikipedia first
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
        }
        resp = requests.get(wiki_api_url, params=search_params, headers=headers, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        
        if results:
            # Step 2: Check external links in top articles for IMDb link
            for result in results[:2]:
                page_title = result["title"]
                ext_params = {
                    "action": "query",
                    "prop": "extlinks",
                    "titles": page_title,
                    "format": "json",
                    "ellimit": 50,
                }
                ext_resp = requests.get(wiki_api_url, params=ext_params, headers=headers, timeout=10)
                ext_resp.raise_for_status()
                pages = ext_resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    for link in page.get("extlinks", []):
                        url = link.get("*", "")
                        imdb_match = re.search(r"imdb\.com/title/(tt\d{7,9})", url)
                        if imdb_match:
                            return imdb_match.group(1)
        
        # Step 3: Fallback to IMDb API if Wikipedia fails
        # Parse title, year, and kind from query string
        title = query
        year = None
        kind = None
        
        # Try to extract year from query (e.g., "Hoppers 2026 film")
        year_match = re.search(r"\b(19|20)\d{2}\b", query)
        if year_match:
            year = int(year_match.group(0))
            title = query[:year_match.start()].strip()
        
        # Try to detect kind from query
        if "series" in query.lower():
            kind = "series"
        elif "film" in query.lower() or "movie" in query.lower():
            kind = "movie"
        
        imdb_id = search_imdb_api(title, year, kind)
        if imdb_id:
            return imdb_id
        
        return "No IMDb ID found"
    except Exception as e:
        return f"Search failed: {str(e)}"

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


def pick_ai_provider() -> tuple[str | None, str | None, str | None]:
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    nvidia_key = os.getenv("NVIDIA_API_KEY")

    if openrouter_key:
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        return "openrouter", openrouter_key, model
    if nvidia_key:
        model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
        return "nvidia", nvidia_key, model
    return None, None, None


def chat_json(system_prompt: str, user_prompt: str, enable_tools: bool = False) -> dict[str, Any] | None:
    provider, api_key, model = pick_ai_provider()
    if not provider or not api_key or not model:
        raise RuntimeError("AI key missing. Set OPENROUTER_API_KEY or NVIDIA_API_KEY")

    # Define tools/functions for LLM
    tools = None
    if enable_tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search_imdb",
                    "description": (
                        "Search Wikipedia for a film or TV show's IMDb ID. "
                        "Use a clean query like 'Inception 2010 film' or 'Breaking Bad TV series'. "
                        "Returns the IMDb ID directly (e.g. tt1375666) if found."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Clean search query, e.g. 'The Buckingham Murders 2024 film'"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    else:
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    max_iterations = 6
    iteration = 0
    found_imdb_id: str | None = None

    while iteration < max_iterations:
        iteration += 1
        try:
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
            }
            # Disable tools once we have the IMDb ID to force final JSON response
            if tools and found_imdb_id is None:
                payload["tools"] = tools

            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]

            # LLM wants to call a tool
            if choice.get("finish_reason") == "tool_calls" and choice.get("message", {}).get("tool_calls"):
                tool_calls = choice["message"]["tool_calls"]
                messages.append({
                    "role": "assistant",
                    "content": choice["message"].get("content") or "",
                    "tool_calls": tool_calls,
                })

                for tool_call in tool_calls:
                    try:
                        tool_args = json.loads(tool_call["function"]["arguments"])
                    except Exception:
                        tool_args = {}

                    query = tool_args.get("query") or (list(tool_args.values())[0] if tool_args else "")
                    search_result = web_search_imdb(str(query))

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": search_result,
                    })

                    # Track if we found an IMDb ID
                    if re.match(r"^tt\d{7,9}$", search_result.strip(), re.IGNORECASE):
                        found_imdb_id = search_result.strip().lower()

                # After processing all tool calls, if we found an IMDb ID,
                # inject a user message so the LLM can now produce the final JSON
                if found_imdb_id:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The IMDb ID is {found_imdb_id}. "
                            "Now return the final metadata as JSON with keys: "
                            "title, kind, year, season, episode, imdb_id, confidence."
                        ),
                    })
                continue

            # Normal text response — extract JSON
            content = choice["message"]["content"] or ""
            try:
                return json.loads(content)
            except Exception:
                json_block = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if not json_block:
                    return None
                try:
                    return json.loads(json_block.group(0))
                except Exception:
                    return None

        except Exception:
            return None

    return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _normalize_kind(kind: Any) -> str | None:
    value = str(kind or "").strip().lower()
    if value in {"movie", "series"}:
        return value
    return None


def _normalize_imdb_id(imdb_id: Any) -> str:
    value = str(imdb_id or "").strip()
    if IMDB_ID_RE.match(value):
        return value.lower()
    return ""


def ai_extract_metadata(torrent_name: str, file_name: str) -> tuple[str, MediaMetadata]:
    # Phase 1: Extract metadata (title, kind, year, season, episode) without tools
    meta_system = (
        "You are a media metadata extractor. Return only JSON with keys: "
        "title, kind, year, season, episode, imdb_id, confidence. "
        "Rules: kind must be 'movie' or 'series' (S##E## patterns = series). "
        "imdb_id: include if you know it from training, else empty string. "
        "Use null for unknown numeric values. Confidence 0-1. JSON only, no extra text."
    )
    meta_user = (
        f"Torrent name: {torrent_name}\n"
        f"File name: {file_name}\n"
        "Extract: title (cleaned), kind, year, season (if series), episode (if series), imdb_id (if known), confidence."
    )
    raw = chat_json(meta_system, meta_user, enable_tools=False) or {}

    title = (raw.get("title") or "").strip() or torrent_name
    kind = _normalize_kind(raw.get("kind"))
    year = _to_int(raw.get("year"))
    season = _to_int(raw.get("season"))
    episode = _to_int(raw.get("episode"))
    confidence = raw.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except Exception:
        confidence_value = None

    metadata = MediaMetadata(
        title=title, kind=kind, year=year, season=season, episode=episode,
        confidence=confidence_value,
    )

    imdb_id = _normalize_imdb_id(raw.get("imdb_id"))

    # Phase 2: If no IMDb ID yet, search Wikipedia via function calling
    if not imdb_id and title:
        search_system = (
            "You are an IMDb lookup assistant. You have web_search_imdb(query) that "
            "searches Wikipedia and returns the IMDb ID (e.g. tt1375666) directly. "
            "Call it once with a clean query, then return JSON: {imdb_id}."
        )
        kind_hint = "TV series" if kind == "series" else "film"
        year_hint = f" {year}" if year else ""
        search_user = f"Find the IMDb ID for: {title}{year_hint} ({kind_hint})"
        search_raw = chat_json(search_system, search_user, enable_tools=True) or {}
        imdb_id = _normalize_imdb_id(search_raw.get("imdb_id")) or imdb_id

    return imdb_id, metadata


def build_imdb_id(base_imdb_id: str, metadata: MediaMetadata) -> str:
    if not base_imdb_id:
        return ""
    if metadata.kind == "series" and metadata.season is not None and metadata.episode is not None:
        return f"{base_imdb_id}:{metadata.season}:{metadata.episode}"
    return base_imdb_id


def resolve_media_identity(
    torrent_name: str,
    file_name: str,
    explicit_imdb_id: str | None = None,
    forced_kind: str | None = None,
    dry_run: bool = False,
) -> tuple[str, MediaMetadata]:
    if dry_run:
        metadata = MediaMetadata(
            title=torrent_name,
            kind=forced_kind,
            year=None,
            season=None,
            episode=None,
            confidence=0.0,
        )
        explicit = _normalize_imdb_id(explicit_imdb_id)
        return build_imdb_id(explicit, metadata), metadata

    explicit = _normalize_imdb_id(explicit_imdb_id)

    # Fast path: explicit IMDb ID provided (via imdbm:/imdbs: prefix or --explicit-imdb-id)
    # Skip AI entirely — just do a lightweight heuristic parse for title/year/season/episode
    if explicit:
        year_match = re.search(r"\b(19|20)\d{2}\b", torrent_name)
        se_match = re.search(r"[Ss](\d{1,2})[Ee](\d{1,2})", file_name or torrent_name)
        # Strip quality/release tokens to get a clean title
        clean = re.split(r"[\.\s]+(19|20)\d{2}|[\.\s]+\d{3,4}[pP]|[\.\s]+(?:BluRay|WEBRip|HDTV|NF|HEVC|x264|x265|AAC|DDP|ESub)\b", torrent_name, flags=re.IGNORECASE)[0]
        clean = re.sub(r"[._]", " ", clean).strip()
        metadata = MediaMetadata(
            title=clean or torrent_name,
            kind=forced_kind,
            year=int(year_match.group(0)) if year_match else None,
            season=int(se_match.group(1)) if se_match else None,
            episode=int(se_match.group(2)) if se_match else None,
            confidence=1.0,
        )
        return build_imdb_id(explicit, metadata), metadata

    # Slow path: no explicit IMDb ID — use AI + Wikipedia
    imdb_base, metadata = ai_extract_metadata(torrent_name, file_name)
    if forced_kind:
        metadata.kind = forced_kind

    final_base = imdb_base
    return build_imdb_id(final_base, metadata), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve torrent metadata and IMDb identity")
    parser.add_argument("--torrent-name", required=True, help="Torrent display name")
    parser.add_argument("--file-name", required=True, help="Video file name")
    parser.add_argument("--explicit-imdb-id", default=None, help="Explicit IMDb ID, e.g. tt1234567")
    parser.add_argument("--forced-kind", choices=["movie", "series"], default=None, help="Force media kind")
    parser.add_argument("--dry-run", action="store_true", help="Do not call AI; only simulate resolution")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pref_kind, pref_imdb_id, clean_name = parse_prefixed_torrent_name(args.torrent_name)
    forced_kind = args.forced_kind or pref_kind
    explicit_imdb_id = args.explicit_imdb_id or pref_imdb_id

    try:
        imdb_id, metadata = resolve_media_identity(
            torrent_name=clean_name,
            file_name=args.file_name,
            explicit_imdb_id=explicit_imdb_id,
            forced_kind=forced_kind,
            dry_run=args.dry_run,
        )
        print(
            json.dumps(
                {
                    "mode": "dry-run" if args.dry_run else "live",
                    "imdb_id": imdb_id,
                    "metadata": {
                        "title": metadata.title,
                        "kind": metadata.kind,
                        "year": metadata.year,
                        "season": metadata.season,
                        "episode": metadata.episode,
                        "confidence": metadata.confidence,
                    },
                },
                ensure_ascii=True,
            )
        )
    except Exception as exc:
        print(f"ERROR: {exc}")