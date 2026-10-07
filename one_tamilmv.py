# VERSION: 6.0
# AUTHORS: Piyush

import concurrent.futures
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from novaprinter import prettyPrinter

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

class one_tamilmv(object):
    url = "https://www.1tamilmv.capital"
    name = "1TamilMV"
    supported_categories = {
        "all": "0",
        "movies": "Movies",
        "tv": "TV"
    }

    def __init__(self):
        pass

    def retrieve_url(self, url, timeout=12):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return res.read().decode("utf-8", "ignore")
        except Exception:
            return ""

    def download_torrent(self, info):
        if info.startswith("magnet:?"):
            print(f"{info} {info}")
            return
        topic_html = self.retrieve_url(info)
        idx = topic_html.find("magnet:?")
        if idx != -1:
            end_idx = topic_html.find('"', idx)
            magnet_url = html.unescape(topic_html[idx:end_idx])
            print(f"{magnet_url} {info}")

    def _process_topic(self, tid, topic_title):
        # Slugify title for topic URL
        base = re.sub(r"[^a-z0-9\s-]+", " ", topic_title.lower()).strip()
        slug = re.sub(r"\s+", "-", base)[:80] or "t"
        topic_url = f"{self.url}/index.php?/forums/topic/{tid}-{slug}/"

        html_content = self.retrieve_url(topic_url)
        if not html_content:
            fallback_url = f"{self.url}/index.php?/topic/{tid}-entry/"
            html_content = self.retrieve_url(fallback_url)
            if not html_content:
                return []

        # Find all magnet links in the topic
        magnets = re.findall(r'href=["\'](magnet:\?[^"\']+)["\']', html_content)
        if not magnets:
            return []

        results = []
        seen_magnets = set()

        for raw_magnet in magnets:
            magnet = html.unescape(raw_magnet)
            if magnet in seen_magnets:
                continue
            seen_magnets.add(magnet)

            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(magnet).query)
            dn = parsed.get("dn", [""])[0]

            clean_name = re.sub(r"^www\.[^\s-]+\s*-\s*", "", dn).strip()
            if not clean_name:
                clean_name = topic_title

            # Extract size (look for GB / MB pattern, ignoring Kbps)
            size_matches = re.findall(r"\b(\d+(?:\.\d+)?\s*(?:GB|MB|KB|TB))\b(?!ps)", clean_name, re.IGNORECASE)
            size_str = size_matches[-1] if size_matches else "0 B"

            results.append({
                "link": magnet,
                "name": clean_name,
                "size": size_str,
                "seeds": -1,
                "leech": -1,
                "engine_url": self.url,
                "desc_link": topic_url
            })

        return results

    def search(self, what, cat="all"):
        clean_what = urllib.parse.unquote(what).strip()
        query = urllib.parse.quote_plus(clean_what)
        api_url = f"{self.url}/search/api/search.php?q={query}&page=1&per_page=6"

        response_text = self.retrieve_url(api_url)
        if not response_text:
            return

        try:
            data = json.loads(response_text)
        except Exception:
            return

        topics = data.get("results", [])
        if not topics:
            return

        # Fetch top topic pages concurrently to extract magnet links
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_topic = {
                executor.submit(self._process_topic, topic.get("tid"), topic.get("title", "")): topic
                for topic in topics[:6]
            }

            for future in concurrent.futures.as_completed(future_to_topic):
                try:
                    topic_results = future.result()
                    for res in topic_results:
                        prettyPrinter(res)
                        sys.stdout.flush()
                except Exception:
                    pass
