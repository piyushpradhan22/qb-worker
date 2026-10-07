# VERSION: 3.0
# AUTHORS: Piyush

import html
import json
import re
import urllib.parse
import urllib.request
import sys
from novaprinter import prettyPrinter

LOCAL_FLARESOLVERR = "http://127.0.0.1:8191/v1"
PUBLIC_FLARESOLVERR = "https://flaresolverr.129.159.238.209.sslip.io/v1"

# Set to > 0 (e.g. 3) if you want search() to resolve real magnet links for top N results
# Note: Resolving each 1337x magnet via FlareSolverr takes ~15-25 seconds due to Cloudflare Turnstile.
# When 0 (default for speed), 'link' is the torrent detail page and download_torrent() resolves the magnet on download.
RESOLVE_TOP_MAGNETS = 0

class one337x(object):
    url = "https://1337x.to"
    name = "1337x"
    supported_categories = {
        "all": "0",
        "movies": "Movies",
        "tv": "TV",
        "music": "Music",
        "games": "Games",
        "anime": "Anime",
        "software": "Apps"
    }

    def __init__(self):
        pass

    def retrieve_url(self, url, timeout=60000):
        endpoints = [LOCAL_FLARESOLVERR, PUBLIC_FLARESOLVERR]
        payload = json.dumps({
            "cmd": "request.get",
            "url": url,
            "maxTimeout": timeout
        }).encode("utf-8")

        for endpoint in endpoints:
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=70) as res:
                    response = json.loads(res.read().decode("utf-8"))
                    if response.get("status") == "ok":
                        return response["solution"]["response"]
            except Exception:
                continue
        return ""

    def download_torrent(self, info):
        if info.startswith("magnet:?"):
            print(f"{info} {info}")
            return

        html_content = self.retrieve_url(info)
        idx = html_content.find("magnet:?")
        if idx != -1:
            end_idx = html_content.find('"', idx)
            magnet_url = html.unescape(html_content[idx:end_idx])
            print(f"{magnet_url} {info}")

    def get_magnet(self, desc_link):
        html_content = self.retrieve_url(desc_link, timeout=30000)
        idx = html_content.find("magnet:?")
        if idx != -1:
            end_idx = html_content.find('"', idx)
            return html.unescape(html_content[idx:end_idx])
        return ""

    def search(self, what, cat="all"):
        clean_what = urllib.parse.unquote(what).strip()
        query = urllib.parse.quote_plus(clean_what)

        category = self.supported_categories.get(cat, "0")
        if category != "0":
            query_url = f"{self.url}/category-search/{query}/{category}/1/"
        else:
            query_url = f"{self.url}/search/{query}/1/"

        html_data = self.retrieve_url(query_url)
        if not html_data:
            return

        row_regex = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
        title_regex = re.compile(r'href="(/torrent/[^"]+)">(.*?)</a>')
        seeds_regex = re.compile(r'<td class="coll-2 seeds">(.*?)</td>')
        leech_regex = re.compile(r'<td class="coll-3 leeches">(.*?)</td>')
        size_regex = re.compile(r'<td class="coll-4 size[^"]*">(.*?)<')

        matched_count = 0
        for row in row_regex.findall(html_data):
            if "/torrent/" not in row:
                continue

            title_match = title_regex.search(row)
            if not title_match:
                continue

            desc_link = self.url + title_match.group(1)
            raw_title = re.sub(r"<[^<]+?>", "", title_match.group(2)).strip()
            name = html.unescape(raw_title)

            seeds_match = seeds_regex.search(row)
            seeds = int(seeds_match.group(1)) if seeds_match else 0

            leech_match = leech_regex.search(row)
            leech = int(leech_match.group(1)) if leech_match else 0

            size_match = size_regex.search(row)
            size = size_match.group(1).strip() if size_match else "0 B"

            # Determine link to report
            link = desc_link
            if RESOLVE_TOP_MAGNETS > 0 and matched_count < RESOLVE_TOP_MAGNETS:
                resolved_magnet = self.get_magnet(desc_link)
                if resolved_magnet:
                    link = resolved_magnet

            result = {
                "link": link,
                "name": name,
                "size": size,
                "seeds": seeds,
                "leech": leech,
                "engine_url": self.url,
                "desc_link": desc_link
            }
            prettyPrinter(result)
            sys.stdout.flush()
            matched_count += 1
