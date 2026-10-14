# VERSION: 5.0
# AUTHORS: Piyush

import concurrent.futures
import html
import json
import re
import ssl
import sys
import threading
import time
import urllib.parse
import urllib.request

try:
    from novaprinter import prettyPrinter
except ImportError:
    def prettyPrinter(d):
        print(f"{d.get('link')}|{d.get('name')}|{d.get('size')}|{d.get('seeds')}|{d.get('leech')}|{d.get('engine_url')}|{d.get('desc_link')}")

LOCAL_FLARESOLVERR = "http://127.0.0.1:8191/v1"
PUBLIC_FLARESOLVERR = "https://flaresolverr.129.159.238.209.sslip.io/v1"

# Maximum number of top results to resolve magnet links for
MAX_RESULTS = 6

try:
    SSL_CTX = ssl._create_unverified_context()
except Exception:
    SSL_CTX = None


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
        self.active_endpoint = None
        self.cached_headers = {}
        self.flaresolverr_lock = threading.Lock()

    def _flaresolverr_post(self, payload_dict, timeout=60):
        endpoints = [self.active_endpoint] if self.active_endpoint else [LOCAL_FLARESOLVERR, PUBLIC_FLARESOLVERR]
        payload = json.dumps(payload_dict).encode("utf-8")

        for endpoint in endpoints:
            if not endpoint:
                continue
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=timeout + 5, context=SSL_CTX) as res:
                    data = json.loads(res.read().decode("utf-8"))
                    if data.get("status") == "ok":
                        self.active_endpoint = endpoint
                        return data
            except Exception:
                continue

        # If cached active endpoint failed, retry the alternate endpoint
        if self.active_endpoint:
            self.active_endpoint = None
            return self._flaresolverr_post(payload_dict, timeout)

        return None

    def _create_session(self):
        session_id = f"qbt_1337x_{int(time.time() * 1000)}"
        res = self._flaresolverr_post({"cmd": "sessions.create", "session": session_id}, timeout=15)
        if res and res.get("status") == "ok":
            return session_id
        return None

    def _destroy_session(self, session_id):
        if session_id:
            try:
                self._flaresolverr_post({"cmd": "sessions.destroy", "session": session_id}, timeout=10)
            except Exception:
                pass

    def retrieve_url(self, url, timeout=60000):
        res = self._flaresolverr_post({"cmd": "request.get", "url": url, "maxTimeout": timeout}, timeout=timeout // 1000)
        if res:
            return res.get("solution", {}).get("response", "")
        return ""

    def _extract_magnet(self, html_content):
        if not html_content:
            return ""
        match = re.search(r'href=["\'](magnet:\?[^"\']+)["\']', html_content)
        if not match:
            match = re.search(r'(magnet:\?[^\s"\'<>]+)', html_content)
        if match:
            raw_magnet = match.group(1)
            return html.unescape(raw_magnet).strip().replace(" ", "%20")
        return ""

    def get_magnet(self, desc_link, session_id=None):
        # 1. Try fast direct HTTP fetch with clearance cookies and User-Agent from FlareSolverr
        if self.cached_headers:
            try:
                req = urllib.request.Request(desc_link, headers=self.cached_headers)
                with urllib.request.urlopen(req, timeout=6, context=SSL_CTX) as res:
                    html_content = res.read().decode("utf-8", "ignore")
                    magnet = self._extract_magnet(html_content)
                    if magnet:
                        return magnet
            except Exception:
                pass

        # 2. Fall back to FlareSolverr (using thread lock to avoid concurrent Puppeteer session conflicts)
        with self.flaresolverr_lock:
            payload = {
                "cmd": "request.get",
                "url": desc_link,
                "maxTimeout": 45000
            }
            if session_id:
                payload["session"] = session_id

            res = self._flaresolverr_post(payload, timeout=50)
            if res:
                html_content = res.get("solution", {}).get("response", "")
                return self._extract_magnet(html_content)

        return ""

    def download_torrent(self, info):
        clean_info = info.strip()
        if clean_info.startswith("magnet:?"):
            url_safe = clean_info.replace(" ", "%20")
            print(f"{url_safe} {url_safe}")
            return

        session_id = self._create_session()
        try:
            magnet = self.get_magnet(clean_info, session_id=session_id)
            if magnet:
                url_safe_magnet = magnet.strip().replace(" ", "%20")
                url_safe_info = clean_info.replace(" ", "%20")
                print(f"{url_safe_magnet} {url_safe_info}")
        finally:
            self._destroy_session(session_id)

    def search(self, what, cat="all"):
        clean_what = urllib.parse.unquote(what).strip()
        query = urllib.parse.quote_plus(clean_what)

        category = self.supported_categories.get(cat, "0")
        if category != "0":
            query_url = f"{self.url}/category-search/{query}/{category}/1/"
        else:
            query_url = f"{self.url}/search/{query}/1/"

        session_id = self._create_session()
        try:
            payload = {
                "cmd": "request.get",
                "url": query_url,
                "maxTimeout": 60000
            }
            if session_id:
                payload["session"] = session_id

            search_res = self._flaresolverr_post(payload, timeout=65)
            if not search_res:
                return

            solution = search_res.get("solution", {})
            html_data = solution.get("response", "")
            if not html_data:
                return

            cookies = solution.get("cookies", [])
            user_agent = solution.get("userAgent", "")
            cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies if 'name' in c and 'value' in c)
            self.cached_headers = {
                "User-Agent": user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": self.url + "/"
            }
            if cookie_str:
                self.cached_headers["Cookie"] = cookie_str

            row_regex = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
            title_regex = re.compile(r'href="(/torrent/[^"]+)">(.*?)</a>')
            seeds_regex = re.compile(r'<td class="coll-2 seeds">(.*?)</td>')
            leech_regex = re.compile(r'<td class="coll-3 leeches">(.*?)</td>')
            size_regex = re.compile(r'<td class="coll-4 size[^"]*">(.*?)<')

            candidates = []
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

                candidates.append({
                    "name": name,
                    "seeds": seeds,
                    "leech": leech,
                    "size": size,
                    "desc_link": desc_link
                })

            if not candidates:
                return

            # Sort candidates by seeders descending so the most active torrents are resolved first
            candidates.sort(key=lambda x: x["seeds"], reverse=True)
            top_candidates = candidates[:MAX_RESULTS]

            def process_candidate(cand):
                magnet = self.get_magnet(cand["desc_link"], session_id=session_id)
                if not magnet:
                    return None
                return {
                    "link": magnet,
                    "name": cand["name"],
                    "size": cand["size"],
                    "seeds": cand["seeds"],
                    "leech": cand["leech"],
                    "engine_url": self.url,
                    "desc_link": cand["desc_link"]
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_cand = {executor.submit(process_candidate, cand): cand for cand in top_candidates}
                for future in concurrent.futures.as_completed(future_to_cand):
                    try:
                        res = future.result()
                        if res:
                            prettyPrinter(res)
                            sys.stdout.flush()
                    except Exception:
                        pass
        finally:
            self._destroy_session(session_id)
