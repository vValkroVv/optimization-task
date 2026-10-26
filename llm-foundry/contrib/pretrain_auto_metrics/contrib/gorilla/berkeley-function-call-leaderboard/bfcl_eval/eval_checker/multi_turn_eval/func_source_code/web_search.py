import os
import random
import re
import threading
import time
import warnings
from abc import ABC, abstractmethod
from ast import literal_eval
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import html2text
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None  # type: ignore

ERROR_TEMPLATES = [
    "503 Server Error: Service Unavailable for url: {url}",
    "429 Client Error: Too Many Requests for url: {url}",
    "403 Client Error: Forbidden for url: {url}",
    (
        "HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: {path} "
        "(Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x{id1:x}>, "
        "'Connection to {host} timed out. (connect timeout=5)'))"
    ),
    "HTTPSConnectionPool(host='{host}', port=443): Read timed out. (read timeout=5)",
    (
        "Max retries exceeded with url: {path} "
        "(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x{id2:x}>: "
        "Failed to establish a new connection: [Errno -2] Name or service not known'))"
    ),
]


def _unescape_literal_json_unicode(s: str) -> str:
    """
    Convert literal \\uXXXX / \\UXXXXXXXX sequences into real Unicode.
    Some search APIs return snippets that were JSON-stringified twice, which surfaces as
    visible backslash-u escapes in the model-facing text.
    """
    if not s or ("\\u" not in s and "\\U" not in s):
        return s

    def _u4(m: re.Match) -> str:
        try:
            return chr(int(m.group(1), 16))
        except (ValueError, OverflowError):
            return m.group(0)

    def _u8(m: re.Match) -> str:
        try:
            return chr(int(m.group(1), 16))
        except (ValueError, OverflowError):
            return m.group(0)

    out = re.sub(r"\\u([0-9a-fA-F]{4})", _u4, s)
    out = re.sub(r"\\U([0-9a-fA-F]{8})", _u8, out)
    return out


def _text_from_response(response: requests.Response) -> str:
    """Decode body with a sensible charset (fixes Cyrillic/HTML when charset is missing/wrong)."""
    if response.content and (
        not response.encoding
        or response.encoding.lower() in ("iso-8859-1", "windows-1252")
    ):
        apparent = response.apparent_encoding
        if apparent:
            response.encoding = apparent
    try:
        return response.text
    except Exception:
        return response.content.decode("utf-8", errors="replace")


def _fetch_url_with_fallbacks(url: str, timeout: float = 20.0) -> requests.Response:
    """
    Fetch HTML with same-origin Referer and without misleading Sec-Fetch headers from Google.
    Retry with a minimal curl-like User-Agent on 403/404 when the origin responds differently.
    """
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme and parsed.netloc else ""

    accept_enc = "gzip, deflate"
    try:
        import brotli  # noqa: F401

        accept_enc = "gzip, deflate, br"
    except ImportError:
        pass

    primary = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8,*;q=0.7",
        "Accept-Encoding": accept_enc,
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        **({"Referer": origin} if origin else {}),
    }

    fallback = {
        "User-Agent": "curl/8.5.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9,*;q=0.8",
    }

    resp = requests.get(url, headers=primary, timeout=timeout, allow_redirects=True)
    if resp.status_code in (403, 404) or resp.status_code >= 500:
        resp2 = requests.get(url, headers=fallback, timeout=timeout, allow_redirects=True)
        if resp2.status_code < 400 or resp.status_code >= 500:
            return resp2
    return resp



def _ensure_project_dotenv_loaded() -> None:
    """Load repo-root `.env` once so Sber settings are available without relying on import order."""
    if getattr(_ensure_project_dotenv_loaded, "_done", False):
        return
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[4] / ".env"
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        pass
    setattr(_ensure_project_dotenv_loaded, "_done", True)


class BaseWebSearchAPI(ABC):
    """
    Shared Web Search API surface: search_engine_query + fetch_url_content.
    Subclasses implement search_engine_query using different HTTP backends; all normalize to
    the same result shape as the original BFCL WebSearchAPI (title / href / body).
    """

    def __init__(self):
        self._api_description = (
            "This tool belongs to the Web Search API category. It provides functions to search "
            "the web and browse search results."
        )
        self.show_snippet = True
        self._random = random.Random(337)
        self._rng = random.Random(1053)

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        self.show_snippet = initial_config["show_snippet"]

    def _append_result(
        self,
        results: list,
        title: str,
        href: str,
        body: str,
    ) -> None:
        title = _unescape_literal_json_unicode(title or "")
        body = _unescape_literal_json_unicode(body or "")
        if self.show_snippet:
            results.append({"title": title, "href": href, "body": body})
        else:
            results.append({"title": title, "href": href})

    @abstractmethod
    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ):
        """
        This function queries the search engine for the provided keywords and region.

        Args:
            keywords (str): The keywords to search for.
            max_results (int, optional): The maximum number of search results to return. Defaults to 10.
            region (str, optional): The region to search in. Defaults to "wt-wt". Possible values include:
                - xa-ar for Arabia
                - xa-en for Arabia (en)
                - ar-es for Argentina
                - au-en for Australia
                - at-de for Austria
                - be-fr for Belgium (fr)
                - be-nl for Belgium (nl)
                - br-pt for Brazil
                - bg-bg for Bulgaria
                - ca-en for Canada
                - ca-fr for Canada (fr)
                - ct-ca for Catalan
                - cl-es for Chile
                - cn-zh for China
                - co-es for Colombia
                - hr-hr for Croatia
                - cz-cs for Czech Republic
                - dk-da for Denmark
                - ee-et for Estonia
                - fi-fi for Finland
                - fr-fr for France
                - de-de for Germany
                - gr-el for Greece
                - hk-tzh for Hong Kong
                - hu-hu for Hungary
                - in-en for India
                - id-id for Indonesia
                - id-en for Indonesia (en)
                - ie-en for Ireland
                - il-he for Israel
                - it-it for Italy
                - jp-jp for Japan
                - kr-kr for Korea
                - lv-lv for Latvia
                - lt-lt for Lithuania
                - xl-es for Latin America
                - my-ms for Malaysia
                - my-en for Malaysia (en)
                - mx-es for Mexico
                - nl-nl for Netherlands
                - nz-en for New Zealand
                - no-no for Norway
                - pe-es for Peru
                - ph-en for Philippines
                - ph-tl for Philippines (tl)
                - pl-pl for Poland
                - pt-pt for Portugal
                - ro-ro for Romania
                - ru-ru for Russia
                - sg-en for Singapore
                - sk-sk for Slovak Republic
                - sl-sl for Slovenia
                - za-en for South Africa
                - es-es for Spain
                - se-sv for Sweden
                - ch-de for Switzerland (de)
                - ch-fr for Switzerland (fr)
                - ch-it for Switzerland (it)
                - tw-tzh for Taiwan
                - th-th for Thailand
                - tr-tr for Turkey
                - ua-uk for Ukraine
                - uk-en for United Kingdom
                - us-en for United States
                - ue-es for United States (es)
                - ve-es for Venezuela
                - vn-vi for Vietnam
                - wt-wt for No region

        Returns:
            list: A list of search result dictionaries, each containing information such as:
            - 'title' (str): The title of the search result.
            - 'href' (str): The URL of the search result.
            - 'body' (str): A brief description or snippet from the search result.
        """
        raise NotImplementedError

    def fetch_url_content(self, url: str, mode: str = "raw") -> str:
        """
        This function retrieves content from the provided URL and processes it based on the selected mode.

        Args:
            url (str): The URL to fetch content from. Must start with 'http://' or 'https://'.
            mode (str, optional): The mode to process the fetched content. Defaults to "raw".
                Supported modes are:
                    - "raw": Returns the raw HTML content.
                    - "markdown": Converts raw HTML content to Markdown format for better readability, using html2text.
                    - "truncate": Extracts and cleans text by removing scripts, styles, and extraneous whitespace.
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        try:
            response = _fetch_url_with_fallbacks(url, timeout=20.0)
            response.raise_for_status()
            html_text = _text_from_response(response)

            if mode == "raw":
                return {"content": html_text}

            elif mode == "markdown":
                converter = html2text.HTML2Text()
                markdown = converter.handle(html_text)
                return {"content": _unescape_literal_json_unicode(markdown)}

            elif mode == "truncate":
                soup = BeautifulSoup(html_text, "html.parser")

                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()

                text = soup.get_text(separator="\n", strip=True)
                return {"content": _unescape_literal_json_unicode(text)}
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        except Exception as e:
            return {"error": f"An error occurred while fetching {url}: {str(e)}"}

    def _fake_requests_get_error_msg(self, url: str) -> str:
        parsed = urlparse(url)

        context = {
            "url": url,
            "host": parsed.hostname or "unknown",
            "path": parsed.path or "/",
            "id1": self._rng.randrange(0x10000000, 0xFFFFFFFF),
            "id2": self._rng.randrange(0x10000000, 0xFFFFFFFF),
        }

        template = self._rng.choice(ERROR_TEMPLATES)

        return template.format(**context)


class SerpAPIWebSearch(BaseWebSearchAPI):
    """DuckDuckGo results via SerpAPI (google-search-results), same as the original implementation."""

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ):
        backoff = 2
        params = {
            "engine": "duckduckgo",
            "q": keywords,
            "kl": region,
            "api_key": os.getenv("SERPAPI_API_KEY"),
        }

        while True:
            try:
                search = GoogleSearch(params)
                search_results = search.get_dict()
            except Exception as e:
                if "429" in str(e):
                    wait_time = backoff + random.uniform(0, backoff)
                    error_block = (
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Received 429 from SerpAPI. The number of requests sent using this API key exceeds the hourly throughput limit OR your account has run out of searches. Retrying in {wait_time:.1f} seconds…"
                        + "*" * 100
                    )
                    print(error_block)
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)
                    continue
                else:
                    error_block = (
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Error from SerpAPI: {str(e)}. This is not a rate-limit error, so it will not be retried."
                        + "*" * 100
                    )
                    print(error_block)
                    return {"error": str(e)}

            if "error" in search_results and "429" in str(search_results["error"]):
                wait_time = backoff + random.uniform(0, backoff)
                error_block = (
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI] Received 429 from SerpAPI. The number of requests sent using this API key exceeds the hourly throughput limit OR your account has run out of searches. Retrying in {wait_time:.1f} seconds…"
                    + "*" * 100
                )
                print(error_block)
                time.sleep(wait_time)
                backoff = min(backoff * 2, 120)
                continue

            break

        if "organic_results" not in search_results:
            return {
                "error": "Failed to retrieve the search results from server. Please try again later."
            }

        organic = search_results["organic_results"]
        results = []
        mr = max_results or 10
        for result in organic[:mr]:
            self._append_result(
                results,
                result["title"],
                result["link"],
                result.get("snippet", "") or "",
            )

        return results


class DuckDuckGoFreeWebSearch(BaseWebSearchAPI):
    """
    Free DuckDuckGo text search via duckduckgo_search (DDGS), matching BrowseComp-Plus/web_client.py.
    """

    @staticmethod
    def _ddgs_search_sync(query: str, max_results: int, region: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with DDGS() as ddgs:
                return list(ddgs.text(query, region=region, max_results=max_results))

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ):
        if DDGS is None:
            return {
                "error": (
                    "The 'duckduckgo_search' package is not installed. "
                    "Install with: pip install duckduckgo-search"
                )
            }

        mr = max(1, int(max_results or 10))
        reg = region or "wt-wt"

        last_err = None
        for attempt in range(3):
            try:
                raw = self._ddgs_search_sync(keywords, mr, reg)
                results = []
                for r in raw or []:
                    title = str(r.get("title", "") or "")
                    href = str(r.get("href", "") or "")
                    body = str(r.get("body", "") or "")
                    self._append_result(results, title, href, body)
                return results
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))

        return {"error": f"Search failed after 3 attempts: {last_err}"}


_sber_search_post_semaphore: Optional[threading.Semaphore] = None
_sber_search_post_semaphore_lock = threading.Lock()


def _get_sber_search_post_semaphore() -> threading.Semaphore:
    """
    Limit concurrent Sber search POSTs across all threads and all WebSearchAPI instances in this process.
    Default 25; override with SBER_SEARCH_MAX_IN_FLIGHT in .env (loaded before first use).
    """
    global _sber_search_post_semaphore
    if _sber_search_post_semaphore is not None:
        return _sber_search_post_semaphore
    with _sber_search_post_semaphore_lock:
        if _sber_search_post_semaphore is not None:
            return _sber_search_post_semaphore
        _ensure_project_dotenv_loaded()
        raw = (os.environ.get("SBER_SEARCH_MAX_IN_FLIGHT") or "5").strip()
        try:
            limit = int(raw)
        except ValueError:
            limit = 25
        limit = max(1, min(limit, 256))
        _sber_search_post_semaphore = threading.Semaphore(limit)
        return _sber_search_post_semaphore



_DEFAULT_RLI_GATEWAY_URL = "https://gym.static.space/v1/tools/execute"

_rli_gateway_reachable: Optional[bool] = None
_rli_gateway_reachable_lock = threading.Lock()


def _rli_gateway_available(gateway_url: str) -> bool:
    """
    The RLI api-gateway is only routable from the clusters that have no direct internet access,
    so its reachability is what decides the backend. Probed once per process; any HTTP answer
    (including 404/405 to the HEAD) counts as reachable, only transport errors do not.
    """
    global _rli_gateway_reachable
    if _rli_gateway_reachable is not None:
        return _rli_gateway_reachable
    with _rli_gateway_reachable_lock:
        if _rli_gateway_reachable is None:
            try:
                requests.head(gateway_url, timeout=5.0, verify=False)
                _rli_gateway_reachable = True
            except requests.RequestException:
                _rli_gateway_reachable = False
    return _rli_gateway_reachable


class SberWebSearch(BaseWebSearchAPI):
    """
    Sber GigaSearch retrieval API, matching BrowseComp-Plus/sber_web_client.py request/response shape.
    Configure via berkeley-function-call-leaderboard/.env: SBER_SEARCH_URL, SBER_SEARCH_API_KEY (or SBER_API_KEY).
    Concurrent search POSTs are capped process-wide (default 25); set SBER_SEARCH_MAX_IN_FLIGHT to override.
    """

    def __init__(self, api_key: Optional[str] = None, search_url: Optional[str] = None):
        super().__init__()
        _ensure_project_dotenv_loaded()
        self._sber_search_url = (search_url or os.environ.get("SBER_SEARCH_URL", "") or "").strip()
        self._sber_api_key = (
            api_key or os.environ.get("SBER_SEARCH_API_KEY") or os.environ.get("SBER_API_KEY") or ""
        ).strip()
        self._rli_gateway_url = (
            os.environ.get("SBER_RLI_GATEWAY_URL") or _DEFAULT_RLI_GATEWAY_URL
        ).strip()
        self._rli_tool_name = (os.environ.get("SBER_RLI_TOOL_NAME") or "websearch").strip()
        self._rli_tool_version = (
            os.environ.get("SBER_RLI_TOOL_VERSION") or "2.2.0"
        ).strip()
        # fetch_url_content goes through a separate RLI tool (get_url_content).
        self._rli_fetch_tool_name = (
            os.environ.get("SBER_RLI_FETCH_TOOL_NAME") or "get_url_content"
        ).strip()
        self._rli_fetch_tool_version = (
            os.environ.get("SBER_RLI_FETCH_TOOL_VERSION") or "1.1.0"
        ).strip()

    @property
    def _use_rli(self) -> bool:
        return _rli_gateway_available(self._rli_gateway_url)

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ):
        _ = region

        if self._use_rli:
            return self._search_engine_query_rli(keywords, max_results)

        if not self._sber_search_url:
            return {"error": "SBER_SEARCH_URL is not set. Add it to berkeley-function-call-leaderboard/.env."}
        if not self._sber_api_key:
            return {
                "error": "Sber API key is not set. Set SBER_SEARCH_API_KEY or SBER_API_KEY in .env."
            }

        headers = {
            "Content-Type": "application/json",
            "X-Forwarded-For": "127.0.0.1",
            "Application-Agent": (
                "Mozilla/5.0 (platform; rv:gecko-version) "
                "Gecko/gecko-trail Firefox/firefox-version"
            ),
            "Accept": "application/json",
            "Authorization": self._sber_api_key,
        }
        mr = max(1, int(max_results or 10))
        payload = {
            "query": keywords,
            "search_mode": "web_search",
            "additional_data": {"doc_count": mr},
        }

        last_err = None
        for attempt in range(3):
            try:
                with _get_sber_search_post_semaphore():
                    resp = requests.post(
                        self._sber_search_url,
                        headers=headers,
                        json=payload,
                        timeout=30.0,
                        verify=False,
                    )
                if resp.status_code != 200:
                    if attempt == 2:
                        return {"error": f"Search API returned HTTP {resp.status_code}"}
                    time.sleep(1.0 * (attempt + 1))
                    continue

                data = resp.json()
                api_payload = data.get("payload", {})

                if api_payload.get("status") != "success":
                    return {"error": f"Search API status: {api_payload.get('status')}"}

                documents = api_payload.get("documents", []) or []
                sources = data.get("sources", {}) or {}
                results = []
                for doc in documents:
                    ref_key = str(doc.get("reference_keyword", ""))
                    source = sources.get(ref_key, {}) or {}
                    title = str(source.get("title", "") or "")
                    url = str(source.get("url", "") or "")
                    snippet = str(doc.get("text", "") or "")
                    self._append_result(results, title, url, snippet)
                return results
            except Exception as e:
                last_err = e
                if attempt == 2:
                    return {"error": f"Search failed after 3 attempts: {e}"}
                time.sleep(1.0 * (attempt + 1))

        return {"error": f"Search failed after 3 attempts: {last_err}"}

    def fetch_url_content(self, url: str, mode: str = "raw") -> str:
        """
        When the RLI api-gateway is reachable the page fetch is proxied through it
        (get_url_content tool); otherwise it falls back to the direct HTTP fetch in the base
        class. The gateway already returns cleaned page text, so ``mode`` is accepted for API
        compatibility but does not change the RLI output.
        """
        if not self._use_rli:
            return super().fetch_url_content(url, mode=mode)
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")
        return self._fetch_url_content_rli(url)

    def _fetch_url_content_rli(self, url: str):
        if not self._rli_gateway_url:
            return {
                "error": f"An error occurred while fetching {url}: "
                "SBER_RLI_GATEWAY_URL is not set."
            }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._sber_api_key:
            headers["Authorization"] = self._sber_api_key

        payload = {
            "tool": {
                "name": self._rli_fetch_tool_name,
                "version": self._rli_fetch_tool_version,
            },
            "input": {
                "links": [{"url": url}],
                "timeout": 15,
                "max_content_length": 20000,
            },
        }

        for attempt in range(3):
            try:
                with _get_sber_search_post_semaphore():
                    resp = requests.post(
                        self._rli_gateway_url,
                        headers=headers,
                        json=payload,
                        timeout=60.0,
                        verify=False,
                    )
                if resp.status_code != 200:
                    if attempt == 2:
                        return {
                            "error": f"An error occurred while fetching {url}: "
                            f"gateway returned HTTP {resp.status_code}"
                        }
                    time.sleep(1.0 * (attempt + 1))
                    continue

                return self._parse_rli_fetch_response(resp.json(), url)
            except Exception as e:
                if attempt == 2:
                    return {"error": f"An error occurred while fetching {url}: {e}"}
                time.sleep(1.0 * (attempt + 1))

        return {"error": f"An error occurred while fetching {url}: unknown error"}

    def _parse_rli_fetch_response(self, data: object, url: str):
        """
        Extract page content from the get_url_content gateway response.

        The gateway nests the payload as a stringified Python literal under ``output.text``,
        shaped like ``{'status': 'success', 'content': [{'content': "...", 'url': "..."}]}``.
        We literal_eval it (which also resolves the embedded \\n / \\xa0 escapes) and return the
        same ``{"content": <text>}`` shape as the direct fetch, preferring the entry whose url
        matches the request.
        """
        entries = self._rli_extract_fetch_entries(data)
        if not entries:
            metadata = self._rli_find_metadata(data)
            status = metadata.get("status") if isinstance(metadata, dict) else None
            if status not in (None, "success"):
                return {
                    "error": f"An error occurred while fetching {url}: gateway status {status}"
                }
            return {"error": f"An error occurred while fetching {url}: empty gateway response"}

        chosen = None
        for entry in entries:
            if str(entry.get("url", "")) == url:
                chosen = entry
                break
        if chosen is None:
            chosen = entries[0]

        content = str(chosen.get("content", "") or "")
        return {"content": _unescape_literal_json_unicode(content)}

    @classmethod
    def _rli_extract_fetch_entries(cls, obj: object) -> list:
        """Collect content entries ({'content': ..., 'url': ...}) from anywhere in the response."""
        entries: list = []
        seen: set = set()
        cls._rli_walk_fetch_entries(obj, entries, seen)
        return entries

    @classmethod
    def _rli_walk_fetch_entries(cls, obj: object, entries: list, seen: set) -> None:
        if isinstance(obj, dict):
            if isinstance(obj.get("content"), str) and "url" in obj:
                key = id(obj)
                if key not in seen:
                    seen.add(key)
                    entries.append(obj)
            for value in obj.values():
                cls._rli_walk_fetch_entries(value, entries, seen)
        elif isinstance(obj, list):
            for value in obj:
                cls._rli_walk_fetch_entries(value, entries, seen)
        elif isinstance(obj, str):
            parsed = cls._rli_try_literal_eval_fetch(obj)
            if isinstance(parsed, (dict, list)):
                cls._rli_walk_fetch_entries(parsed, entries, seen)

    @staticmethod
    def _rli_try_literal_eval_fetch(s: str):
        if not s or "content" not in s:
            return None
        try:
            return literal_eval(s)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return None

    def _search_engine_query_rli(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
    ):
        """
        Query Sber search through the RLI api-gateway.

        POSTs to SBER_RLI_GATEWAY_URL with the retrieval endpoint forwarded in the body's
        ``input.url``. The gateway nests the actual search payload, returning the documents
        as a stringified Python literal (with embedded \\xa0-style escapes) plus a ``metadata``
        block that carries the per-citation ``sources`` mapping. We literal_eval the document
        string and normalise to the same {title, href, body} shape as the direct backend.
        """
        if not self._rli_gateway_url:
            return {
                "error": "SBER_RLI_GATEWAY_URL is not set; cannot reach the RLI api-gateway."
            }
        if not self._sber_search_url:
            return {
                "error": "SBER_SEARCH_URL (retrieval URL) is not set; required as the RLI gateway target."
            }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._sber_api_key:
            headers["Authorization"] = self._sber_api_key

        mr = max(1, int(max_results or 10))
        payload = {
            "tool": {"name": self._rli_tool_name, "version": self._rli_tool_version},
            "input": {
                "url": self._sber_search_url,
                "query": keywords,
                "search_mode": "web_search",
                "doc_count": mr,
                "timeout": 5,
                "max_content_length": 20000,
            },
        }

        last_err = None
        for attempt in range(3):
            try:
                with _get_sber_search_post_semaphore():
                    resp = requests.post(
                        self._rli_gateway_url,
                        headers=headers,
                        json=payload,
                        timeout=60.0,
                        verify=False,
                    )
                if resp.status_code != 200:
                    if attempt == 2:
                        return {"error": f"Search API returned HTTP {resp.status_code}"}
                    time.sleep(1.0 * (attempt + 1))
                    continue

                return self._parse_rli_response(resp.json())
            except Exception as e:
                last_err = e
                if attempt == 2:
                    return {"error": f"Search failed after 3 attempts: {e}"}
                time.sleep(1.0 * (attempt + 1))

        return {"error": f"Search failed after 3 attempts: {last_err}"}

    def _parse_rli_response(self, data: object):
        metadata = self._rli_find_metadata(data)
        sources = (metadata.get("sources") if isinstance(metadata, dict) else None) or {}

        documents = self._rli_extract_documents(data)
        results: list = []
        for doc in documents:
            text = str(doc.get("text", "") or "")
            title = str(doc.get("title", "") or "")
            href = ""
            source = self._rli_lookup_source(sources, doc.get("cite_index"))
            if source:
                href = str(source.get("url", "") or "")
                if not title:
                    title = str(source.get("title", "") or "")
            self._append_result(results, title, href, text)

        if not results:
            status = metadata.get("status") if isinstance(metadata, dict) else None
            if status not in (None, "success"):
                return {"error": f"Search API status: {status}"}
        return results

    @staticmethod
    def _rli_find_metadata(obj: object) -> dict:
        """Locate the metadata dict (the one carrying ``sources``) anywhere in the response."""
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if "sources" in cur or "document_count" in cur:
                    return cur
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
        return {}

    @classmethod
    def _rli_extract_documents(cls, obj: object) -> list:
        documents: list = []
        seen: set = set()
        cls._rli_walk_documents(obj, documents, seen)
        return documents

    @classmethod
    def _rli_walk_documents(cls, obj: object, documents: list, seen: set) -> None:
        if isinstance(obj, dict):
            if isinstance(obj.get("text"), str) and (
                "cite_index" in obj or "title" in obj
            ):
                key = id(obj)
                if key not in seen:
                    seen.add(key)
                    documents.append(obj)
            for value in obj.values():
                cls._rli_walk_documents(value, documents, seen)
        elif isinstance(obj, list):
            for value in obj:
                cls._rli_walk_documents(value, documents, seen)
        elif isinstance(obj, str):
            parsed = cls._rli_try_literal_eval(obj)
            if isinstance(parsed, (dict, list)):
                cls._rli_walk_documents(parsed, documents, seen)

    @staticmethod
    def _rli_try_literal_eval(s: str):
        """literal_eval a stringified document payload; returns None when it isn't one."""
        if not s or ("text" not in s and "cite_index" not in s):
            return None
        try:
            return literal_eval(s)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return None

    @staticmethod
    def _rli_lookup_source(sources: dict, cite_index: object) -> dict:
        """
        Map a document's ``cite_index`` to its source entry. The ``sources`` mapping is keyed
        "0", "1", ... while ``cite_index`` is 1-based (e.g. cite_index 5 -> sources["4"] in a
        5-document response), so we try the 1-based -> 0-based conversion first and fall back to
        the raw index.
        """
        if not isinstance(sources, dict) or cite_index is None:
            return {}
        candidates: list[str] = []
        try:
            ci = int(cite_index)
            candidates = [str(ci - 1), str(ci)]
        except (TypeError, ValueError):
            candidates = [str(cite_index)]
        for key in candidates:
            value = sources.get(key)
            if isinstance(value, dict):
                return value
        return {}


_WEB_SEARCH_BACKENDS = {
    "serpapi": SerpAPIWebSearch,
    "duckduckgo": DuckDuckGoFreeWebSearch,
    "sber": SberWebSearch,
}


def configure_web_search_backend(backend: str) -> None:
    """Set module-level WebSearchAPI to the chosen backend class."""
    global WebSearchAPI
    key = (backend or "serpapi").strip().lower()
    if key not in _WEB_SEARCH_BACKENDS:
        allowed = ", ".join(sorted(_WEB_SEARCH_BACKENDS))
        raise ValueError(f"Unknown web search backend {backend!r}. Choose one of: {allowed}")
    WebSearchAPI = _WEB_SEARCH_BACKENDS[key]


WebSearchAPI = SerpAPIWebSearch
