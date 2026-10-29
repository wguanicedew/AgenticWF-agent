"""
The search providers behind `web_search`, one function each: (client, provider, query, n) →
results. Endpoints are the providers' public APIs unless `[web_search.<name>] url` says otherwise.

    duckduckgo   no key; the HTML endpoint (html.duckduckgo.com), parsed
    searxng      a SearXNG instance's JSON API (`url`), no key
    tavily       api.tavily.com (TAVILY_API_KEY); returns page text with the results
    brave        api.search.brave.com (BRAVE_API_KEY)
    bing         api.bing.microsoft.com v7 (BING_API_KEY)
    google       Google Programmable Search, customsearch v1 (GOOGLE_API_KEY, `cx`)
    exa          api.exa.ai (EXA_API_KEY); returns page text with the results
    bocha        api.bochaai.com (BOCHA_API_KEY); returns summaries
    kb           the knowledge service's /web-search (agenticwf.agent.runtime.web, not here: it needs the caller)
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Awaitable, Callable
from urllib.parse import parse_qs, urlparse

import httpx


class WebError(Exception):
    pass


@dataclass(frozen=True)
class ProviderInfo:
    url: str           # the default endpoint
    key_env: str       # the default variable holding its API key; "" = no key
    note: str          # for the model: what the provider is good at


INFO: dict[str, ProviderInfo] = {
    "kb": ProviderInfo("", "", "the knowledge service's web search: several engines, de-duplicated, with the top "
                               "results' page text (PDF and Office pages too)"),
    "duckduckgo": ProviderInfo("https://html.duckduckgo.com", "", "general web search; titles and snippets"),
    "searxng": ProviderInfo("", "", "metasearch over many engines; titles and snippets"),
    "tavily": ProviderInfo("https://api.tavily.com", "TAVILY_API_KEY",
                           "search built for agents: page text with each result; good for research questions"),
    "brave": ProviderInfo("https://api.search.brave.com", "BRAVE_API_KEY",
                          "independent index; general web and news; snippets"),
    "bing": ProviderInfo("https://api.bing.microsoft.com", "BING_API_KEY",
                         "broad coverage, news and recent events; snippets"),
    "google": ProviderInfo("https://www.googleapis.com", "GOOGLE_API_KEY", "the broadest coverage; snippets"),
    "exa": ProviderInfo("https://api.exa.ai", "EXA_API_KEY",
                        "neural search: describe the page you want in plain words; page text with each result; "
                        "good for papers, documentation, companies"),
    "bocha": ProviderInfo("https://api.bochaai.com", "BOCHA_API_KEY", "Chinese-language web; summaries of each page"),
}
PROVIDERS = tuple(INFO)


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    url: str = ""                 # empty: the provider's default
    api_key_env: str = ""         # empty: the provider's default
    cx: str = ""                  # google: the Programmable Search Engine id

    @property
    def endpoint(self) -> str:
        return (self.url or INFO[self.name].url).rstrip("/")

    @property
    def key_env(self) -> str:
        return self.api_key_env or INFO[self.name].key_env

    @property
    def api_key(self) -> str:
        return os.environ.get(self.key_env, "") if self.key_env else ""

    def check(self) -> str | None:
        """None if the provider can be used, else what is missing."""
        if self.name not in INFO:
            return f"unknown provider {self.name!r} (one of {', '.join(PROVIDERS)})"
        if self.name == "searxng" and not self.url:
            return "searxng needs url (the SearXNG instance)"
        if self.name == "google" and not self.cx:
            return "google needs cx (the Programmable Search Engine id)"
        if self.key_env and not self.api_key:
            return f"{self.name} needs an API key (${self.key_env})"
        return None


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    content: str = ""            # providers that return page text (tavily, exa, kb) or summaries (bocha)
    published: str = ""


Search = Callable[[httpx.AsyncClient, ProviderConfig, str, int], Awaitable[list[WebResult]]]


def _json(response: httpx.Response, name: str) -> dict:
    if response.status_code >= 400:
        raise WebError(f"{name}: HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError:
        raise WebError(f"{name}: not JSON")


def _key(p: ProviderConfig) -> str:
    if not p.api_key:
        raise WebError(f"{p.name} needs an API key (${p.key_env})")
    return p.api_key


def strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or ""))


# --- duckduckgo: the no-JavaScript results page --------------------------------------------------
class _DuckDuckGoPage(HTMLParser):
    """Results are `<a class="result__a" href=…>title</a>` and `<a class="result__snippet">…`;
    ads carry `result--ad` on their container and are skipped."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[WebResult] = []
        self._field: str | None = None
        self._ad_depth = 0
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        classes = (dict(attrs).get("class") or "").split()
        if tag == "div":
            self._depth += 1
            if "result--ad" in classes and not self._ad_depth:
                self._ad_depth = self._depth
        if self._ad_depth or tag != "a":
            return
        if "result__a" in classes:
            self.results.append(WebResult("", _ddg_target(dict(attrs).get("href") or ""), ""))
            self._field = "title"
        elif "result__snippet" in classes and self.results:
            self._field = "snippet"

    def handle_endtag(self, tag):
        if tag == "a":
            self._field = None
        elif tag == "div":
            if self._ad_depth == self._depth:
                self._ad_depth = 0
            self._depth -= 1

    def handle_data(self, data):
        if self._field == "title":
            self.results[-1].title += data
        elif self._field == "snippet":
            self.results[-1].snippet += data


def _ddg_target(href: str) -> str:
    """The real URL of a result: DuckDuckGo wraps it as //duckduckgo.com/l/?uddg=<url>."""
    parsed = urlparse(href if "://" in href else "https:" + href if href.startswith("//") else href)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
        return (parse_qs(parsed.query).get("uddg") or [""])[0]
    return href


async def duckduckgo(client: httpx.AsyncClient, p: ProviderConfig, query: str, n: int) -> list[WebResult]:
    response = await client.post(f"{p.endpoint}/html/", data={"q": query, "kp": "1"}, headers={
        "User-Agent": "Mozilla/5.0 (compatible; AgenticWF-agent/1)", "Accept": "text/html"})
    if response.status_code >= 400 or response.status_code == 202:   # 202: a challenge page (rate limited)
        raise WebError(f"duckduckgo: HTTP {response.status_code}")
    page = _DuckDuckGoPage()
    page.feed(response.text)
    results = [r for r in page.results if r.url.startswith(("http://", "https://"))]
    if not results and "anomaly" in response.text.lower():
        raise WebError("duckduckgo: refused (rate limited)")
    for r in results:
        r.title, r.snippet = r.title.strip(), " ".join(r.snippet.split())
    return results[:n]


async def searxng(client: httpx.AsyncClient, p: ProviderConfig, query: str, n: int) -> list[WebResult]:
    data = _json(await client.get(f"{p.endpoint}/search", params={"q": query, "format": "json", "safesearch": 1}),
                 p.name)
    return [WebResult(r.get("title", ""), r.get("url", ""), r.get("content", "") or "",
                      published=r.get("publishedDate") or "") for r in data.get("results", [])[:n]]


async def tavily(client: httpx.AsyncClient, p: ProviderConfig, query: str, n: int) -> list[WebResult]:
    data = _json(await client.post(f"{p.endpoint}/search", json={
        "query": query, "max_results": n, "search_depth": "basic", "include_answer": False,
        "include_raw_content": True}, headers={"Authorization": f"Bearer {_key(p)}"}), p.name)
    return [WebResult(r.get("title", ""), r.get("url", ""), r.get("content", "") or "",
                      content=r.get("raw_content") or "", published=r.get("published_date") or "")
            for r in data.get("results", [])[:n]]


async def brave(client: httpx.AsyncClient, p: ProviderConfig, query: str, n: int) -> list[WebResult]:
    data = _json(await client.get(f"{p.endpoint}/res/v1/web/search", params={"q": query, "count": min(n, 20)},
                                  headers={"X-Subscription-Token": _key(p), "Accept": "application/json"}), p.name)
    return [WebResult(r.get("title", ""), r.get("url", ""), strip_tags(r.get("description", "")),
                      published=r.get("age") or "") for r in (data.get("web") or {}).get("results", [])[:n]]


async def bing(client: httpx.AsyncClient, p: ProviderConfig, query: str, n: int) -> list[WebResult]:
    data = _json(await client.get(f"{p.endpoint}/v7.0/search", params={"q": query, "count": n,
                                                                       "textDecorations": False},
                                  headers={"Ocp-Apim-Subscription-Key": _key(p)}), p.name)
    return [WebResult(r.get("name", ""), r.get("url", ""), r.get("snippet", ""),
                      published=r.get("datePublished") or r.get("dateLastCrawled") or "")
            for r in (data.get("webPages") or {}).get("value", [])[:n]]


async def google(client: httpx.AsyncClient, p: ProviderConfig, query: str, n: int) -> list[WebResult]:
    data = _json(await client.get(f"{p.endpoint}/customsearch/v1", params={
        "key": _key(p), "cx": p.cx, "q": query, "num": min(n, 10), "safe": "active"}), p.name)
    return [WebResult(r.get("title", ""), r.get("link", ""), r.get("snippet", "")) for r in data.get("items", [])[:n]]


async def exa(client: httpx.AsyncClient, p: ProviderConfig, query: str, n: int) -> list[WebResult]:
    data = _json(await client.post(f"{p.endpoint}/search", json={
        "query": query, "numResults": n, "type": "auto", "contents": {"text": {"maxCharacters": 4000}}},
        headers={"x-api-key": _key(p)}), p.name)
    return [WebResult(r.get("title") or "", r.get("url", ""), (r.get("text") or "")[:300], content=r.get("text") or "",
                      published=r.get("publishedDate") or "") for r in data.get("results", [])[:n]]


async def bocha(client: httpx.AsyncClient, p: ProviderConfig, query: str, n: int) -> list[WebResult]:
    data = _json(await client.post(f"{p.endpoint}/v1/web-search", json={
        "query": query, "count": n, "summary": True, "freshness": "noLimit"},
        headers={"Authorization": f"Bearer {_key(p)}"}), p.name)
    if data.get("code") not in (None, 200):
        raise WebError(f"bocha: {data.get('msg') or data.get('code')}")
    pages = (data.get("data") or data).get("webPages") or {}
    return [WebResult(r.get("name", ""), r.get("url", ""), r.get("snippet", ""), content=r.get("summary") or "",
                      published=r.get("datePublished") or "") for r in pages.get("value", [])[:n]]


SEARCH: dict[str, Search] = {"duckduckgo": duckduckgo, "searxng": searxng, "tavily": tavily, "brave": brave,
                             "bing": bing, "google": google, "exa": exa, "bocha": bocha}
