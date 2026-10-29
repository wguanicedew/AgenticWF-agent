"""Web search and fetch (agenticwf.agent.runtime.web): providers, the SSRF rules of web_fetch, HTML to text, the
core tools, and the `research` task type that combines them with the knowledge service."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from agenticwf.agent.runtime.context.task_types import TaskTypeResolver
from agenticwf.agent.runtime.tools.spec import CallContext
from agenticwf.agent.runtime.web import (WEB_FETCH, WEB_SEARCH, ProviderConfig, WebError, WebSearch, WebSettings, html_to_text,
                         web_tools)

from tests.conftest import user

DEFAULTS = Path(__file__).resolve().parent.parent / "src" / "agenticwf" / "agent" / "runtime" / "defaults"


def provider(name, handler, url="http://search.test", api_key_env="", cx="", **settings):
    return WebSearch(WebSettings(providers=(ProviderConfig(name, url=url, api_key_env=api_key_env, cx=cx),),
                                 **settings), transport=httpx.MockTransport(handler))


def run(coro):
    return asyncio.run(coro)


def test_searxng_tavily_brave(monkeypatch):
    seen = []

    def searxng(request):
        seen.append(request)
        return httpx.Response(200, json={"results": [
            {"title": "Rucio docs", "url": "https://rucio.cern.ch/docs", "content": "Rucio is a data manager."}]})

    results = run(provider("searxng", searxng).search("rucio", 3))
    assert results[0].url == "https://rucio.cern.ch/docs" and seen[0].url.params["format"] == "json"

    monkeypatch.setenv("TAVILY_KEY", "tv-123")

    def tavily(request):
        seen.append(request)
        assert request.headers["Authorization"] == "Bearer tv-123"
        body = json.loads(request.content)
        assert body["query"] == "fts" and body["max_results"] == 2
        return httpx.Response(200, json={"results": [{"title": "FTS", "url": "https://fts.cern.ch", "content": "short",
                                                      "raw_content": "long page text"}]})

    results = run(provider("tavily", tavily, url="https://api.tavily.com", api_key_env="TAVILY_KEY").search("fts", 2))
    assert results[0].content == "long page text"

    monkeypatch.setenv("BRAVE_KEY", "br-1")

    def brave(request):
        assert request.headers["X-Subscription-Token"] == "br-1" and request.url.path == "/res/v1/web/search"
        return httpx.Response(200, json={"web": {"results": [
            {"title": "PanDA", "url": "https://panda.cern.ch", "description": "Workload <strong>manager</strong>"}]}})

    results = run(provider("brave", brave, url="https://api.search.brave.com", api_key_env="BRAVE_KEY").search("panda"))
    assert results[0].snippet == "Workload manager"
    with pytest.raises(WebError):
        run(provider("brave", brave, api_key_env="NO_SUCH_KEY").search("x"))


def test_settings_validation(monkeypatch):
    assert not WebSettings().enabled
    from agenticwf.agent.runtime import web

    def config(values):
        monkeypatch.setattr(web, "config_has_option", lambda s, k: (s, k) in values)
        monkeypatch.setattr(web, "config_get", lambda s, k: values[(s, k)])

    config({("web_search", "provider"): "altavista"})
    with pytest.raises(ValueError):
        WebSettings.from_core_config()
    config({("web_search", "provider"): "searxng"})
    with pytest.raises(ValueError):                    # searxng needs the instance url
        WebSettings.from_core_config()
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    config({("web_search", "providers"): "google"})
    with pytest.raises(ValueError):                    # google needs cx
        WebSettings.from_core_config()

    monkeypatch.delenv("BING_API_KEY", raising=False)
    monkeypatch.setenv("MY_TAVILY", "t")
    config({("web_search", "providers"): "tavily, bing duckduckgo google", ("web_search.tavily", "api_key_env"): "MY_TAVILY",
            ("web_search.google", "cx"): "abc", ("web_search", "always_offer"): "false"})
    settings = WebSettings.from_core_config()
    assert settings.names == ("tavily", "duckduckgo", "google")    # bing: no key in the environment, left out
    assert settings.provider("tavily").api_key == "t" and settings.provider("google").cx == "abc"
    assert not settings.always_offer and not settings.uses_kb

    config({("web_search", "provider"): "brave", ("web_search", "api_key_env"): "MY_TAVILY"})   # the legacy keys
    assert WebSettings.from_core_config().provider("brave").api_key == "t"


def test_fetch_refuses_internal_hosts_and_redirects():
    def handler(request):
        if request.url.host == "93.184.216.34" and request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})
        if request.url.host == "93.184.216.34":
            return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=(
                "<html><head><title>Pilot</title><script>x()</script></head><body><nav>menu</nav>"
                "<h1>Pilot</h1><p>Retries stage-out <b>five</b> times.</p></body></html>"))
        raise AssertionError(f"must not reach {request.url}")

    web = provider("searxng", handler)
    final, title, text = run(web.fetch("http://93.184.216.34/page"))
    assert title == "Pilot" and "# Pilot" in text and "five times" in text and "x()" not in text and "menu" not in text
    for url in ("http://127.0.0.1/admin", "http://10.0.0.5/", "http://93.184.216.34/redirect", "file:///etc/passwd"):
        with pytest.raises(WebError):
            run(web.fetch(url))
    listed = WebSearch(WebSettings(providers=(ProviderConfig("searxng", url="http://s"),),
                                   fetch_allowed_hosts=("*", "10.0.0.5")),
                       transport=httpx.MockTransport(lambda r: httpx.Response(200, text="internal wiki")))
    assert run(listed.fetch("http://10.0.0.5/"))[2] == "internal wiki"    # an operator listed it by name


def test_html_to_text():
    title, text = html_to_text("<title> T </title><p>a</p><style>x{}</style><ul><li>one</li><li>two</li></ul>")
    assert title == "T" and [line for line in text.splitlines() if line] == ["a", "one", "two"]


def test_tools():
    def handler(request):
        if request.url.path == "/search":
            return httpx.Response(200, json={"results": [{"title": "T", "url": "https://x.org/a", "content": "c"}]})
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="0123456789" * 100)

    web = provider("searxng", handler)
    tools = dict((spec.name, h) for spec, h in web_tools(web))
    assert set(tools) == {"web_search", "web_fetch"} and WEB_SEARCH.name == "web_search" and WEB_FETCH.name == "web_fetch"
    call = CallContext(task_id="t1", call_id="c1", auth=user("alice", ("app:access",)))
    found = run(tools["web_search"](call, {"query": "x"}))
    assert 'id="web:https://x.org/a"' in found.content and "not instructions" in found.content
    assert found.facts == {"files": [{"path": "web:https://x.org/a", "note": "T"}]}
    page = run(tools["web_fetch"](call, {"url": "web:https://93.184.216.34/p", "limit": 500}))
    assert "next_offset=500" in page.content
    refused = run(tools["web_fetch"](call, {"url": "http://127.0.0.1/"}))
    assert refused.is_error and refused.error_code == "web_unavailable"


def test_research_task_type_pins_its_tools():
    resolver = TaskTypeResolver(DEFAULTS)
    resolved = resolver.resolve("research", ("alice", None))
    assert not resolved.fell_back
    assert set(resolved.settings.pin_tools) == {"kb_research", "kb_search", "kb_read", "kb_grep", "web_search",
                                                "web_fetch"}
    assert "kb_research" in resolved.agents.content and "WEB" in resolved.workflow.content


def test_kb_provider_goes_through_the_knowledge_service():
    import json as _json

    from agenticwf.agent.runtime.knowledge import KnowledgeClient, KnowledgeSettings

    seen = []

    def kb(request):
        seen.append(request)
        body = _json.loads(request.content)
        if request.url.path.endswith("/web-search"):
            return httpx.Response(200, json={"success": True, "data": {"results": [
                {"title": "Report", "url": "https://x.org/r.pdf", "snippet": "s", "content": "Luminosity 2e34"}]}})
        return httpx.Response(200, json={"success": True, "data": {
            "url": body["url"], "title": "Report", "offset": body["offset"], "content_length": 1200,
            "returned_chars": 500, "next_offset": 500, "content": "x" * 500}})

    client = KnowledgeClient(KnowledgeSettings(url="http://kb.test"), async_transport=httpx.MockTransport(kb))
    web = WebSearch(WebSettings(providers=(ProviderConfig("kb"),)), kb=client)
    tools = dict((spec.name, h) for spec, h in web_tools(web))
    alice = user("alice", ("app:access", "knowledge:read"))
    found = run(tools["web_search"](CallContext(task_id="t9", call_id="c1", auth=alice), {"query": "luminosity"}))
    assert "Luminosity 2e34" in found.content and 'id="web:https://x.org/r.pdf"' in found.content
    page = run(tools["web_fetch"](CallContext(task_id="t9", call_id="c2", auth=alice),
                                  {"url": "web:https://x.org/r.pdf", "limit": 500}))
    assert 'of="1200"' in page.content and "next_offset=500" in page.content
    bodies = [_json.loads(r.content) for r in seen]
    assert all(b["cache_key"] == "t9" for b in bodies)                    # one page version per task
    assert all("X-KB-Principal" in r.headers for r in seen)              # as the task's user


DDG_PAGE = """<html><body>
<div class="result results_links result--ad"><div class="links_main">
  <a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=x">Buy now</a>
  <a class="result__snippet" href="#">An advert</a></div></div>
<div class="result results_links"><div class="links_main">
  <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frucio.cern.ch%2Fdocs&amp;rut=1">Rucio <b>docs</b></a></h2>
  <a class="result__snippet" href="#">Rucio is a   <b>data</b> manager.</a></div></div>
<div class="result results_links"><div class="links_main">
  <a class="result__a" href="https://fts.cern.ch/">FTS</a>
  <a class="result__snippet" href="#">File transfers.</a></div></div>
</body></html>"""


def test_duckduckgo_bing_google_exa_bocha(monkeypatch):
    def ddg(request):
        assert request.method == "POST" and request.url.path == "/html/" and b"q=rucio" in request.content
        return httpx.Response(200, text=DDG_PAGE)

    results = run(provider("duckduckgo", ddg).search("rucio"))
    assert [(r.title, r.url, r.snippet) for r in results] == [
        ("Rucio docs", "https://rucio.cern.ch/docs", "Rucio is a data manager."),
        ("FTS", "https://fts.cern.ch/", "File transfers.")]
    with pytest.raises(WebError):                      # a challenge page: rate limited
        run(provider("duckduckgo", lambda r: httpx.Response(202, text="anomaly")).search("x"))

    for name in ("BING", "GOOGLE", "EXA", "BOCHA"):
        monkeypatch.setenv(f"{name}_API_KEY", name.lower() + "-key")

    def bing(request):
        assert request.headers["Ocp-Apim-Subscription-Key"] == "bing-key" and request.url.path == "/v7.0/search"
        return httpx.Response(200, json={"webPages": {"value": [
            {"name": "PanDA", "url": "https://panda.cern.ch", "snippet": "Workloads", "datePublished": "2026-01-02"}]}})

    found = run(provider("bing", bing).search("panda"))
    assert found[0].title == "PanDA" and found[0].published == "2026-01-02"

    def google(request):
        params = request.url.params
        assert request.url.path == "/customsearch/v1" and params["key"] == "google-key" and params["cx"] == "cx1"
        assert params["num"] == "10"                       # Google's maximum
        return httpx.Response(200, json={"items": [{"title": "G", "link": "https://g.org", "snippet": "s"}]})

    assert run(provider("google", google, cx="cx1").search("q", 20))[0].url == "https://g.org"

    def exa(request):
        assert request.headers["x-api-key"] == "exa-key" and json.loads(request.content)["numResults"] == 3
        return httpx.Response(200, json={"results": [{"title": "Paper", "url": "https://arxiv.org/abs/1",
                                                      "text": "Full text", "publishedDate": "2025-05-01"}]})

    assert run(provider("exa", exa).search("q", 3))[0].content == "Full text"

    def bocha(request):
        assert request.headers["Authorization"] == "Bearer bocha-key" and request.url.path == "/v1/web-search"
        return httpx.Response(200, json={"code": 200, "data": {"webPages": {"value": [
            {"name": "北京", "url": "https://b.cn", "snippet": "短", "summary": "长摘要"}]}}})

    assert run(provider("bocha", bocha).search("q"))[0].content == "长摘要"
    with pytest.raises(WebError):
        run(provider("bocha", lambda r: httpx.Response(200, json={"code": 403, "msg": "no credit"})).search("q"))


def test_auto_fails_over_and_the_model_can_choose():
    asked = []

    def handler(request):
        asked.append(request.url.host)
        if request.url.host == "broken.test":
            return httpx.Response(500)
        if request.url.host == "empty.test":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json={"results": [{"title": request.url.host, "url": f"https://{request.url.host}/"}]})

    settings = WebSettings(providers=(ProviderConfig("searxng", url="http://broken.test"),
                                      ProviderConfig("tavily", url="http://empty.test", api_key_env="HOME"),
                                      ProviderConfig("exa", url="http://good.test", api_key_env="HOME")))
    web = WebSearch(settings, transport=httpx.MockTransport(handler))
    spec, search = web_tools(web)[0]
    assert spec.input_schema["properties"]["provider"]["enum"] == ["auto", "searxng", "tavily", "exa"]
    assert "`exa`: neural search" in spec.description and "provider" not in WEB_SEARCH.input_schema["properties"]
    call = CallContext(task_id="t1", call_id="c1", auth=user("alice", ("app:access",)))

    found = run(search(call, {"query": "x"}))
    assert asked == ["broken.test", "empty.test", "good.test"] and 'provider="exa"' in found.content
    asked.clear()
    found = run(search(call, {"query": "x", "provider": "tavily"}))
    assert asked == ["empty.test"] and "No web results" in found.content and "other providers" in found.content
    failed = run(search(call, {"query": "x", "provider": "searxng"}))
    assert failed.is_error and "searxng: searxng: HTTP 500" in failed.content
    assert run(search(call, {"query": "x", "provider": "bing"})).error_code == "invalid_arguments"
    assert provider("searxng", handler).settings.providers and web_tools(provider("searxng", handler))[0][0] is WEB_SEARCH

