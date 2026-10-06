"""The search and page-reading services behind the Web search and Web reader modules.

The model sees one stable tool (``web_search``, ``read_webpage``); *which service answers* is a setting, so adding another
provider never means a new tool. Four are supported, as in the .NET editions:

| Engine | Search | Read a page | Needs |
|---|---|---|---|
| SearXNG | yes | direct fetch (the web reader's own, SSRF-protected) | your own SearXNG server, no key |
| Jina | ``s.jina.ai`` | ``r.jina.ai`` | a key for search; reading works without one at a lower rate |
| Tavily | ``/search`` | ``/extract`` | an API key |
| Firecrawl | ``/v2/search`` | ``/v2/scrape`` (renders JavaScript), plus ``/v2/crawl`` | an API key |

Every function here takes the settings and an optional ``transport`` (tests pass one so no network is needed), returns
plain text for the model, and raises ``ToolError`` with a message that says what went wrong, because that message is what the
model sees and can relay. Results are *untrusted*; the modules run them through ``sanitize_untrusted`` and label them.

(Firecrawl's calls are written from its documented REST API; unlike the others they could not be tried against a real key
while this was built, so the tests check the request and response shapes against the documentation only.)

Python ideas used here:

* A small class per service sharing one helper (``_request``) that turns every failure into a readable message.
* ``asyncio.timeout`` and a polling loop (the Firecrawl crawl is started, then asked about every few seconds).
* Parsing JSON defensively with ``.get(...)`` because a service may add or omit fields.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx2 as httpx

from lemonrind.config import WebSearchSettings
from lemonrind.modules.tool import ToolError

REQUEST_TIMEOUT_SECONDS = 30.0
SNIPPET_CHARS = 400
SEARCH_LIMIT = 5

JINA_SEARCH_URL = "https://s.jina.ai/"
JINA_READER_URL = "https://r.jina.ai/"
TAVILY_URL = "https://api.tavily.com"
FIRECRAWL_URL = "https://api.firecrawl.dev"

CRAWL_POLL_SECONDS = 3.0
CRAWL_TIMEOUT_SECONDS = 120.0
MAX_CRAWL_PAGES = 30


async def _request(
    method: str,
    url: str,
    *,
    service: str,
    transport: httpx.AsyncBaseTransport | None,
    **kwargs: Any,
) -> httpx.Response:
    """Send a request; on any failure raise ``ToolError`` naming the service and, when it said why, its own explanation."""
    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_SECONDS, transport=transport
        ) as client:
            response = await client.request(method, url, **kwargs)
    except httpx.HTTPError as error:
        raise ToolError(f"Could not reach {service}: {type(error).__name__}: {error}") from error
    if response.status_code >= 400:
        raise ToolError(f"{service} returned {response.status_code}: {response.text[:500].strip()}")
    return response


def _json(response: httpx.Response, service: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as error:
        raise ToolError(f"{service} sent something that is not JSON.") from error
    return data if isinstance(data, dict) else {}


# --- SearXNG ---------------------------------------------------------------------------------------------------


def format_results(results: list[dict[str, Any]], limit: int) -> str:
    """Turn SearXNG's result list into numbered plain text the model can read."""
    lines: list[str] = []
    for number, item in enumerate(results[:limit], start=1):
        title = (item.get("title") or "(no title)").strip()
        snippet = " ".join((item.get("content") or "").split())[:SNIPPET_CHARS]
        lines.append(f"{number}. {title}\n   {item.get('url', '')}\n   {snippet}".rstrip())
    return "\n\n".join(lines)


async def searxng_search(
    config: WebSearchSettings,
    query: str,
    limit: int,
    time_range: str | None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    base_url = config.searxng_url.strip()
    if not base_url:
        raise ToolError("Web search is not set up: the SearXNG address is empty in Settings.")
    if not base_url.endswith("/"):
        base_url += "/"

    async def ask(time_filter: str | None) -> dict[str, Any]:
        params = {"q": query, "format": "json"}
        if time_filter:
            params["time_range"] = time_filter
        try:
            async with httpx.AsyncClient(
                base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS, transport=transport
            ) as client:
                response = await client.get("search", params=params)
                response.raise_for_status()
                return _json(response, "SearXNG")
        except httpx.HTTPStatusError as error:
            hint = (
                " (SearXNG refused the JSON format: add 'json' to search.formats in its settings.yml)"
                if error.response.status_code == 403
                else ""
            )
            raise ToolError(
                f"The search server answered {error.response.status_code}{hint}."
            ) from error
        except httpx.HTTPError as error:
            raise ToolError(
                f"Could not search at {base_url}: {type(error).__name__}: {error}"
            ) from error

    data = await ask(time_range)
    # Some SearXNG engines quietly return nothing whenever a time filter is set. A broader answer is better than none.
    if not data.get("results") and time_range:
        data = await ask(None)
    text = format_results(data.get("results", []), limit)
    if text:
        return text
    broken = data.get("unresponsive_engines") or []
    if broken:
        names = ", ".join(f"{item[0]} ({item[1]})" for item in broken if len(item) >= 2)
        raise ToolError(
            "The search backend is currently degraded, not this specific query - unresponsive engines: "
            f"{names}. Do not keep retrying with different phrasing; tell the user directly instead."
        )
    return f"No results for '{query}'."


# --- Jina ------------------------------------------------------------------------------------------------------


def _jina_headers(config: WebSearchSettings) -> dict[str, str]:
    return {"Authorization": f"Bearer {config.jina_api_key}"} if config.jina_api_key.strip() else {}


async def jina_search(
    config: WebSearchSettings, query: str, transport: httpx.AsyncBaseTransport | None = None
) -> str:
    response = await _request(
        "GET", JINA_SEARCH_URL, service="Jina", transport=transport,
        params={"q": query}, headers=_jina_headers(config),
    )  # fmt: skip
    return response.text


async def jina_read(
    config: WebSearchSettings, url: str, transport: httpx.AsyncBaseTransport | None = None
) -> str:
    response = await _request(
        "GET",
        JINA_READER_URL + url,
        service="Jina",
        transport=transport,
        headers=_jina_headers(config),
    )
    return response.text


# --- Tavily ----------------------------------------------------------------------------------------------------


def _tavily_headers(config: WebSearchSettings) -> dict[str, str]:
    if not config.tavily_api_key.strip():
        raise ToolError("Tavily needs an API key: add it in Settings > Web search.")
    return {"Authorization": f"Bearer {config.tavily_api_key}"}


async def tavily_search(
    config: WebSearchSettings, query: str, transport: httpx.AsyncBaseTransport | None = None
) -> str:
    response = await _request(
        "POST", f"{TAVILY_URL}/search", service="Tavily", transport=transport,
        json={"query": query, "max_results": SEARCH_LIMIT}, headers=_tavily_headers(config),
    )  # fmt: skip
    data = _json(response, "Tavily")
    sections: list[str] = []
    if answer := (data.get("answer") or "").strip():
        sections.append(f"Answer: {answer}")
    results = data.get("results") or []
    if results:
        sections.append(
            "\n".join(
                f"- {r.get('title', '')}: {r.get('content', '')} ({r.get('url', '')})"
                for r in results
            )
        )
    return "\n\n".join(sections) or "No results found."


async def tavily_extract(
    config: WebSearchSettings, url: str, transport: httpx.AsyncBaseTransport | None = None
) -> str:
    response = await _request(
        "POST", f"{TAVILY_URL}/extract", service="Tavily", transport=transport,
        json={"urls": url}, headers=_tavily_headers(config),
    )  # fmt: skip
    data = _json(response, "Tavily")
    if failed := (data.get("failed_results") or []):
        raise ToolError(
            f"Tavily couldn't extract '{failed[0].get('url', url)}': {failed[0].get('error', '')}"
        )
    results = data.get("results") or []
    content = (results[0].get("raw_content") or "") if results else ""
    if not content.strip():
        raise ToolError(f"Tavily returned no content for '{url}'.")
    return content


# --- Firecrawl -------------------------------------------------------------------------------------------------


def _firecrawl_headers(config: WebSearchSettings) -> dict[str, str]:
    if not config.firecrawl_api_key.strip():
        raise ToolError("Firecrawl needs an API key: add it in Settings > Web search.")
    return {"Authorization": f"Bearer {config.firecrawl_api_key}"}


async def firecrawl_search(
    config: WebSearchSettings, query: str, transport: httpx.AsyncBaseTransport | None = None
) -> str:
    response = await _request(
        "POST", f"{FIRECRAWL_URL}/v2/search", service="Firecrawl", transport=transport,
        json={"query": query, "limit": SEARCH_LIMIT}, headers=_firecrawl_headers(config),
    )  # fmt: skip
    data = _json(response, "Firecrawl").get("data") or {}
    hits = (
        data.get("web", []) if isinstance(data, dict) else data
    )  # older shapes put the list straight in "data"
    if not hits:
        return "No results found."
    return "\n".join(
        f"- {h.get('title', '')}: {h.get('description', '')} ({h.get('url', '')})" for h in hits
    )


async def firecrawl_scrape(
    config: WebSearchSettings, url: str, transport: httpx.AsyncBaseTransport | None = None
) -> str:
    response = await _request(
        "POST", f"{FIRECRAWL_URL}/v2/scrape", service="Firecrawl", transport=transport,
        json={"url": url, "formats": ["markdown"]}, headers=_firecrawl_headers(config),
    )  # fmt: skip
    markdown = ((_json(response, "Firecrawl").get("data") or {}).get("markdown") or "").strip()
    if not markdown:
        raise ToolError(f"Firecrawl returned no readable content for '{url}'.")
    return markdown


async def firecrawl_crawl(
    config: WebSearchSettings,
    url: str,
    max_pages: int,
    focus: str | None,
    transport: httpx.AsyncBaseTransport | None = None,
    *,
    poll_seconds: float = CRAWL_POLL_SECONDS,
    timeout_seconds: float = CRAWL_TIMEOUT_SECONDS,
) -> str:
    """Crawl a site from ``url`` (up to ``max_pages``, at most 30) and return every page's text, labelled by address."""
    headers = _firecrawl_headers(config)
    body: dict[str, Any] = {"url": url, "limit": max(1, min(max_pages, MAX_CRAWL_PAGES))}
    if focus and focus.strip():
        body["prompt"] = focus.strip()
    started = _json(
        await _request(
            "POST",
            f"{FIRECRAWL_URL}/v2/crawl",
            service="Firecrawl",
            transport=transport,
            json=body,
            headers=headers,
        ),
        "Firecrawl",
    )
    job_id = started.get("id")
    if not job_id:
        raise ToolError(f"Firecrawl did not start a crawl of '{url}'.")

    status: dict[str, Any] = {}
    try:
        async with asyncio.timeout(timeout_seconds):
            while True:
                status = _json(
                    await _request(
                        "GET",
                        f"{FIRECRAWL_URL}/v2/crawl/{job_id}",
                        service="Firecrawl",
                        transport=transport,
                        headers=headers,
                    ),
                    "Firecrawl",
                )
                if status.get("status") in ("completed", "failed", "cancelled"):
                    break
                await asyncio.sleep(poll_seconds)
    except TimeoutError as error:
        raise ToolError(
            f"Firecrawl's crawl of '{url}' did not finish within {int(timeout_seconds)} seconds."
        ) from error

    pages = status.get("data") or []
    if status.get("status") != "completed" or not pages:
        raise ToolError(
            f"Firecrawl crawl of '{url}' finished with no pages (status: {status.get('status')})."
        )
    parts = []
    for index, page in enumerate(pages, start=1):
        source = (page.get("metadata") or {}).get("sourceURL") or (page.get("metadata") or {}).get(
            "url"
        )
        parts.append(f"--- {source or f'page {index}'} ---\n{page.get('markdown', '')}")
    return "\n\n".join(parts)
