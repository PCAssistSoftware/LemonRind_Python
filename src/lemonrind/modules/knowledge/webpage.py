"""Fetching a web page and reducing it to its readable text.

A web page is HTML: the words you read are surrounded by tags, scripts, styles and navigation. Embedding
that raw would mostly embed markup. The standard library's ``html.parser`` walks the tags as a stream of events
(start tag, text, end tag), which is enough to keep the visible text and drop the rest. (Libraries such as
BeautifulSoup do this with a nicer API, but nothing here needs more than this.)

Only ``http`` and ``https`` addresses are accepted, and pages are size-limited, because the address comes from a
person typing into a form and the response comes from the internet.

Python ideas used here:

* **Subclassing** ``HTMLParser`` and overriding its callbacks (``handle_starttag``, ``handle_data``...): the
  parser calls your methods as it reads, the classic "visitor" pattern.
* ``httpx2.AsyncClient`` with ``follow_redirects`` and a ``transport`` for tests, as in the web search module.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx2 as httpx

MAX_PAGE_BYTES = 5_000_000
TIMEOUT_SECONDS = 30.0

_SKIPPED = {"script", "style", "noscript", "svg", "head", "template", "iframe"}
_BLOCKS = {
    "p",
    "div",
    "br",
    "li",
    "ul",
    "ol",
    "tr",
    "table",
    "section",
    "article",
    "header",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "pre",
    "blockquote",
}  # fmt: skip  (tags after which the text should start a new line)


class _TextExtractor(HTMLParser):
    def __init__(self, extra_skipped: frozenset[str] = frozenset()) -> None:
        super().__init__(convert_charrefs=True)  # turns "&amp;" into "&" for us
        self._skipped = _SKIPPED | extra_skipped  # tags whose whole contents are ignored
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True
        if tag in self._skipped:
            self._skip_depth += 1
        elif tag in _BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self._skipped and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str, extra_skipped: frozenset[str] = frozenset()) -> tuple[str, str]:
    """``(title, text)`` of an HTML document. Whitespace is tidied; paragraphs stay on separate lines.

    ``extra_skipped`` names more tags whose contents to drop (the web reader drops navigation and footers).
    """
    extractor = _TextExtractor(extra_skipped)
    extractor.feed(html)
    extractor.close()
    text = "".join(extractor.parts)
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.splitlines())
    return extractor.title.strip(), "\n".join(line for line in lines if line)


class WebPageError(Exception):
    """A page could not be fetched or read. The message is fit to show the user."""


async def fetch_page_text(
    url: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[str, str]:
    """Download ``url`` and return ``(title, text)``."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise WebPageError("Enter a web address starting with http:// or https://")
    try:
        async with httpx.AsyncClient(
            timeout=TIMEOUT_SECONDS, follow_redirects=True, transport=transport
        ) as client:
            response = await client.get(url.strip(), headers={"User-Agent": "LemonRind/0.1"})
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise WebPageError(f"Could not fetch {url}: {type(error).__name__}: {error}") from error

    if len(response.content) > MAX_PAGE_BYTES:
        raise WebPageError(f"The page is larger than {MAX_PAGE_BYTES // 1_000_000} MB.")
    content_type = response.headers.get("content-type", "")
    if "html" in content_type or "xml" in content_type or not content_type:
        title, text = html_to_text(response.text)
    elif content_type.startswith("text/"):
        title, text = "", response.text
    else:
        raise WebPageError(f"Cannot read content of type '{content_type}'.")
    if not text.strip():
        raise WebPageError(
            "The page has no readable text (it may need JavaScript to show content)."
        )
    return title, text
