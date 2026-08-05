"""Search and fetch, against a local SearXNG.

SearXNG rather than a search API because it needs no key, imposes no quota, and
aggregates several engines instead of depending on one — which matches how the rest of
this project is put together. It runs in Docker; see `deploy/searxng/`.

**`json` must be listed under `search.formats` in the SearXNG config.** The shipped
default is html only, and a JSON request against a default install returns 403 — which
reads like an authentication problem rather than a missing feature, and is the single
easiest thing to lose an hour to. `search()` detects that specific case and says so.

**Why fetching gets an address guard.** Everything else in this package is driven by a
model reading the user's own words. This one is driven by a model reading a URL, and the
next thing it reads is whatever that page says — so a page can try to steer the assistant
into fetching something else. On this machine "something else" includes the Ollama daemon
on 11434 and the router's own API on 8000, both unauthenticated. `_check_address` resolves
the host and refuses anything that lands on loopback, a private range, or link-local
(which is where cloud metadata services live). It runs again after redirects, because a
public host is allowed to redirect to a private one.

None of that makes fetched text trustworthy. It is untrusted input being handed to a
model, and the writer prompt should treat it as a quotation rather than as instructions.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse, urlsplit

import httpx

from app.tools.registry import Tool

log = logging.getLogger(__name__)

MAX_PAGE_BYTES = 2_000_000
"""Stop reading a response body past this. A model cannot use two megabytes of HTML and
`MAX_RESULT_CHARS` will discard almost all of it anyway; the cap is about not spending the
user's request downloading something huge."""


class WebConfig:
    """Runtime knobs, kept out of `Settings` so this module stays importable alone."""

    def __init__(
        self,
        searxng_url: str = "http://127.0.0.1:8080",
        timeout: float = 15.0,
        max_results: int = 5,
        allow_private_hosts: bool = False,
    ) -> None:
        self.searxng_url = searxng_url.rstrip("/")
        self.timeout = timeout
        self.max_results = max_results
        self.allow_private_hosts = allow_private_hosts


_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def _effective_address(address: object) -> object:
    """Unwrap an IPv6 address that is really carrying an IPv4 one.

    **Measured, and it blocked a legitimate site.** This machine's resolver returns NAT64
    addresses: `www.liquid.ai` came back as `64:ff9b::d896:1001`, which Python reports as
    `is_reserved` — so an enumerate-the-bad-categories check refused a perfectly public
    host. The low 32 bits are the real address, `216.150.16.1`, and that is what has to be
    judged. The same applies to IPv4-mapped `::ffff:a.b.c.d`.

    Unwrapping is a safety measure as much as a correctness one: `64:ff9b::7f00:1` carries
    127.0.0.1, and only the unwrapped form shows that.
    """
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return address.ipv4_mapped
        if address in _NAT64:
            return ipaddress.ip_address(int(address) & 0xFFFFFFFF)
    return address


def _check_address(url: str, *, allow_private: bool) -> str | None:
    """None if the URL is safe to fetch, otherwise the reason it is not."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        return f"only http and https URLs can be fetched, not {parts.scheme or 'that'!r}"
    host = parts.hostname
    if not host:
        return "that URL has no host in it"
    if allow_private:
        return None

    try:
        # Every address the name resolves to, not just the first: a host that returns one
        # public and one loopback address must not pass on the strength of the public one.
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except OSError as exc:
        return f"could not resolve {host!r} ({exc.strerror or exc})"

    for info in infos:
        raw = ipaddress.ip_address(info[4][0])
        address = _effective_address(raw)
        # `is_global` rather than a list of bad categories. Enumerating them is what let
        # NAT64 through as "reserved", and the standard library already knows which
        # addresses are routable on the public internet — including the ones added to the
        # special-purpose registry after this was written.
        if not getattr(address, "is_global", False):
            seen = f"{raw} ({address})" if address is not raw else str(raw)
            return (
                f"{host!r} resolves to {seen}, which is on this machine or its local "
                f"network. Only public addresses can be fetched."
            )
    return None


async def search(query: str, config: WebConfig | None = None) -> str:
    """Web search via SearXNG. Returns a numbered list of title, URL and snippet."""
    cfg = config or WebConfig()
    query = query.strip()
    if not query:
        return "No search query was given."

    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            response = await client.get(
                f"{cfg.searxng_url}/search",
                params={"q": query, "format": "json", "safesearch": 0},
                headers={"Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        return (
            f"The search service at {cfg.searxng_url} is not reachable ({exc}). "
            f"Start it with deploy/searxng/up.sh."
        )

    if response.status_code == 403:
        # The default-install failure, named precisely because it looks like auth.
        return (
            "SearXNG refused the JSON request (403). Its config almost certainly does not "
            "list `json` under `search.formats` — that is the default, and it has to be "
            "added in deploy/searxng/settings.yml."
        )
    if response.status_code >= 400:
        return f"Search failed: HTTP {response.status_code}."

    try:
        payload = response.json()
    except ValueError:
        return "The search service returned something that was not JSON."

    results = payload.get("results") or []
    if not results:
        return f"No results for {query!r}."

    lines = []
    for index, hit in enumerate(results[: cfg.max_results], start=1):
        title = (hit.get("title") or "untitled").strip()
        url = (hit.get("url") or "").strip()
        snippet = " ".join((hit.get("content") or "").split())
        lines.append(f"{index}. {title}\n   {url}\n   {snippet[:300]}")
    return f"Search results for {query!r}:\n\n" + "\n\n".join(lines)


async def fetch(url: str, config: WebConfig | None = None) -> str:
    """Fetch a page and return its readable text, with the boilerplate stripped."""
    cfg = config or WebConfig()
    url = url.strip()
    if not url:
        return "No URL was given."
    if "://" not in url:
        url = f"https://{url}"

    problem = _check_address(url, allow_private=cfg.allow_private_hosts)
    if problem:
        return f"I won't fetch that: {problem}"

    try:
        async with httpx.AsyncClient(
            timeout=cfg.timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LegendMode/0.1)"},
        ) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        return f"Could not fetch {url}: {exc}"

    # A public URL is allowed to redirect somewhere private, so the destination is
    # checked too, not only what was asked for.
    final = str(response.url)
    if final != url:
        problem = _check_address(final, allow_private=cfg.allow_private_hosts)
        if problem:
            return f"I won't follow that redirect: {problem}"

    if response.status_code >= 400:
        return f"{url} returned HTTP {response.status_code}."

    content_type = response.headers.get("content-type", "")
    body = response.content[:MAX_PAGE_BYTES]
    if "html" not in content_type and "xml" not in content_type:
        if content_type.startswith("text/") or "json" in content_type:
            return body.decode(response.encoding or "utf-8", errors="replace")
        return f"{url} is {content_type or 'an unknown type'}, which I can't read as text."

    html = body.decode(response.encoding or "utf-8", errors="replace")
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=True)
    except ImportError:
        return "Page extraction needs the `trafilatura` package, which is not installed."

    if not text:
        return f"Fetched {final} but found no readable article text in it."
    host = urlparse(final).netloc
    return f"Content of {final} (from {host}):\n\n{text}"


def tools(config: WebConfig | None = None) -> list[Tool]:
    cfg = config or WebConfig()

    async def _search(query: str) -> str:
        return await search(query, cfg)

    async def _fetch(url: str) -> str:
        return await fetch(url, cfg)

    return [
        Tool(
            name="web_search",
            description=(
                "Use for anything current, recent or otherwise not knowable from memory: "
                "news, prices, weather, sports results, release notes, who won something, "
                "what happened recently. Returns titles, URLs and snippets."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, as search terms rather than a question.",
                    }
                },
                "required": ["query"],
            },
            run=_search,
            family="web",
        ),
        Tool(
            name="fetch_url",
            description=(
                "Use when the user gives a URL, or to read a page found by web_search. "
                "Returns the page's readable text with navigation and adverts stripped."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL, including https://",
                    }
                },
                "required": ["url"],
            },
            run=_fetch,
            family="web",
        ),
    ]
