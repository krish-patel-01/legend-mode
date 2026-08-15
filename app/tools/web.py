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

MAX_SUMMARY_CHARS = 600
"""How much of an infobox to keep, against 300 for a result snippet.

Twice the budget because it is worth more: a snippet is whatever text happened to sit near
the match on a page, while an infobox is an encyclopaedia's own opening description of the
thing that was asked about. 600 covers a full Wikipedia lead paragraph.
"""


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


def _summarise(payload: dict) -> str | None:
    """The direct answer SearXNG found, if it found one, as a labelled block.

    **Why this exists.** `results` is a ranked list of *pages*, and for a question whose
    answer is a fact rather than a document the top of that list is often navigation:
    "last f1 race" returns formula1.com and an ESPN calendar page, neither of which
    contains a result. Handed only that, a 1.2B writer invents something plausible instead
    of saying the evidence does not answer the question — the failure this is aimed at.

    SearXNG already carries the fact separately when it has one. `infoboxes` is the
    encyclopaedia entry for the subject; `answers` is the instant-answer slot fed by
    plugins. Both were being dropped on the floor here, and only `results` was read.

    **Measured on this instance, 2026-08-14, SearXNG in `legend-searxng`:**

    | query                | answers | infoboxes | results |
    |----------------------|---------|-----------|---------|
    | capital of Nigeria   | 0       | 1         | 36      |
    | population of India  | 0       | 1         | 27      |
    | mass of Earth        | 0       | 1         | 34      |
    | last f1 race         | 0       | **0**     | 39      |

    Two things follow, and the second is the awkward one.

    `infoboxes` is real and worth reading: three of four factual queries carried a
    Wikipedia lead paragraph that answers the question outright.

    `answers` was empty on all eleven queries tried, including `1+1` and `user agent`,
    whose plugins are `active: true` in the container's own defaults — and empty in the
    HTML output too, so it is the plugins not firing rather than the JSON serialiser
    dropping them. It is still read here: it is three lines, it is the documented field,
    and NEO's wrapper preferred it for good reason. But it is covered by a synthetic
    payload in the tests rather than by anything observed live, and nothing here should be
    described as verified against a real answer until one is seen.

    **Re-tested 2026-08-15 while probing scoping, and it is still empty.** One run did
    return `answers: 1` for "mass of Earth" under `engines=google,duckduckgo,wikipedia` —
    the only non-empty `answers` ever seen here — and it did not reproduce: 0 on all 36
    subsequent observations, across 4 queries x 3 engine configurations x 3 repetitions,
    `1+1` among them. So it is reachable in principle and unreliable in practice, which
    changes nothing about the code and does mean a future single sighting should not be
    read as the field having started working.

    **This does not fix "last f1 race".** That query has no infobox and no answer, so it
    still reaches the writer as navigational links, exactly as before. The grounding
    problem is narrowed to the queries SearXNG has no direct answer for, not solved.
    """
    parts: list[str] = []

    for answer in (payload.get("answers") or [])[:1]:
        # Plugins hand back either a bare string or an object with the text under `answer`.
        text = answer.get("answer") if isinstance(answer, dict) else answer
        text = " ".join(str(text or "").split())
        if text:
            parts.append(f"Direct answer: {text[:MAX_SUMMARY_CHARS]}")

    for box in (payload.get("infoboxes") or [])[:1]:
        content = " ".join((box.get("content") or "").split())
        if not content:
            continue
        # The subject's name lives under `infobox`; `title` is present but empty on the
        # wikipedia engine, so reading `title` first would silently label everything "".
        name = (box.get("infobox") or box.get("title") or "").strip()
        source = (box.get("id") or box.get("url") or "").strip()
        engine = (box.get("engine") or "reference").strip()

        block = f"Summary of {name} (from {engine})" if name else f"Summary (from {engine})"
        lines = [f"{block}: {content[:MAX_SUMMARY_CHARS]}"]
        for attribute in (box.get("attributes") or [])[:5]:
            label = " ".join(str(attribute.get("label") or "").split())
            value = attribute.get("value")
            if isinstance(value, list):  # some engines nest the value
                value = ", ".join(str(v) for v in value)
            value = " ".join(str(value or "").split())
            if label and value:
                lines.append(f"   {label}: {value[:120]}")
        if source:
            lines.append(f"   source: {source}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else None


async def search(query: str, config: WebConfig | None = None) -> str:
    """Web search via SearXNG. Returns any direct answer found, then the ranked pages."""
    cfg = config or WebConfig()
    query = query.strip()
    if not query:
        return "No search query was given."

    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            # **The parameters are minimal on purpose. Scoping was measured and rejected.**
            #
            # SearXNG also accepts `engines`, `categories` and `time_range`, and the
            # obvious theory was that pointing a question at the right engines would turn
            # navigational results into answers. Probed 2026-08-15 against this instance
            # over 8 queries — 3 encyclopaedic, 5 recency — with the config order rotated
            # per query:
            #
            #   config                        enc direct   rec direct   note
            #   default                          3/3          0/5
            #   categories=news                  0/3          0/5       loses every infobox
            #   categories=general,news          3/3          0/5       wash; see below
            #   engines=wikipedia,wikidata       3/3          0/5       0 result pages at all
            #   engines=google,duckduckgo,...    3/3          0/5       fewer results, no gain
            #   time_range=week / month          0/6         0/10       0 results, 10/10
            #
            # `time_range` is the one that looked most targeted and is the worst: it
            # returned nothing at all for every recency query and stripped the infoboxes
            # off the encyclopaedic ones. `categories=general,news` was the only candidate
            # that survived, and at 3 repetitions it scored 9/9 and 14/15 against the
            # default's 8/9 and 13/15 — inside the noise. Case by case it trades rather
            # than wins: it found "Gold Prices Per Ounce, $4,377.00" where the default
            # offered $1,300 from 2019, and it lost Ahmedabad's actual temperature to an
            # air-quality page and the shipping iPhone to rumours about an unreleased one.
            #
            # Swapping one failure mode for another is not an improvement, and the rule
            # needed to tell those cases apart — sports result versus product name — is
            # exactly the kind of guard that is worse than no guard when it misreads the
            # question.
            #
            # One measurement to re-take before trusting any of this: the unscoped results
            # are not stable. The same gold query returned decade-old prices on one run and
            # "1 hour ago" live pages on the next, so a top-5 quality comparison at low
            # repetition is reading noise. Anything re-testing scoping needs more samples
            # than this did.
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

    summary = _summarise(payload)
    results = payload.get("results") or []
    if not results:
        # A direct answer with no ranked pages behind it is still an answer. Returning
        # "no results" here would have thrown away the better half of the response.
        return summary or f"No results for {query!r}."

    lines = []
    for index, hit in enumerate(results[: cfg.max_results], start=1):
        title = (hit.get("title") or "untitled").strip()
        url = (hit.get("url") or "").strip()
        snippet = " ".join((hit.get("content") or "").split())
        lines.append(f"{index}. {title}\n   {url}\n   {snippet[:300]}")

    # Summary first: it is the part most likely to contain the answer, and on a small
    # model position in the context is not neutral.
    body = "\n\n".join(lines)
    if summary:
        return f"Search results for {query!r}:\n\n{summary}\n\n{body}"
    return f"Search results for {query!r}:\n\n{body}"


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
