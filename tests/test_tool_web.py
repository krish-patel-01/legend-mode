"""Search and fetch.

No network is touched. What matters here is the address guard and the failure messages —
the guard because it is the only thing between a model-supplied URL and the unauthenticated
services on this machine, and the messages because "403" from a default SearXNG install
looks like an auth problem and is actually a missing config line.
"""

from __future__ import annotations

import pytest

from app import api
from app.tools import web


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/api/tags",  # the Ollama daemon, unauthenticated
        "http://localhost:8000/v1/models",  # this router's own API
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/admin",
        "http://192.168.1.1/",
        "http://[::1]:8080/",
    ],
)
def test_private_and_loopback_addresses_are_refused(url: str) -> None:
    assert web._check_address(url, allow_private=False) is not None


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://x/"])
def test_only_http_urls_are_fetchable(url: str) -> None:
    reason = web._check_address(url, allow_private=False)
    assert reason is not None and "http" in reason


def test_a_public_host_passes() -> None:
    assert web._check_address("https://example.com/page", allow_private=False) is None


@pytest.mark.parametrize(
    ("wrapped", "embedded", "public"),
    [
        ("64:ff9b::d896:1001", "216.150.16.1", True),  # NAT64, a real site this resolver returns
        ("64:ff9b::7f00:1", "127.0.0.1", False),  # NAT64 hiding loopback
        ("64:ff9b::a00:5", "10.0.0.5", False),  # NAT64 hiding a private range
        ("::ffff:93.184.216.34", "93.184.216.34", True),  # IPv4-mapped, public
        ("::ffff:127.0.0.1", "127.0.0.1", False),  # IPv4-mapped loopback
    ],
)
def test_ipv6_wrappers_are_judged_by_the_address_they_carry(
    wrapped: str, embedded: str, public: bool
) -> None:
    """A NAT64 address reads as `is_reserved`, which wrongly refused a live site."""
    import ipaddress

    effective = web._effective_address(ipaddress.ip_address(wrapped))
    assert str(effective) == embedded
    assert getattr(effective, "is_global", False) is public


def test_the_guard_can_be_switched_off_deliberately() -> None:
    assert web._check_address("http://127.0.0.1:8080/", allow_private=True) is None


def test_a_host_that_does_not_resolve_is_refused() -> None:
    reason = web._check_address(
        "https://this-host-should-not-exist.invalid/", allow_private=False
    )
    assert reason is not None


async def test_fetch_refuses_a_loopback_url_without_making_a_request() -> None:
    out = await fetch_with_no_network("http://127.0.0.1:11434/api/tags")
    assert "won't fetch" in out


async def fetch_with_no_network(url: str) -> str:
    """If the guard fails to fire this raises rather than silently hitting the network."""

    class Exploding:
        def __init__(self, *a, **k) -> None: ...
        async def __aenter__(self):
            raise AssertionError("a request was made for a URL the guard should have refused")
        async def __aexit__(self, *a) -> None: ...

    import httpx

    original = httpx.AsyncClient
    httpx.AsyncClient = Exploding  # type: ignore[misc,assignment]
    try:
        return await web.fetch(url)
    finally:
        httpx.AsyncClient = original  # type: ignore[misc]


async def test_empty_inputs_are_handled() -> None:
    assert "No URL" in await web.fetch("   ")
    assert "No search query" in await web.search("   ")


async def test_a_403_is_explained_as_the_config_default_it_is() -> None:
    out = await _search_returning(403, {})
    assert "search.formats" in out and "json" in out


async def test_an_unreachable_searxng_says_how_to_start_it() -> None:
    import httpx

    class Failing:
        def __init__(self, *a, **k) -> None: ...
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a) -> None: ...
        async def get(self, *a, **k):
            raise httpx.ConnectError("connection refused")

    original = httpx.AsyncClient
    httpx.AsyncClient = Failing  # type: ignore[misc,assignment]
    try:
        out = await web.search("anything")
    finally:
        httpx.AsyncClient = original  # type: ignore[misc]
    assert "not reachable" in out and "up.sh" in out


async def test_results_are_formatted_with_titles_and_urls() -> None:
    payload = {
        "results": [
            {"title": "First", "url": "https://a.example/1", "content": "snippet one"},
            {"title": "Second", "url": "https://b.example/2", "content": "snippet two"},
        ]
    }
    out = await _search_returning(200, payload)
    assert "First" in out and "https://a.example/1" in out and "snippet one" in out


async def test_no_results_says_so() -> None:
    assert "No results" in await _search_returning(200, {"results": []})


async def test_max_results_is_respected() -> None:
    payload = {"results": [{"title": f"t{i}", "url": f"https://x/{i}"} for i in range(20)]}
    out = await _search_returning(200, payload, config=web.WebConfig(max_results=3))
    assert out.count("https://x/") == 3


async def _search_returning(status: int, payload: dict, config=None) -> str:
    import httpx

    class Response:
        status_code = status

        def json(self) -> dict:
            return payload

    class Client:
        def __init__(self, *a, **k) -> None: ...
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a) -> None: ...
        async def get(self, *a, **k):
            return Response()

    original = httpx.AsyncClient
    httpx.AsyncClient = Client  # type: ignore[misc,assignment]
    try:
        return await web.search("q", config)
    finally:
        httpx.AsyncClient = original  # type: ignore[misc]


def test_web_tools_declare_the_web_family() -> None:
    assert {t.family for t in web.tools()} == {"web"}
    assert {t.name for t in web.tools()} == {"web_search", "fetch_url"}


# --- the console's health pill -------------------------------------------------


def test_an_empty_query_probe_reads_as_healthy() -> None:
    """SearXNG answers `q=` with 400 "No query" — after it has cleared the format gate.

    That is what makes the cheap probe possible: reaching the missing-query error proves
    `json` is an allowed format, without running a federated search to find out. The
    probe this replaced searched for "ping", took 2.6 s against 11 ms, and reported a
    healthy backend as unreachable whenever a cold instance overran the timeout.
    """
    assert api.web_health(400) == "ok"
    assert api.web_health(200) == "ok"


def test_the_json_format_trap_is_still_caught() -> None:
    """The one misconfiguration worth a red pill: `json` missing from search.formats."""
    assert api.web_health(403) == "no json format"


def test_an_unexpected_status_is_reported_verbatim() -> None:
    """Not silently healthy — an unknown code is news, and guessing at it hides it."""
    assert api.web_health(502) == "http 502"
