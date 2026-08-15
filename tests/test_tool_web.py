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


# --- the direct-answer block ---------------------------------------------------
#
# `_summarise` reads the two fields SearXNG uses to carry a fact rather than a page, and
# `search` was previously dropping both. The infobox below is the real response for
# "capital of Nigeria" from the wikipedia engine, trimmed; its shape matters, because
# `title` is present and empty there while the name sits under `infobox`.


def _infobox_payload() -> dict:
    return {
        "results": [
            {"title": "Abuja - Wikipedia", "url": "https://en.wikipedia.org/wiki/Abuja",
             "content": "Abuja is the capital of Nigeria."},
        ],
        "infoboxes": [
            {
                "infobox": "Abuja",
                "title": "",
                "engine": "wikipedia",
                "id": "https://en.wikipedia.org/wiki/Abuja",
                "content": "Abuja is the capital city of Nigeria, situated at the "
                           "geographic midpoint of the country.",
                "attributes": [{"label": "Population", "value": "3,652,000"}],
            }
        ],
        "answers": [],
    }


def test_an_infobox_becomes_a_labelled_summary() -> None:
    summary = web._summarise(_infobox_payload())
    assert summary is not None
    assert "Summary of Abuja (from wikipedia)" in summary
    assert "capital city of Nigeria" in summary
    assert "Population: 3,652,000" in summary
    assert "source: https://en.wikipedia.org/wiki/Abuja" in summary


def test_the_subject_name_is_read_from_infobox_not_title() -> None:
    """The wikipedia engine sends `title: ""`, so reading it first labels everything "".""" 
    payload = _infobox_payload()
    assert payload["infoboxes"][0]["title"] == ""
    assert "Summary of Abuja" in (web._summarise(payload) or "")


def test_a_plain_string_answer_is_read() -> None:
    """Plugins hand back a bare string. Synthetic: no plugin on this instance emits one
    (see `_summarise`), so this pins the contract rather than reproducing an observation."""
    assert "Direct answer: 12" in (web._summarise({"answers": ["12"]}) or "")


def test_an_object_answer_is_read_too() -> None:
    summary = web._summarise({"answers": [{"answer": "1.428 billion", "url": "x"}]})
    assert "Direct answer: 1.428 billion" in (summary or "")


def test_a_response_with_no_direct_answer_summarises_to_nothing() -> None:
    """"last f1 race" is this case: 39 results, no infobox, no answer. The caller has to
    get None back so it falls through to the ranked list unchanged."""
    assert web._summarise({"results": [{"title": "F1"}], "answers": [], "infoboxes": []}) is None


def test_an_empty_infobox_content_is_not_reported_as_a_summary() -> None:
    assert web._summarise({"infoboxes": [{"infobox": "X", "content": ""}]}) is None


def test_a_long_infobox_is_capped() -> None:
    summary = web._summarise({"infoboxes": [{"infobox": "X", "content": "y" * 5000}]})
    assert summary is not None and "y" * web.MAX_SUMMARY_CHARS in summary
    assert "y" * (web.MAX_SUMMARY_CHARS + 1) not in summary


# --- how search() composes the two halves --------------------------------------


def _stub_searxng(monkeypatch, payload: dict) -> None:
    """Answer any SearXNG request with `payload`, without a socket.

    A MockTransport rather than a patched `search`: it exercises the real status-code
    and JSON handling above the part under test, so a change there cannot pass silently.
    """
    import httpx

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    original = httpx.AsyncClient

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client)


async def test_the_summary_is_placed_above_the_ranked_results(monkeypatch) -> None:
    """Order is the point. On a 1.2B, what comes first in the context is not neutral, and
    the answer arriving after five navigational links is the case that reads badly."""
    _stub_searxng(monkeypatch, _infobox_payload())

    out = await web.search("capital of Nigeria")
    assert out.index("Summary of Abuja") < out.index("1. Abuja - Wikipedia")


async def test_a_direct_answer_survives_an_empty_result_list(monkeypatch) -> None:
    """This used to return "No results" and throw the answer away."""
    _stub_searxng(monkeypatch, {"results": [], "answers": ["42"], "infoboxes": []})

    out = await web.search("the answer")
    assert "Direct answer: 42" in out
    assert "No results" not in out


async def test_a_search_with_no_direct_answer_is_unchanged(monkeypatch) -> None:
    """The "last f1 race" shape. Nothing is added, so the existing behaviour is intact."""
    _stub_searxng(
        monkeypatch,
        {"results": [{"title": "F1", "url": "https://formula1.com", "content": "Home"}],
         "answers": [], "infoboxes": []},
    )

    out = await web.search("last f1 race")
    assert out == "Search results for 'last f1 race':\n\n1. F1\n   https://formula1.com\n   Home"


async def test_a_genuinely_empty_response_still_says_so(monkeypatch) -> None:
    _stub_searxng(monkeypatch, {"results": [], "answers": [], "infoboxes": []})
    assert "No results for 'nothing at all'" in await web.search("nothing at all")


# --- the query is sent unscoped, and that is a measured decision --------------------


async def test_the_search_request_is_deliberately_unscoped(monkeypatch) -> None:
    """`engines`, `categories` and `time_range` were probed and none of them shipped.

    This pins the outgoing parameters so the rejection is enforced rather than merely
    written down. The measurement is in `app/tools/web.py` next to the request; the short
    version is that `time_range` returned zero results on 10 of 10 recency queries,
    category scoping traded one failure mode for another, and engine pinning bought
    nothing. If a later change adds a parameter here, this test is the prompt to go and
    read why the last attempt did not.
    """
    import httpx

    seen: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json={"results": [], "answers": [], "infoboxes": []})

    transport = httpx.MockTransport(_handler)
    original = httpx.AsyncClient

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client)

    await web.search("who won the last f1 race")

    assert seen == {"q": "who won the last f1 race", "format": "json", "safesearch": "0"}
