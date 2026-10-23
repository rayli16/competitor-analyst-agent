import pytest
from tavily.errors import BadRequestError, ForbiddenError, UsageLimitExceededError

from app.tools.tavily_search import TavilySearchClient, get_tavily_api_keys

RESULT = {"results": [{"title": "T", "url": "https://a.com", "content": "c", "score": 0.9}]}


class FakeTavily:
    """Stands in for TavilyClient: fails with `error` if given, else returns RESULT."""

    def __init__(self, api_key: str, behaviors: dict) -> None:
        self.api_key = api_key
        self.behaviors = behaviors
        self.calls = 0

    def search(self, query, **kwargs):
        self.calls += 1
        error = self.behaviors.get(self.api_key)
        if error is not None:
            raise error
        return RESULT


def make_client(behaviors: dict, keys: list[str]) -> tuple[TavilySearchClient, dict[str, FakeTavily]]:
    fakes: dict[str, FakeTavily] = {}

    def factory(api_key):
        fakes[api_key] = FakeTavily(api_key, behaviors)
        return fakes[api_key]

    return TavilySearchClient(api_keys=keys, client_factory=factory), fakes


def test_primary_key_used_when_healthy():
    client, fakes = make_client({}, ["k1", "k2"])
    assert client.search("q") == [{"title": "T", "url": "https://a.com", "content": "c"}]
    assert fakes["k1"].calls == 1
    assert fakes["k2"].calls == 0


def test_quota_exhausted_falls_over_to_next_key():
    client, fakes = make_client({"k1": ForbiddenError("plan limit")}, ["k1", "k2"])
    assert client.search("q")[0]["url"] == "https://a.com"
    assert fakes["k2"].calls == 1


def test_switch_is_sticky_so_dead_key_is_not_retried_first():
    client, fakes = make_client({"k1": ForbiddenError("plan limit")}, ["k1", "k2"])
    client.search("q1")
    client.search("q2")
    client.search("q3")
    assert fakes["k1"].calls == 1  # only the first search paid for discovering it's dead
    assert fakes["k2"].calls == 3


def test_wraps_around_to_earlier_key_when_later_one_fails():
    behaviors = {"k1": ForbiddenError("plan limit")}
    client, fakes = make_client(behaviors, ["k1", "k2"])
    client.search("q1")  # now sticky on k2
    behaviors["k1"] = None  # k1 refilled (e.g. monthly reset)
    behaviors["k2"] = UsageLimitExceededError("rate limited")
    assert client.search("q2")[0]["url"] == "https://a.com"
    assert fakes["k1"].calls == 2


def test_all_keys_exhausted_raises_the_real_error():
    client, _ = make_client({"k1": ForbiddenError("a"), "k2": ForbiddenError("b")}, ["k1", "k2"])
    with pytest.raises(ForbiddenError):
        client.search("q")


def test_non_key_error_is_raised_without_trying_other_keys():
    client, fakes = make_client({"k1": BadRequestError("bad query")}, ["k1", "k2"])
    with pytest.raises(BadRequestError):
        client.search("q")
    assert fakes["k2"].calls == 0


def test_keys_parsed_from_comma_list(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEYS", " k1 , k2,, ")
    assert get_tavily_api_keys() == ["k1", "k2"]


def test_single_key_variable_still_supported(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEYS", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "only")
    assert get_tavily_api_keys() == ["only"]


def test_no_keys_fails_fast(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEYS", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        get_tavily_api_keys()
