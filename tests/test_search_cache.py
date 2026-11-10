"""Eval search cache: replays stored results, searches live only on a miss,
and never stores a failure."""

import pytest

from evals.search_cache import CachedSearchClient


class CountingSearch:
    def __init__(self, fail_on=()):
        self.calls, self.fail_on = [], set(fail_on)

    def search(self, query, max_results=5):
        self.calls.append((query, max_results))
        if query in self.fail_on:
            raise RuntimeError("quota")
        return [{"title": query, "url": f"https://{len(self.calls)}.example", "content": ""}]


def test_a_repeated_query_is_replayed_not_searched(tmp_path):
    inner = CountingSearch()
    cache = CachedSearchClient(inner, tmp_path / "c.json")
    first = cache.search("q")
    assert cache.search("q") == first
    assert len(inner.calls) == 1 and (cache.hits, cache.misses) == (1, 1)


def test_saved_results_are_replayed_by_a_later_run(tmp_path):
    path = tmp_path / "cache" / "c.json"  # parent dir created on save
    first = CachedSearchClient(CountingSearch(), path)
    stored = first.search("q")
    first.save()

    inner = CountingSearch()
    second = CachedSearchClient(inner, path)
    assert second.search("q") == stored
    assert inner.calls == []


def test_max_results_is_part_of_the_key(tmp_path):
    inner = CountingSearch()
    cache = CachedSearchClient(inner, tmp_path / "c.json")
    cache.search("q", max_results=3)
    cache.search("q", max_results=5)
    assert inner.calls == [("q", 3), ("q", 5)]


def test_a_failed_search_is_not_stored(tmp_path):
    # Storing it would replay a quota error on every later run.
    inner = CountingSearch(fail_on={"q"})
    cache = CachedSearchClient(inner, tmp_path / "c.json")
    with pytest.raises(RuntimeError):
        cache.search("q")
    inner.fail_on.clear()
    cache.search("q")
    assert len(inner.calls) == 2 and cache.misses == 1
