"""Search cache for evals, so runs compare code changes on fixed inputs
instead of on whatever live search returned that day.

A hit returns the stored results; a miss searches live and stores them.
Misses are expected: only discovery's 10 fixed queries are the same every
run, while homepage and profile queries are built from company names the
model extracts, which can differ between runs. hits/misses say how much of a
run was replayed.

The cache file holds real search results about real companies, so it lives
under the gitignored evals/cache/.
"""

import json
import threading
from pathlib import Path
from typing import Protocol


class SearchClient(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]: ...


class CachedSearchClient:
    def __init__(self, inner: SearchClient, path: Path) -> None:
        self._inner = inner
        self.path = path
        self._entries: dict[str, list[dict[str, str]]] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(query: str, max_results: int) -> str:
        # max_results is part of the key: the same query asked for 3 results
        # must not be answered with a stored list of 5, or the reverse.
        return json.dumps([query, max_results], ensure_ascii=False)

    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        key = self._key(query, max_results)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self.hits += 1
                return cached
        # Outside the lock, so concurrent misses search in parallel. A failed
        # search raises before anything is stored: caching a quota error would
        # replay that failure on every later run.
        results = self._inner.search(query, max_results=max_results)
        with self._lock:
            self._entries[key] = results
            self.misses += 1
        return results

    def save(self) -> None:
        with self._lock:
            data = json.dumps(self._entries, ensure_ascii=False, indent=1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename, so an interrupted save can't leave a truncated
        # file that breaks the next run.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(self.path)
