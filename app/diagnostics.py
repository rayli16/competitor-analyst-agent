"""Partial-success reporting. A failed search is skipped rather than allowed
to sink a run, but the caller still needs to hear about it: a result built on
missing data must say so, or it reads as a complete answer. Every case is a
200 with an entry in `warnings` — like a search engine returning hits plus a
failed-shard count, rather than failing the whole query."""

import threading
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

WarningCode = Literal[
    "search_unavailable",  # every search failed: an empty result means no data, not no competitors
    "partial_search_failure",  # some searches failed; results are built on what came back
    "evidence_unavailable",  # a company's profile searches all failed, so it wasn't scored
    "classification_incomplete",  # the model skipped some search results when classifying
    "scoring_incomplete",  # the model skipped some companies when scoring
    "no_sources",  # deep-dive gathered no sources, so no report was generated
    "iteration_cap_hit",  # deep-dive research stopped at the iteration cap, not by choice
]

Severity = Literal["error", "warning", "info"]

# Fixed per code, so every client styles the same situation the same way:
# error = the result is unusable, warning = usable but incomplete,
# info = worth knowing, little effect on the result.
SEVERITY: dict[str, Severity] = {
    "search_unavailable": "error",
    "no_sources": "error",
    "partial_search_failure": "warning",
    "evidence_unavailable": "warning",
    "scoring_incomplete": "warning",
    "classification_incomplete": "info",
    "iteration_cap_hit": "info",
}

SearchStage = Literal["discovery", "profile", "research"]


class AffectedCompany(BaseModel):
    name: str
    url: str


class WarningDetails(BaseModel):
    # Everything a client needs to render the warning lives here as data, so
    # nothing ever has to be parsed back out of `message`. Fields that don't
    # apply to a given code are left at their defaults.
    stage: SearchStage | None = Field(default=None, description="Which step's searches this refers to")
    attempted: int | None = Field(default=None, description="Searches attempted at that stage")
    failed: int | None = Field(default=None, description="Searches that failed at that stage")
    error_types: dict[str, int] = Field(
        default_factory=dict,
        description="Failure counts by error type, e.g. {'ForbiddenError': 10} — lets a client tell a "
        "config problem (InvalidAPIKeyError) from an exhausted quota (ForbiddenError)",
    )
    skipped: int | None = Field(default=None, description="Inputs the model skipped (classification_incomplete)")
    companies: list[AffectedCompany] = Field(default_factory=list, description="Companies this warning is about")


class ResponseWarning(BaseModel):
    # code and details are for programs (a UI branches on code and renders
    # from details); message is for developers and logs, not for end users.
    code: WarningCode = Field(description="Stable machine-readable warning code")
    severity: Severity = Field(description="error: result unusable; warning: incomplete; info: minor")
    message: str = Field(description="Developer-facing explanation (English); UIs should show their own copy")
    details: WarningDetails = Field(default_factory=WarningDetails)


def make_warning(code: WarningCode, message: str, **details) -> ResponseWarning:
    # Severity comes from SEVERITY, never from the call site, so the same
    # code can't end up styled two different ways.
    return ResponseWarning(code=code, severity=SEVERITY[code], message=message, details=WarningDetails(**details))


class SearchTally:
    """Counts search attempts and failures across concurrent worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.attempted = 0
        # Error type names only — raw messages can carry internal detail
        # (URLs, account info) that doesn't belong in an API response.
        self.errors: Counter[str] = Counter()

    def record(self, error: Exception | None = None) -> None:
        with self._lock:
            self.attempted += 1
            if error is not None:
                self.errors[type(error).__name__] += 1

    @property
    def failed(self) -> int:
        return sum(self.errors.values())

    @property
    def all_failed(self) -> bool:
        return self.attempted > 0 and self.failed == self.attempted

    def describe_errors(self) -> str:
        return ", ".join(f"{name} ×{count}" for name, count in self.errors.most_common())

    def as_details(self, stage: SearchStage) -> dict:
        return {"stage": stage, "attempted": self.attempted, "failed": self.failed, "error_types": dict(self.errors)}


def partial_search_failure(stage: SearchStage, tally: SearchTally, impact: str) -> ResponseWarning:
    return make_warning(
        "partial_search_failure",
        f"{tally.failed} of {tally.attempted} {stage} searches failed ({tally.describe_errors()}). {impact}",
        **tally.as_details(stage),
    )
