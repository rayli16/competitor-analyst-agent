"""map_in_context must carry the caller's trace context into worker threads,
or Langfuse records every parallel call as its own root trace."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry import context, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from app.discovery import map_in_context

PARENT = SpanContext(
    trace_id=0x1234567890ABCDEF1234567890ABCDEF,
    span_id=0x1234567890ABCDEF,
    is_remote=False,
    trace_flags=TraceFlags(TraceFlags.SAMPLED),
)


def current_span_context(_item=None) -> SpanContext:
    return trace.get_current_span().get_span_context()


@pytest.fixture
def parent_span_active():
    # A non-recording span is enough: nesting only depends on which span is
    # current in the context, not on a tracer provider exporting anything.
    token = context.attach(trace.set_span_in_context(NonRecordingSpan(PARENT)))
    yield
    context.detach(token)


def test_plain_thread_pool_loses_the_parent_span(parent_span_active):
    # The bug being fixed, pinned down: if this ever starts passing on its
    # own, Python changed and map_in_context may no longer be needed.
    with ThreadPoolExecutor(max_workers=2) as pool:
        seen = list(pool.map(current_span_context, [1, 2]))
    assert all(not s.is_valid for s in seen)


def test_map_in_context_carries_the_parent_span_into_every_worker(parent_span_active):
    seen = map_in_context(current_span_context, list(range(8)), max_workers=4)
    assert seen == [PARENT] * 8


def test_map_in_context_preserves_input_order():
    assert map_in_context(lambda x: x * 10, [3, 1, 2], max_workers=3) == [30, 10, 20]
