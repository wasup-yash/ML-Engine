from contextlib import contextmanager
from typing import Iterator


@contextmanager
def request_span(name: str) -> Iterator[None]:
    """Create an OpenTelemetry span when the optional SDK is installed."""
    try:
        from opentelemetry import trace

        with trace.get_tracer("ml-serving-engine").start_as_current_span(name):
            yield
    except ImportError:
        yield
