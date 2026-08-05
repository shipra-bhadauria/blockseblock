"""
Production middleware: request tracing and latency recording.

WHAT YOU BUILT → FRAMEWORK EQUIVALENT:
  RequestIDMiddleware  → OpenTelemetry trace context propagation
  TimingMiddleware     → Prometheus request_duration_seconds histogram middleware
  get_request_id()     → OpenTelemetry span.get_span_context().trace_id

Two middleware classes are provided:
  RequestIDMiddleware  — assigns a UUID to every request and echoes it back
                         in the X-Request-ID response header so callers can
                         correlate logs with specific requests.
  TimingMiddleware     — measures wall-clock latency for every request and
                         records it to shared/metrics.py. Also flags 5xx
                         responses as errors for the error-rate counter.

Usage in main.py:
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(TimingMiddleware)

The middleware order matters: RequestIDMiddleware should be added first (outermost)
so the request_id is available when TimingMiddleware records the request.
"""
import time
import uuid
from contextvars import ContextVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from shared import metrics

# ContextVar lets us carry the request_id through the async call chain
# without threading issues (each coroutine has its own copy).
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Return the X-Request-ID for the currently executing request."""
    return _request_id_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique ID to every request.

    If the caller sends X-Request-ID, we honour it (useful for distributed
    tracing where the frontend generates the ID). Otherwise we generate a
    fresh UUID. The ID is written to:
      - request.state.request_id  — readable by endpoint handlers
      - the X-Request-ID response header — returned to the caller
      - a ContextVar  — accessible via get_request_id() anywhere in the call chain
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        _request_id_var.set(request_id)
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measures wall-clock latency for every request and records it to metrics.

    Only records /api/* paths — skips static file serving which would inflate
    the request count without representing actual API workload.
    """

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        start = time.perf_counter()
        had_error = False
        try:
            response: Response = await call_next(request)
            had_error = response.status_code >= 500
            return response
        except Exception:
            had_error = True
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_request(duration_ms=duration_ms, had_error=had_error)
