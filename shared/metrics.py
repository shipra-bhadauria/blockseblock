"""
In-memory metrics store for the Feature 11 observability layer.

WHAT YOU BUILT → FRAMEWORK EQUIVALENT:
  record_request()   → Prometheus Counter + Histogram .observe()
  get_metrics()      → Prometheus /metrics scrape endpoint (text exposition)
  set_eval_result()  → Prometheus Gauge for eval pass rate
  _metrics dict      → Prometheus in-process registry

Why in-memory instead of Prometheus?
  Prometheus requires a separate scrape server and pull-based architecture.
  This in-memory store is simpler to teach and still demonstrates the concepts:
  counters, rates, and derived metrics (avg latency = total / count).
  In production, swap this module for prometheus-client or OpenTelemetry SDK.

Thread safety: Python's GIL protects individual dict operations, but the
read-modify-write pattern (e.g., x += 1) is not atomic under the GIL.
We use a threading.Lock for correctness even in single-worker deployments,
since FastAPI runs in a threadpool for sync endpoints.
"""
import threading
from typing import Any

_lock = threading.Lock()

_metrics: dict[str, Any] = {
    "total_requests":   0,
    "total_errors":     0,
    "total_latency_ms": 0.0,
    "eval_last_run":    None,
}


def record_request(duration_ms: float, had_error: bool = False) -> None:
    """Increment request counter and accumulate latency. Called by TimingMiddleware."""
    with _lock:
        _metrics["total_requests"]   += 1
        _metrics["total_latency_ms"] += duration_ms
        if had_error:
            _metrics["total_errors"] += 1


def set_eval_result(report: dict) -> None:
    """Store the most recent eval report so GET /api/eval/last can retrieve it."""
    with _lock:
        _metrics["eval_last_run"] = report


def get_metrics() -> dict:
    """
    Return a snapshot of current metrics with derived values computed.
    avg_latency_ms and error_rate are computed on read so the store stays simple.
    """
    with _lock:
        total = _metrics["total_requests"]
        avg_ms    = (_metrics["total_latency_ms"] / total) if total > 0 else 0.0
        error_rate = (_metrics["total_errors"] / total)    if total > 0 else 0.0
        return {
            "total_requests":  total,
            "total_errors":    _metrics["total_errors"],
            "avg_latency_ms":  round(avg_ms, 2),
            "error_rate":      round(error_rate, 4),
            "eval_last_run":   _metrics["eval_last_run"],
        }
