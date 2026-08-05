"""
Structured JSON logging for production.

Replaces the default Python log format (human-readable plain text) with
machine-readable JSON lines. Each log line is a single JSON object that
log aggregators (Datadog, Loki, CloudWatch) can parse and index automatically.

WHAT YOU BUILT → FRAMEWORK EQUIVALENT:
  JSONFormatter      → structlog.processors.JSONRenderer (structlog library)
  setup_logging()    → structlog.configure() or logging.config.dictConfig()
  X-Request-ID flow  → structlog.contextvars.bind_contextvars()

Why structured logs matter in production:
  Plain text: "ERROR 2024-03-01 12:34:56 Something went wrong"
  JSON:       {"timestamp":"2024-03-01T12:34:56Z","level":"ERROR","request_id":"abc",
               "path":"/api/chat","duration_ms":234,"message":"Something went wrong"}
  The JSON version is searchable, filterable, and alertable in any log platform.
"""
import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        # Middleware injects these extras via LogRecord.
        for extra_key in ("request_id", "path", "method", "duration_ms", "status_code"):
            if hasattr(record, extra_key):
                payload[extra_key] = getattr(record, extra_key)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger to emit JSON-formatted log lines to stdout.
    Call once at application startup before the first request is served.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers = [handler]
