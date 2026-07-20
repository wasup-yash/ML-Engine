import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional


request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="-")

_LOG_LEVEL = logging.INFO
_LOGGERS = {}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "tenant_id": tenant_id_var.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: Optional[str] = None) -> None:
    global _LOG_LEVEL
    if level:
        _LOG_LEVEL = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(_LOG_LEVEL)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    if name not in _LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(_LOG_LEVEL)
        _LOGGERS[name] = logger
    return _LOGGERS[name]
