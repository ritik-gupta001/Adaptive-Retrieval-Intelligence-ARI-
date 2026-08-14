import logging
import sys
from contextvars import ContextVar
from typing import Optional

from pythonjsonlogger import jsonlogger

from app.config.settings import settings

# Carries a run_id across every node in a single graph invocation without
# threading it through every function signature manually.
current_run_id: ContextVar[Optional[str]] = ContextVar("current_run_id", default=None)


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = current_run_id.get() or "no-run-id"
        return True


_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(run_id)s %(message)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(RunIdFilter())

    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers = [handler]
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
