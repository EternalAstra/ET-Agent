"""Stub: Session context helpers (gateway removed)."""
from contextvars import ContextVar

_current_session_id: ContextVar = ContextVar("current_session_id", default=None)

def set_current_session_id(session_id: str | None) -> None:
    _current_session_id.set(session_id)

def get_current_session_id() -> str | None:
    return _current_session_id.get(None)

def get_session_env() -> dict:
    """Return empty session env since gateway is not available."""
    return {}

class SessionContext:
    """Minimal stub."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
