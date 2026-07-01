"""Stub: Gateway config (gateway removed)."""
from enum import Enum

class Platform(Enum):
    """Stub Platform enum. Only CLI is supported."""
    CLI = "cli"

def load_gateway_config(*args, **kwargs):
    """Stub: raise error indicating gateway is not available."""
    raise RuntimeError(
        "Gateway has been removed in this trimmed version. "
        "Use CLI mode only (hermes chat)."
    )
