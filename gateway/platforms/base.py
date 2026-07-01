"""Stub: Base platform adapter and helpers (gateway removed)."""

class BasePlatformAdapter:
    """Stub base adapter — raises on use."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Gateway platforms removed in trimmed version")

def cache_image_from_bytes(*args, **kwargs):
    raise NotImplementedError("Gateway platforms removed")

def utf16_len(s: str) -> int:
    return len(s)

def resolve_proxy_url(*args, **kwargs):
    return None

def proxy_kwargs_for_aiohttp(*args, **kwargs):
    return {}
