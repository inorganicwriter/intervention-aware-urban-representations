"""Explicit and opt-in local proxy configuration."""

import os
import socket

_PROXY_CACHE: str | None = None

_PROXY_DETECTED = False


def detect_proxy() -> str | None:
    """Return proxy URL if explicitly configured, else None.

    Checks the HTTPS_PROXY/HTTP_PROXY env var first, then an optional
    MIT_AUTO_PROXY_PORT variable for environments that run a local proxy.
    Automatic TCP probing of arbitrary ports is forbidden per
    code_standards.md §1.
    """
    env_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if env_proxy:
        return env_proxy
    auto_port = os.environ.get("MIT_AUTO_PROXY_PORT")
    if not auto_port:
        return None
    try:
        with socket.create_connection(("127.0.0.1", int(auto_port)), timeout=1):
            return f"http://127.0.0.1:{auto_port}"
    except OSError:
        return None


def get_proxy() -> str | None:
    """Lazily detect and cache the proxy URL (singleton)."""
    global _PROXY_CACHE, _PROXY_DETECTED
    if not _PROXY_DETECTED:
        _PROXY_CACHE = detect_proxy()
        _PROXY_DETECTED = True
    return _PROXY_CACHE


def get_proxies() -> dict:
    """Return ``{"http": url, "https": url}`` for requests, or empty dict.

    Usage:
        import requests
        from urban_intervention.config.project import get_proxies
        resp = requests.get(url, proxies=get_proxies(), ...)
    """
    p = get_proxy()
    return {"http": p, "https": p} if p else {}


def set_proxy(proxy_url: str | None) -> None:
    """Override the detected proxy (used by --proxy CLI argument)."""
    global _PROXY_CACHE, _PROXY_DETECTED
    _PROXY_CACHE = proxy_url
    _PROXY_DETECTED = True
