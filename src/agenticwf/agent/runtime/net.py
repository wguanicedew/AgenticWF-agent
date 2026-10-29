"""
Outbound network rules for anything users point the agent at: their own models (base_url) and
their own MCP endpoints. Operators' configuration is trusted and not checked here.

    allowed_hosts   exact names ("mcp.example.org", "10.1.2.3"), suffixes ("*.example.org"), or "*"
    internal        loopback, private, link-local (cloud metadata), multicast, reserved addresses

A host the operator lists by name may be internal (e.g. a vLLM server on the cluster network);
a host that only matches a wildcard may not.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Iterable


def host_allowed(host: str, allowed: Iterable[str]) -> bool:
    host = (host or "").lower().rstrip(".")
    for pattern in (p.lower().strip() for p in allowed):
        if pattern == "*" or host == pattern or (pattern.startswith("*.") and host.endswith(pattern[1:])):
            return bool(host)
    return False


def listed_by_name(host: str, allowed: Iterable[str]) -> bool:
    """The operator named this host itself (not through a wildcard)."""
    host = (host or "").lower().rstrip(".")
    return any(host == p.lower().strip() for p in allowed)


def is_internal(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return True   # not an address we can judge: refuse
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
            or ip.is_unspecified)


def resolves_internal(host: str, port: int = 443) -> bool:
    """True if any address `host` resolves to is internal (or it does not resolve)."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError:
        return True
    return any(is_internal(info[4][0]) for info in infos)


def check_host(host: str, allowed: Iterable[str], what: str) -> str | None:
    """None if a user-supplied `host` may be reached, else why not."""
    allowed = list(allowed)
    if not host_allowed(host, allowed):
        return f"host {host!r} is not on the allow-list for {what}"
    if not listed_by_name(host, allowed) and resolves_internal(host):
        return f"host {host!r} is an internal address; an operator must list it by name to allow it"
    return None
