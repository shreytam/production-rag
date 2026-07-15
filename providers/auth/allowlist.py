"""Per-tenant acl_tag allowlists. Default is NullAllowlist (claims pass through);
StaticAllowlist intersects claimed tags against a per-tenant grant (least privilege)."""

from __future__ import annotations

import json
from pathlib import Path


class NullAllowlist:
    def allowed(self, tenant_id: str) -> frozenset[str] | None:
        return None


class StaticAllowlist:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._map = {t: frozenset(tags) for t, tags in mapping.items()}

    @classmethod
    def from_file(cls, path: str) -> "StaticAllowlist":
        data = json.loads(Path(path).read_text())
        return cls(data)

    def allowed(self, tenant_id: str) -> frozenset[str] | None:
        # Unknown tenant → no permitted tags (fail closed / least privilege).
        return self._map.get(tenant_id, frozenset())


def apply_allowlist(allowlist, tenant_id: str, acl_tags: tuple[str, ...]) -> tuple[str, ...]:
    permitted = allowlist.allowed(tenant_id)
    if permitted is None:
        return tuple(acl_tags)
    return tuple(t for t in acl_tags if t in permitted)
