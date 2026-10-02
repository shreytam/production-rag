"""infra/.env.example must document every Settings field.

A missing key fails closed and silently for operators (the key is never set, the
default applies unnoticed). Commented-out `# KEY=` lines count as documented.
Extra keys in the example are fine.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / "infra" / ".env.example"


def _documented_keys() -> set[str]:
    keys: set[str] = set()
    for line in ENV_EXAMPLE.read_text().splitlines():
        m = re.match(r"^\s*#?\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if m:
            keys.add(m.group(1).upper())
    return keys


def _field_keys(name: str, field) -> set[str]:
    keys = {name.upper()}
    alias = field.validation_alias
    if alias is not None and hasattr(alias, "choices"):
        keys |= {c.upper() for c in alias.choices if isinstance(c, str)}
    return keys


def test_every_settings_field_is_in_env_example():
    documented = _documented_keys()
    missing = sorted(
        name.upper()
        for name, field in Settings.model_fields.items()
        if not (_field_keys(name, field) & documented)
    )
    assert not missing, f"infra/.env.example is missing Settings keys: {missing}"


def test_wheel_packages_cover_every_top_level_package():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    cfg = tomllib.loads((root / "pyproject.toml").read_text())
    packages = set(cfg["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    on_disk = {p.parent.name for p in root.glob("*/__init__.py") if p.parent.name not in {"tests"}}
    assert on_disk <= packages, f"missing from wheel: {sorted(on_disk - packages)}"
