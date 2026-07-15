"""Mint a dev JWT for local testing / the demo.

    uv run python -m scripts.mint_token --tenant acme --tags finance,hr --ttl 3600

Requires auth_dev_signer_enabled=true and a jwt_secret (HS256). A prod instance
(RS256, public-key-only) cannot mint and this will refuse.
"""

from __future__ import annotations

import argparse

from core.config import Settings, get_settings
from providers.auth.dev_signer import mint_token


def build_token(settings: Settings, *, tenant: str, tags: list[str], ttl: int) -> str:
    if not settings.auth_dev_signer_enabled or not settings.jwt_secret:
        raise RuntimeError(
            "minting requires auth_dev_signer_enabled=true and a jwt_secret (HS256)"
        )
    return mint_token(
        tenant_id=tenant,
        acl_tags=tags,
        secret=settings.jwt_secret,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        ttl_seconds=ttl,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint a dev JWT.")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--tags", default="", help="comma-separated acl_tags")
    parser.add_argument("--ttl", type=int, default=3600)
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    token = build_token(get_settings(), tenant=args.tenant, tags=tags, ttl=args.ttl)
    print(token)


if __name__ == "__main__":
    main()
