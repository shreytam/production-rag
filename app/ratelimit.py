"""Per-tenant fixed-window rate limiting for the HTTP API.

Backed by Redis (INCR + EXPIRE in one atomic Lua script) so the limit holds
across API replicas; an in-memory backend serves dev/tests. Keyed on the
verified principal's tenant_id, never on client-supplied data.

Failure policy: FAIL OPEN. If Redis is unreachable the request is allowed and a
warning is logged — a limiter outage must not become an API outage.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from fastapi import Depends, HTTPException

from app.auth import require_principal
from core.config import get_settings
from core.types import Principal

log = logging.getLogger(__name__)

WINDOW_SECONDS = 60

# KEYS[1]=counter key, ARGV[1]=window seconds. Returns {count, ttl_seconds}.
_LUA = """
local c = redis.call('INCR', KEYS[1])
if c == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
local ttl = redis.call('TTL', KEYS[1])
if ttl < 0 then redis.call('EXPIRE', KEYS[1], ARGV[1]); ttl = tonumber(ARGV[1]) end
return {c, ttl}
"""


class MemoryRateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[float, int]] = {}  # key -> (window_end, count)

    async def hit(self, key: str, limit: int, window: int = WINDOW_SECONDS) -> int:
        """Record a hit. Returns 0 if allowed, else seconds until the window resets."""
        now = self._clock()
        with self._lock:
            end, count = self._windows.get(key, (0.0, 0))
            if now >= end:
                end, count = now + window, 0
            count += 1
            self._windows[key] = (end, count)
            if count > limit:
                return max(1, int(end - now + 0.999))
        return 0


class RedisRateLimiter:
    def __init__(self, url: str, password: str = "") -> None:
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(url, password=password or None)
        self._script = self._client.register_script(_LUA)

    async def hit(self, key: str, limit: int, window: int = WINDOW_SECONDS) -> int:
        try:
            count, ttl = await self._script(keys=[f"ratelimit:{key}"], args=[window])
        except Exception as exc:  # fail open: availability over strictness
            log.warning("rate limiter unavailable, failing open: %s", exc)
            return 0
        if int(count) > limit:
            return max(1, int(ttl))
        return 0


_limiter = None


def get_limiter():
    global _limiter
    if _limiter is None:
        s = get_settings()
        if s.rate_limit_backend == "memory":
            _limiter = MemoryRateLimiter()
        else:
            _limiter = RedisRateLimiter(s.redis_url, s.redis_password)
    return _limiter


def rate_limit(scope: str, limit_attr: str):
    """Build a dependency enforcing the per-tenant limit named by a Settings field."""

    async def dependency(
        principal: Principal = Depends(require_principal),
        limiter=Depends(get_limiter),
    ) -> None:
        settings = get_settings()
        if not settings.rate_limit_enabled:
            return
        limit = getattr(settings, limit_attr)
        retry_after = await limiter.hit(f"{scope}:{principal.tenant_id}", limit)
        if retry_after:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

    return dependency


query_rate_limit = rate_limit("query", "rate_limit_query_per_minute")
upload_rate_limit = rate_limit("upload", "rate_limit_upload_per_minute")
