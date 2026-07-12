# SP1 · Security & Tenancy — Design Spec

**Date:** 2026-07-12
**Status:** Approved (design), pending spec review → writing-plans
**Program:** Production-hardening, Phase 0 (risk-ordered). This is the first slice; nothing downstream is safe without it.

---

## 1. Context & problem

The production-readiness audit (`docs/PRODUCTION_READINESS_AUDIT.md`) found the primary `/query` entrypoint has **no authentication**: `tenant_id` comes from the `X-Tenant-Id` header (or request body) and `acl_tags` are copied verbatim from the body (`app/api.py:77,80`). The store-side ACL filters are *correct* — they faithfully scope every retriever to the tenant they're given — but that tenant is **fully attacker-controlled**. One header (`X-Tenant-Id: victim`) reads any org's entire corpus. A test even certifies the spoofable header as intended behavior, giving false confidence.

This is the #1 critical finding and the gate for the whole program: every later slice assumes a trustworthy identity.

**Tenancy model recap:** a *tenant* is an *org*. Documents are tagged with `tenant_id` at ingest; the ACL filter runs inside the vector-store and BM25 query (before similarity), so each org gets an isolated RAG over its own corpus on one shared deployment (pooled multi-tenancy). `acl_tags` provide finer within-org access (e.g. `finance` vs `hr`).

---

## 2. Goals

- Derive `tenant_id` + `acl_tags` **only** from a cryptographically verified source (a signed JWT), never from client-controlled headers/body.
- Remove every client-controlled identity path from the API.
- Fail closed on every auth failure (missing/malformed/expired/bad-signature token, missing/blank tenant claim).
- Keep the design behind the codebase's existing Protocol + registry pattern so a second auth mechanism (API keys, a different IdP) drops in without touching the pipeline.
- Basic input-abuse caps (question length, tag count).
- Preserve the Streamlit demo as an isolation showcase — now through the real auth path.

## 3. Non-goals (explicitly deferred, to keep this slice focused)

- **Output-guardrail-block leak fix** → SP2 (Guardrail correctness).
- **Rate limiting, request-size caps, CORS/TrustedHost, global exception handler** → SP9 (Deployability) / SP6 (Resilience). SP1 adds only per-field input caps.
- **Org self-service (create org, per-org ingestion API, org metadata)** → SP1.5 (Org & corpus management), if elected.
- **Token revocation / logout** → documented limitation here; real solution (per-tenant `token_version`) lands with the registry.
- **Physical namespace-per-org isolation** → VDB-Decision / SP11. SP1 keeps the current pooled-filter model.
- **A user/identity database, password login, refresh-token rotation** → out of scope; SP1 verifies tokens, it does not issue user sessions (beyond the dev signer).

---

## 4. Decisions locked

| Decision | Choice |
|---|---|
| Auth mechanism | **JWT bearer** (`Authorization: Bearer <jwt>`) |
| Verifier shape | Pluggable `AuthVerifier` **Protocol**; `JWTVerifier` the first impl |
| ACL-tag authority | **Claims authoritative**; optional per-tenant allowlist intersect (off by default) |
| Algorithms | **HS256 for dev** (shared secret), **RS256 + JWKS for prod** (asymmetric); exactly one pinned per deployment |
| Request identity | `tenant_id`/`acl_tags` **deleted** from the request body; body carries only `{ question }` |
| Demo/dev tokens | Dev signer mints HS256 tokens; demo selects an org → mints a real token → sends it |

---

## 5. Architecture & components

New/changed units, each with one clear purpose, communicating through typed interfaces:

### 5.1 `Principal` (type) — `core/types.py`
Immutable verified identity returned by the verifier:
```
Principal(tenant_id: str, acl_tags: tuple[str, ...], subject: str, claims: Mapping)
```
`tenant_id` is validated non-empty at construction. `ACLContext` is built *from* a `Principal`, never from raw request data.

### 5.2 `AuthVerifier` (Protocol) — `core/interfaces.py`
```
class AuthVerifier(Protocol):
    def verify(self, token: str) -> Principal: ...   # raises AuthError on ANY failure
```
`AuthError` is a small typed exception (with an HTTP-status hint: 401 vs 403) so the API layer maps it to a sanitized response without leaking internals.

### 5.3 `JWTVerifier` — `providers/auth/jwt_verifier.py`
- Decodes with **exactly one pinned algorithm**: `algorithms=[settings.jwt_alg]`. Never a multi-element list; never the token's own `alg` header. (Kills algorithm-confusion + `alg:none`.)
- Validates `exp`, `nbf`, `iat` (with `jwt_leeway_seconds`), and — when set — `aud` and `iss`.
- HS256: verifies with `jwt_secret`. RS256: verifies with a public key resolved from **JWKS** (`jwks_url`), selecting by `kid`, caching keys with a TTL, refreshing on unknown `kid`.
- Extracts `tenant_id` and `acl_tags` claims → `Principal`. Missing/blank `tenant_id` → `AuthError(403)`.

### 5.4 `TenantAllowlist` (Protocol) + impls — `providers/auth/allowlist.py`
```
class TenantAllowlist(Protocol):
    def allowed(self, tenant_id: str) -> frozenset[str] | None: ...  # None = unrestricted
```
- `NullAllowlist` — default; `allowed()` returns `None` → claims pass through unchanged.
- `StaticAllowlist` — loads `tenant_id → permitted tags` from a JSON file (`acl_allowlist_source`). When present, effective tags = `claims.acl_tags ∩ allowed(tenant_id)`; tags outside the grant are silently dropped (least privilege). This is the seed of the future org registry.

### 5.5 Registry wiring — `core/registry.py`
`build_auth_verifier(settings)` and `build_allowlist(settings)` — the only place concrete auth classes are named (consistent with every other component).

### 5.6 FastAPI dependency — `app/auth.py`
`require_principal(authorization: str = Header(...)) -> Principal`:
1. Extract the Bearer token (missing/blank → `401` + `WWW-Authenticate: Bearer`).
2. `verifier.verify(token)` → `Principal` (any `AuthError` → sanitized `401`/`403`).
3. Apply the allowlist intersect.
4. Return the `Principal`. The route builds `ACLContext` from it.

### 5.7 Dev signer — `providers/auth/dev_signer.py` + `scripts/mint_token.py`
Mints HS256 tokens for a given `tenant_id`/`acl_tags`/TTL. **Signing capability is tied to `jwt_secret` presence + `auth_dev_signer_enabled`** — a prod instance runs RS256 with only a public key and therefore *cannot* mint, regardless of the flag. `scripts/mint_token.py --tenant acme --tags finance,hr --ttl 3600` prints a ready Bearer token for local/demo use.

---

## 6. Security hardening (best-practice requirements — must-haves)

These are explicit acceptance criteria, not nice-to-haves:

1. **Algorithm pinning.** Decode passes `algorithms=[settings.jwt_alg]` (single element). A test forges an HS256 token signed with the RS256 *public* key and asserts it is **rejected** under RS256 config. `alg:none` tokens rejected.
2. **Audience/issuer enforcement.** On a served (non-dev) instance, boot-validation requires both `jwt_audience` and `jwt_issuer` to be set; tokens with mismatched `aud`/`iss` are rejected. `nbf`/`iat` validated alongside `exp`.
3. **Dev signer structurally impossible in prod.** No HS256 secret ⇒ no minting. Additionally, boot **refuses to start** if `auth_enabled=False` outside an explicit dev environment (`APP_ENV=dev`), so the eval/test auth-off path can never reach a served instance.
4. **Revocation limitation documented.** Stateless JWTs remain valid until `exp`; SP1's answer is short access-token TTL + the checks above. True early revocation (compromised token, de-provisioned org) needs a per-tenant `token_version` checked server-side — noted as a natural extension of the allowlist/registry, not built here.
5. **JWKS robustness.** Keys cached by `kid` with a TTL; unknown `kid` triggers a single refresh; JWKS fetch failure fails closed (`401`, never fail-open).
6. **No secret logging.** Tokens, secrets, and keys never logged. Auth failures log a reason code + subject (if parseable) only.

---

## 7. Data flow

```
POST /query
Authorization: Bearer <jwt>
body: { "question": "..." }            ← tenant_id / acl_tags REMOVED

require_principal:
  token → JWTVerifier.verify()          # sig + alg-pinned + exp/nbf/iat + aud/iss
        → Principal{ tenant_id, acl_tags, subject, claims }
  acl_tags = allowlist.intersect(tenant_id, acl_tags)   # no-op when NullAllowlist
route:
  acl = ACLContext(tenant_id, acl_tags)  # derived ONLY from verified claims
  result = pipeline.run(question, acl)
```
There is **no client-controlled identity path** remaining.

---

## 8. API changes

- `QueryRequest`: drop `tenant_id` and `acl_tags`; keep `question` (with `max_length = max_question_chars`).
- `/query`: `Authorization` header **required**; remove the `x_tenant_id` header param and the body-identity handling.
- Remove/ignore the `X-Tenant-Id` header entirely (no silent honoring).
- Error responses are sanitized JSON (`{ "detail": "...", "request_id": "..." }`), no stack traces: `401` (unauthenticated), `403` (authenticated but no valid tenant, or an over-scoped/malformed token), `422` (user-input caps, e.g. oversized `question`). `401` includes `WWW-Authenticate: Bearer`.

---

## 9. Config knobs (`core/config.py`)

| Knob | Default | Purpose |
|---|---|---|
| `auth_enabled` | `True` | Master switch; `False` only permitted when `APP_ENV=dev` |
| `app_env` | `"dev"` | `dev` \| `prod`; gates dev-only behavior + boot checks |
| `jwt_alg` | `"HS256"` | Pinned algorithm (`HS256` \| `RS256`) |
| `jwt_secret` | `""` | HS256 shared secret (dev); presence enables minting |
| `jwks_url` | `""` | RS256 JWKS endpoint (prod) |
| `jwt_issuer` | `""` | Required + enforced on served instances |
| `jwt_audience` | `""` | Required + enforced on served instances |
| `jwt_leeway_seconds` | `60` | Clock-skew tolerance for exp/nbf/iat |
| `acl_allowlist_source` | `""` | Empty = `NullAllowlist`; path = `StaticAllowlist` JSON |
| `auth_dev_signer_enabled` | `False` | Dev token minting; also requires `jwt_secret` |
| `max_question_chars` | `8000` | User-input cap on `question` → 422 |
| `max_acl_tags` | `32` | Bound on tags in a token claim; exceeding → 403 (over-scoped/malformed token, not a user-input error) |

Boot validation (`Settings` validator + a startup check): served instance (`app_env=prod`) must have `auth_enabled=True`, a key source matching `jwt_alg`, and non-empty `jwt_issuer`/`jwt_audience`; otherwise refuse to start.

New dependency: `pyjwt[crypto]` (added to the `app` extra).

---

## 10. Demo & dev

The Streamlit demo keeps its org selector, but selecting an org now **mints a real signed token** (dev signer) and sends it as a Bearer header — so the demo still showcases org switching **and** demonstrates that a token for org A cannot retrieve org B's data. Prod builds run RS256 public-key-only, so the demo's minting path is inert there.

---

## 11. Testing (TDD — red first)

Written before implementation, asserting the real behavior:

- Spoofed `X-Tenant-Id` header is ignored (no identity leaks from it).
- No token → `401`; malformed token → `401`; bad signature → `401`; expired token → `401`; `nbf` in the future → `401`.
- **Algorithm confusion:** HS256 token signed with the RS256 public key is rejected under RS256 config; `alg:none` rejected.
- `aud`/`iss` mismatch → `401`; correct `aud`/`iss` accepted.
- Valid org-A token → `ACLContext.tenant_id == "A"`; a `/query` returns **only** A's chunks through the **real** store filter (not a fake) — a poisoned B chunk engineered to outrank A returns zero rows.
- Missing/blank `tenant_id` claim → `403`.
- Allowlist **on**: an over-scoped claim tag is dropped (effective = claim ∩ grant); allowlist **off**: claim tags pass through.
- Oversized `question` → `422`. A token presenting more than `max_acl_tags` tags → `403` (over-scoped/malformed token).
- Dev signer: minting works with `jwt_secret` + flag on; **unavailable** when no secret / flag off / RS256-only.
- Boot check: `auth_enabled=False` with `app_env=prod` refuses to start.
- **Invert the old test** that certified the spoofable header into one asserting identity comes only from a verified principal.

Coverage note: the real-filter isolation test must exercise the actual `qdrant_filter`/`pg_where` path (ephemeral store), not a fake — this is coordinated with SP5 (CI runs the real ACL tests).

---

## 12. Files

**Add:** `providers/auth/__init__.py`, `providers/auth/jwt_verifier.py`, `providers/auth/allowlist.py`, `providers/auth/dev_signer.py`, `app/auth.py`, `scripts/mint_token.py`, `tests/test_auth.py`.
**Modify:** `core/types.py` (Principal, ACLContext validator), `core/interfaces.py` (AuthVerifier), `core/config.py` (knobs + boot validation), `core/registry.py` (builders), `app/api.py` (require_principal, QueryRequest, error shapes), `app/demo.py` (mint + send token), `pyproject.toml` (pyjwt[crypto]), and invert the spoof test in `tests/test_app.py`.

---

## 13. Open questions / future hooks

- **Revocation** → per-tenant `token_version` checked server-side; lands with the org registry (SP1.5).
- **Org self-service + per-org ingestion** → SP1.5, if elected. The `StaticAllowlist` is the seed of the registry.
- **Physical per-org isolation** (namespace-per-tenant) → VDB-Decision before SP8 ingest.
- **API-key auth** for programmatic clients → second `AuthVerifier` impl, no pipeline change.
