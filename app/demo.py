"""Streamlit demo for the Production RAG system.

Guard: all heavy pipeline construction is inside cached functions so that
``import app.demo`` completes instantly without a running backend or loaded
models.
"""

from __future__ import annotations

import time

# ---------------------------------------------------------------------------
# Streamlit is optional at import time — the module must import cleanly even
# when streamlit is not installed (e.g. during test collection).
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    _HAS_ST = True
except ImportError:  # pragma: no cover
    _HAS_ST = False


def _run_app():
    """Entry-point called only when running under Streamlit."""
    # Deferred imports so the module can be imported without services.
    from core.pipeline import build  # noqa: PLC0415
    from observability.cost import cost_usd  # noqa: PLC0415

    @st.cache_resource(show_spinner="Loading pipeline…")
    def _get_pipeline(version: str, corpus: str):
        return build(version=version, corpus=corpus or None)

    # ------------------------------------------------------------------
    # Sidebar controls
    # ------------------------------------------------------------------
    st.sidebar.title("RAG Demo Settings")

    tenant = st.sidebar.selectbox(
        "Tenant",
        ["public", "tenant_a", "tenant_b"],
        index=0,
    )
    version = st.sidebar.radio("Pipeline version", ["baseline", "full"], index=1)
    corpus = st.sidebar.text_input("Dataset (blank = default)", value="")

    # ------------------------------------------------------------------
    # Main panel
    # ------------------------------------------------------------------
    st.title("Production RAG — Query Interface")
    st.warning(
        "SECURITY NOTICE: retrieved text is untrusted user/document content "
        "and has not been verified for accuracy.",
        icon="⚠️",
    )

    question = st.text_area("Your question", height=100)

    if st.button("Submit") and question.strip():
        pipeline = _get_pipeline(version, corpus)

        from app.auth import demo_principal  # noqa: PLC0415

        try:
            acl = demo_principal(tenant).to_acl()
        except RuntimeError:
            st.error("Demo auth is disabled. Set AUTH_DEV_SIGNER_ENABLED=true and JWT_SECRET to run the demo.")
            st.stop()

        t0 = time.perf_counter()
        result = pipeline.run(question, acl)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Answer
        st.subheader("Answer")
        st.write(result["answer"])

        # Citations
        if result.get("citations"):
            st.subheader("Citations")
            for c in result["citations"]:
                st.markdown(f"- **{c.get('marker', '')}** chunk `{c.get('chunk_id', '')}` — *{c.get('quote', '')}*")

        # Retrieved chunks
        with st.expander(f"Retrieved chunks ({len(result.get('contexts', []))})"):
            for idx, (chunk_id, text) in enumerate(
                zip(result.get("retrieved_ids", []), result.get("contexts", []))
            ):
                st.markdown(f"**[{idx + 1}] `{chunk_id}`**")
                st.text(text[:500] + ("…" if len(text) > 500 else ""))

        # Metrics
        usage = result.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        completion_tok = usage.get("completion_tokens", 0)
        model = result.get("answer_obj", None)
        model_name = model.model if model is not None else ""
        cost = cost_usd(model_name, prompt_tok, completion_tok) if model_name else usage.get("cost_usd", 0.0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latency (ms)", f"{latency_ms:.0f}")
        col2.metric("Prompt tokens", prompt_tok)
        col3.metric("Completion tokens", completion_tok)
        col4.metric("Est. cost (USD)", f"${cost:.6f}")


# ---------------------------------------------------------------------------
# Run only under Streamlit, not during import / pytest collection.
# ---------------------------------------------------------------------------
if _HAS_ST and __name__ != "__test__":
    try:
        # `st.runtime.exists()` is True when running under the Streamlit server.
        if st.runtime.exists():  # type: ignore[attr-defined]
            _run_app()
    except Exception:
        # Older Streamlit versions don't have `st.runtime.exists()`.
        # Fall back: if the module is being executed as main, run the app.
        if __name__ == "__main__":  # pragma: no cover
            _run_app()
