"""
Streamlit UI for the Pakistan Law Assistant.

This is a self-contained alternative to the FastAPI + React setup, built for
one reason: Streamlit Community Cloud deploys straight from a GitHub repo,
is genuinely free, and doesn't require a credit card. It reuses the exact
same RAG pipeline (app.ingestion, app.retrieval, app.rag) as the FastAPI
version — nothing about the actual RAG logic is duplicated or different.

Run locally:
    streamlit run streamlit_app.py

Deploy: push to GitHub, then create a new app at share.streamlit.io pointing
at this file. See README.md for the full walkthrough.
"""

import os

import streamlit as st

# Bridge Streamlit Cloud's secrets manager into environment variables, so
# app/config.py (which reads everything via os.getenv) works identically
# whether it's running locally off .env or deployed off Streamlit secrets.
if hasattr(st, "secrets"):
    try:
        for key in st.secrets.keys():
            os.environ.setdefault(key, str(st.secrets[key]))
    except Exception:
        pass  # no secrets configured yet (e.g. first local run) — fine

from app.config import settings  # noqa: E402
from app.retrieval.vectorstore import collection_is_empty  # noqa: E402
from app.ingestion.build_index import build_index  # noqa: E402
from app.rag.chain import answer_question  # noqa: E402

st.set_page_config(
    page_title=settings.PRODUCT_NAME,
    page_icon="⚖️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# One-time setup: build the index automatically if it's empty. This is what
# lets the app "just work" on a fresh Streamlit Cloud deploy with no manual
# ingestion step — the first visitor triggers it, cached for everyone after.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_index_built():
    if collection_is_empty():
        build_index()
    return True


# ---------------------------------------------------------------------------
# Styling — same ink/emerald/brass identity as the React frontend.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #14171C; }
    .exhibit-tag {
        display: inline-block;
        border: 1px solid rgba(176, 141, 87, 0.4);
        background: rgba(176, 141, 87, 0.1);
        color: #C9A876;
        font-family: monospace;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 8px;
    }
    .disclaimer-text {
        font-size: 0.78rem;
        font-style: italic;
        color: rgba(233, 230, 222, 0.45);
        border-top: 1px solid #333A47;
        padding-top: 8px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(f"⚖️ {settings.PRODUCT_NAME}")
st.caption(settings.PRODUCT_TAGLINE)

if not settings.GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is not set. Add it to your .env file locally, or to "
        "your app's Secrets in the Streamlit Cloud dashboard."
    )
    st.stop()

with st.spinner("Preparing the document index…"):
    ensure_index_built()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Referenced excerpts ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"]):
                    label = f"Exhibit {LETTERS[i] if i < 26 else i + 1}"
                    page = (
                        f", p.{src['page'] + 1}" if src.get("page") is not None else ""
                    )
                    st.markdown(
                        f'<span class="exhibit-tag">{label}</span>'
                        f"**{src['source']}**{page}",
                        unsafe_allow_html=True,
                    )
                    st.text(src["excerpt"])
                    st.divider()
            if msg.get("disclaimer"):
                st.markdown(
                    f'<div class="disclaimer-text">{msg["disclaimer"]}</div>',
                    unsafe_allow_html=True,
                )

question = st.chat_input(
    "Ask about a section of the Penal Code, the Constitution, PECA…"
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = answer_question(question)
                st.write(result["answer"])

                if result["sources"]:
                    with st.expander(f"Referenced excerpts ({len(result['sources'])})"):
                        for i, src in enumerate(result["sources"]):
                            label = f"Exhibit {LETTERS[i] if i < 26 else i + 1}"
                            page = (
                                f", p.{src['page'] + 1}"
                                if src.get("page") is not None
                                else ""
                            )
                            st.markdown(
                                f'<span class="exhibit-tag">{label}</span>'
                                f"**{src['source']}**{page}",
                                unsafe_allow_html=True,
                            )
                            st.text(src["excerpt"])
                            st.divider()

                st.markdown(
                    f'<div class="disclaimer-text">{result["disclaimer"]}</div>',
                    unsafe_allow_html=True,
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "disclaimer": result["disclaimer"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                error_text = f"Something went wrong: {exc}"
                st.error(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text, "sources": []}
                )

st.caption("Educational demo · Not a substitute for professional legal advice")
