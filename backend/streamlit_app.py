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

# set_page_config MUST be the very first Streamlit command that runs.
# Touching st.secrets below can itself render a "no secrets found" warning
# in some environments, which would count as an earlier Streamlit command
# and break this rule -- so this has to come before that block.
st.set_page_config(
    page_title="Pakistan Law Assistant",
    page_icon="⚖️",
    layout="centered",
)

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
from app.agent.graph import run_agent  # noqa: E402

CONTACT_EMAIL = "buildnexdigital@gmail.com"
LINKEDIN_URL = "https://www.linkedin.com/in/kashaf-junaid-1b84b331b/"

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
# Styling — "letterhead" identity: navy, gold, cream, serif headings.
# This is deliberately closer to a law firm's own site than a typical
# dark-mode AI-product look, since that's the visual language this
# specific audience (US law firms) already trusts.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0B1F3A; }

    /* Sidebar — explicit background + forced text color so it never
       silently falls back to a light default that clashes with cream text. */
    section[data-testid="stSidebar"] {
        background-color: #0A1628;
        border-right: 1px solid #23324A;
    }
    section[data-testid="stSidebar"] * {
        color: #E7E3D8 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #23324A;
    }
    section[data-testid="stSidebar"] a {
        border: 1px solid #D4AF37 !important;
        color: #F2C94C !important;
        background: transparent !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .sidebar-brand h3 {
        font-family: Georgia, 'Times New Roman', serif;
        color: #F2C94C !important;
        letter-spacing: 0.3px;
        margin-bottom: 4px;
    }
    .sidebar-brand p {
        color: #9FADC2 !important;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    /* Letterhead header — short accent rule instead of a stark full-width line */
    .letterhead { padding-bottom: 6px; margin-bottom: 12px; }
    .letterhead h1 {
        font-family: Georgia, 'Times New Roman', serif;
        color: #F5F1E6;
        margin-bottom: 6px;
    }
    .letterhead p {
        color: #93A0B4;
        font-size: 0.95rem;
        margin: 0 0 14px 0;
    }
    .letterhead .rule {
        width: 64px;
        height: 3px;
        background: linear-gradient(90deg, #D4AF37, #F2C94C);
        border-radius: 2px;
    }

    .exhibit-tag {
        display: inline-block;
        border: 1px solid #D4AF37;
        background: rgba(212, 175, 55, 0.12);
        color: #F2C94C;
        font-family: 'Courier New', monospace;
        font-size: 0.7rem;
        padding: 2px 9px;
        border-radius: 4px;
        margin-right: 8px;
    }
    .disclaimer-text {
        font-size: 0.78rem;
        font-style: italic;
        color: #7C8AA0;
        border-top: 1px solid #1E2C42;
        padding-top: 8px;
        margin-top: 12px;
    }

    /* Chat bubbles — actual cards instead of flat background text */
    [data-testid="stChatMessage"] {
        background-color: #101E33;
        border: 1px solid #1E2C42;
        border-radius: 12px;
        padding: 6px 8px;
    }
    [data-testid="stChatInput"] {
        background-color: #101E33;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header ("letterhead")
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="letterhead">
        <h1>⚖️ {settings.PRODUCT_NAME}</h1>
        <p>{settings.PRODUCT_TAGLINE}</p>
        <div class="rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — value proposition + contact CTA. This is what turns a demo
# into a lead: anyone who likes what they see has an immediate, obvious
# way to get in touch about getting this built for their own firm.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h3>BUILDNEX Digital</h3>
            <p>Custom AI-powered document assistants for law firms and
            professional service businesses.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("**What this demo shows**")
    st.markdown(
        "- Answers are grounded strictly in the documents provided — "
        "no fabricated case law or statutes\n"
        "- Every answer names its source document and section\n"
        "- Built on your firm's own filings, contracts, or research memos, "
        "not generic internet text"
    )

    st.markdown("---")

    st.markdown("**Mode**")
    mode = st.radio(
        "Mode",
        ["Grounded RAG", "Agent (RAG + tools)"],
        label_visibility="collapsed",
        help=(
            "Grounded RAG always searches the documents. Agent mode first "
            "decides whether a tool is needed at all — legal search, a "
            "calculator, or today's date — before answering."
        ),
    )

    st.markdown("---")

    st.markdown("**Want this built for your firm's own documents?**")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button(
            "Email",
            f"mailto:{CONTACT_EMAIL}?subject=Custom%20Legal%20AI%20Assistant%20Inquiry",
            use_container_width=True,
        )
    with col2:
        st.link_button(
            "LinkedIn",
            LINKEDIN_URL,
            use_container_width=True,
        )
    st.caption(CONTACT_EMAIL)

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
        if msg["role"] == "assistant" and msg.get("tools_used") is not None:
            if msg["tools_used"]:
                tool_badges = " ".join(
                    f'<span class="exhibit-tag">{t}</span>' for t in msg["tools_used"]
                )
                st.markdown(
                    f'<div style="margin-top:8px;">'
                    f'<span style="color:#7C8AA0;font-size:11px;'
                    f'text-transform:uppercase;margin-right:8px;">'
                    f'Tools used:</span>{tool_badges}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No tool was needed for this question.")
        elif msg["role"] == "assistant" and msg.get("sources"):
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
                if mode == "Agent (RAG + tools)":
                    agent_result = run_agent(question)
                    st.write(agent_result["answer"])

                    if agent_result["tools_used"]:
                        tool_badges = " ".join(
                            f'<span class="exhibit-tag">{t}</span>'
                            for t in agent_result["tools_used"]
                        )
                        st.markdown(
                            f'<div style="margin-top:8px;">'
                            f'<span style="color:#7C8AA0;font-size:11px;'
                            f'text-transform:uppercase;margin-right:8px;">'
                            f'Tools used:</span>{tool_badges}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No tool was needed for this question.")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": agent_result["answer"],
                            "tools_used": agent_result["tools_used"],
                        }
                    )
                else:
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