
"""Reusable Streamlit UI helpers (styling, cards, badges, safe backend calls)."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Callable

import streamlit as st

from backend.exceptions import SmartAssistError


# ============================================================
# SMARTASSIST AI - MODERN UI THEME
# ============================================================

CSS = """
<style>

/* ---------- GLOBAL THEME ---------- */

:root {
    --sa-accent: #8B5CF6;
    --sa-accent2: #22D3EE;
    --sa-border: rgba(139, 92, 246, 0.20);
    --sa-card: #171C31;
    --sa-bg: #0B1020;
    --sa-text: #F8FAFC;
    --sa-muted: #94A3B8;
}

/* Main application */

.stApp {
    background:
        radial-gradient(
            ellipse at 90% 0%,
            rgba(139, 92, 246, 0.10),
            transparent 45%
        ),
        #0B1020;
    color: var(--sa-text);
}

/* Main content container */

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ---------- SIDEBAR ---------- */

[data-testid="stSidebar"] {
    background: #151A2E;
    border-right: 1px solid var(--sa-border);
}


[data-testid="stSidebar"] button {
    border-radius: 10px;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] button:hover {
    background: rgba(139, 92, 246, 0.12);
}


/* ---------- HEADINGS ---------- */

h1, h2, h3 {
    letter-spacing: -0.025em;
}

.sa-header {
    margin-bottom: 1.8rem;
}

.sa-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;

    background: linear-gradient(
        90deg,
        #FFFFFF,
        #A78BFA
    );

    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sa-header p {
    color: var(--sa-muted);
    margin-top: 0.5rem;
    font-size: 1rem;
}

/* ---------- DASHBOARD CARDS ---------- */

.sa-card {
    background: linear-gradient(
        145deg,
        #1B2138,
        #13182B
    );

    border: 1px solid var(--sa-border);
    border-radius: 18px;
    padding: 1.5rem;
    height: 100%;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease,
        border-color 0.25s ease;
}

.sa-card:hover {
    transform: translateY(-5px);

    border-color: rgba(139, 92, 246, 0.50);

    box-shadow:
        0 12px 30px rgba(139, 92, 246, 0.18);
}

/* Card label */

.sa-card .label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--sa-muted);
    font-weight: 600;
}

/* Card value */

.sa-card .value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.3;
    margin-top: 0.4rem;

    background: linear-gradient(
        90deg,
        #8B5CF6,
        #22D3EE
    );

    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Card description */

.sa-card .hint {
    font-size: 0.8rem;
    color: var(--sa-muted);
    margin-top: 0.25rem;
}

/* ---------- BUTTONS ---------- */

.stButton > button {
    border-radius: 12px;
    font-weight: 500;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 6px 20px rgba(139, 92, 246, 0.15);
}

/* Primary buttons */

.stButton > button[kind="primary"] {
    background: linear-gradient(
        90deg,
        #7C3AED,
        #4F46E5
    );

    color: white;
    border: none;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(
        90deg,
        #8B5CF6,
        #6366F1
    );
}

/* ---------- CHAT INTERFACE ---------- */

[data-testid="stChatMessage"] {
    background: rgba(23, 28, 49, 0.85);
    border: 1px solid var(--sa-border);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}

/* Chat input */

[data-testid="stChatInput"] {
    border-radius: 16px;
}

/* ---------- INPUT FIELDS ---------- */

.stTextInput input,
.stTextArea textarea {
    border-radius: 12px;
}

/* ---------- EXPANDERS ---------- */

[data-testid="stExpander"] {
    border: 1px solid var(--sa-border);
    border-radius: 14px;
    overflow: hidden;
}

/* ---------- STATUS BADGES ---------- */

.sa-badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Success */

.sa-ok {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
}

/* Warning */

.sa-warn {
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
}

/* Error */

.sa-err {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
}

/* Information */

.sa-info {
    background: rgba(139, 92, 246, 0.15);
    color: #A78BFA;
}

/* ---------- BRAND ---------- */

.sa-brand {
    font-size: 1.3rem;
    font-weight: 800;
}

.sa-brand span {
    background: linear-gradient(
        90deg,
        #8B5CF6,
        #22D3EE
    );

    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---------- ACTIVITY TIMELINE ---------- */

.sa-timeline {
    border-left: 2px solid var(--sa-border);
    padding-left: 1rem;
    margin-left: 0.3rem;
}

.sa-timeline div {
    margin-bottom: 0.65rem;
    font-size: 0.9rem;
}

.sa-timeline small {
    color: var(--sa-muted);
}

/* ---------- DATA TABLES ---------- */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- TABS ---------- */

.stTabs [data-baseweb="tab"] {
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    color: #A78BFA;
}

/* ---------- ANIMATIONS ---------- */

@keyframes saFadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.sa-header,
.sa-card {
    animation: saFadeIn 0.4s ease-out both;
}

/* ---------- MOBILE RESPONSIVE ---------- */

@media (max-width: 760px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .sa-header h1 {
        font-size: 1.6rem;
    }

    .sa-card {
        padding: 1rem;
    }

    .sa-card .value {
        font-size: 1.5rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    .sa-header,
    .sa-card {
        animation: none;
    }

    .sa-card,
    .stButton > button {
        transition: none;
    }
}



</style>
"""


# ============================================================
# STATUS CLASSES
# ============================================================

STATUS_CLASS = {
    "processed": "sa-ok",
    "completed": "sa-ok",
    "success": "sa-ok",
    "processing": "sa-info",
    "running": "sa-info",
    "uploaded": "sa-warn",
    "step_limit": "sa-warn",
    "failed": "sa-err",
    "error": "sa-err",
}


# ============================================================
# INJECT GLOBAL CSS
# ============================================================

def inject_css() -> None:
    """Apply SmartAssist AI theme."""
    st.markdown(
        CSS,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE HEADER
# ============================================================

def page_header(
    title: str,
    subtitle: str = ""
) -> None:

    st.markdown(
        f"""
        <div class="sa-header">
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# METRIC CARD
# ============================================================


def metric_card(label: str, value: Any, hint: str = "") -> None:
    card_html = (
        '<div class="sa-card">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div>'
        f'<div class="hint">{html.escape(hint)}</div>'
        '</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)


# ============================================================
# STATUS BADGE
# ============================================================

def badge(
    text: str,
    kind: str | None = None
) -> str:

    cls = STATUS_CLASS.get(
        kind or text,
        "sa-info"
    )

    return (
        f'<span class="sa-badge {cls}">'
        f'{html.escape(str(text))}'
        f'</span>'
    )


# ============================================================
# HUMAN READABLE FILE SIZE
# ============================================================

def human_size(n: int | None) -> str:

    n = n or 0

    for unit in ("B", "KB", "MB", "GB"):

        if n < 1024 or unit == "GB":

            return (
                f"{n:.0f} {unit}"
                if unit == "B"
                else f"{n:.1f} {unit}"
            )

        n /= 1024

    return f"{n} B"


# ============================================================
# FORMAT DATE AND TIME
# ============================================================

def fmt_time(iso: str | None) -> str:

    if not iso:
        return "—"

    try:
        return datetime.fromisoformat(
            iso
        ).strftime("%Y-%m-%d %H:%M")

    except ValueError:
        return iso


# ============================================================
# SAFE BACKEND CALL
# ============================================================

def call_backend(
    fn: Callable,
    *args,
    **kwargs
):

    """Run backend calls safely."""

    try:
        return fn(*args, **kwargs)

    except SmartAssistError as exc:
        st.error(exc.user_message)

    except Exception:

        import logging

        logging.getLogger(
            "frontend"
        ).exception(
            "Unexpected UI error"
        )

        st.error(
            "An unexpected error occurred. "
            "Details were written to "
            "logs/smartassist.log."
        )

    return None


# ============================================================
# USER PREFERENCES
# ============================================================

def get_pref(
    name: str,
    default: Any
) -> Any:

    return st.session_state.get(
        f"pref_{name}",
        default
    )


