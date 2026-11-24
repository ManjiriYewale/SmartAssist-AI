"""SmartAssist AI – Streamlit entry point.   Run:  streamlit run frontend/app.py   (or: python run.py ui)"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="SmartAssist AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

from backend import services  # noqa: E402
from backend.config.settings import get_settings  # noqa: E402
from backend.database.db import check_connection  # noqa: E402
from frontend.components import ui  # noqa: E402
from frontend.pages import analytics, assistant, dashboard, documents, rag_explorer, settings, tools  # noqa: E402


@st.cache_resource(show_spinner="Starting SmartAssist AI…")
def boot() -> bool:
    services.init_app()
    return True


ui.inject_css()
try:
    boot()
except Exception as exc:  # noqa: BLE001
    import logging
    logging.getLogger("frontend").exception("Start-up failed")
    st.error("SmartAssist AI could not start: the database is not reachable.")
    st.info("1. Start PostgreSQL  2. Check DATABASE_URL / POSTGRES_* in `.env`  3. Run `python run.py setup`, then refresh this page.")
    st.stop()

pages = [
    st.Page(dashboard.render, title="Dashboard", icon=":material/space_dashboard:", url_path="dashboard", default=True),
    st.Page(assistant.render, title="AI Assistant", icon=":material/smart_toy:", url_path="assistant"),
    st.Page(documents.render, title="Document Library", icon=":material/folder_open:", url_path="documents"),
    st.Page(rag_explorer.render, title="RAG Explorer", icon=":material/travel_explore:", url_path="rag-explorer"),
    st.Page(tools.render, title="Tool Center", icon=":material/build:", url_path="tools"),
    st.Page(analytics.render, title="Analytics", icon=":material/monitoring:", url_path="analytics"),
    st.Page(settings.render, title="Settings", icon=":material/settings:", url_path="settings"),
]
st.session_state["app_pages"] = pages
nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.markdown('<div class="sa-brand">🧠 Smart<span>Assist</span> AI</div>', unsafe_allow_html=True)
    st.caption("Multi-tool RAG agent for your documents")
    s = get_settings()
    db_ok, _ = check_connection()
    st.markdown(
        ui.badge("Database ✓" if db_ok else "Database ✗", "success" if db_ok else "error") + " " +
        ui.badge("LLM ✓" if s.llm_configured else "LLM key missing", "success" if s.llm_configured else "warn"),
        unsafe_allow_html=True)
    if not s.llm_configured:
        st.caption("Set `GROQ_API_KEY` in `.env` to enable the assistant.")

nav.run()
