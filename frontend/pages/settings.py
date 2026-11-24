
from __future__ import annotations

import streamlit as st

from backend.config.settings import (
    PROJECT_ROOT,
    get_settings,
    reload_settings
)
from backend.database.db import check_connection, configure_engine
from backend.llm.model import check_llm_connection
from backend.rag import embeddings, vector_store
from frontend.components import ui


# ---------- AI MODEL SETTINGS ----------

def model_settings(settings):

    st.subheader("🤖 AI Model Configuration")

    with st.container(border=True):

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**AI Provider**")
            st.info("Groq")

            st.markdown("**Model Name**")
            st.code(settings.groq_model)

            st.markdown("**Temperature**")
            st.write(settings.llm_temperature)

        with col2:
            st.markdown("**API Key Status**")

            if settings.llm_configured:
                st.success("✅ API Key Configured")
            else:
                st.error("❌ API Key Missing")

            st.markdown("**Timeout**")
            st.write(f"{settings.llm_timeout_seconds} seconds")

            st.markdown("**Maximum Retries**")
            st.write(settings.llm_max_retries)

        if st.button("🔌 Test AI Connection"):

            with st.spinner("Connecting to Groq..."):
                ok, message = check_llm_connection()

            if ok:
                st.success(message)
            else:
                st.error(message)

    st.caption(
        "To change the AI model, update GROQ_MODEL in your .env file."
    )


# ---------- DATABASE SETTINGS ----------

def database_settings(settings):

    st.subheader("🗄️ Database Configuration")

    with st.container(border=True):

        st.markdown("**Database Backend**")
        st.write(settings.database_backend)

        st.markdown("**Database Connection URL**")
        st.code(settings.masked_database_url())

        if st.button("🔌 Test Database Connection"):

            with st.spinner("Checking database..."):
                ok, message = check_connection()

            if ok:
                st.success(message)
            else:
                st.error(message)


# ---------- RAG SETTINGS ----------

def rag_settings(settings):

    st.subheader("📚 RAG & Document Processing")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Embedding Model",
            settings.embedding_model.split("/")[-1]
        )

        st.metric(
            "Chunk Size",
            settings.chunk_size
        )

        st.metric(
            "Chunk Overlap",
            settings.chunk_overlap
        )

    with col2:
        st.metric(
            "Top-K Results",
            settings.top_k
        )

        st.metric(
            "Minimum Relevance",
            settings.min_relevance_score
        )

        st.metric(
            "Maximum Upload",
            f"{settings.max_upload_mb} MB"
        )

    with st.expander("⚙️ Advanced RAG Configuration"):

        st.markdown("**Vector Database**")
        st.code(str(settings.chroma_path))

        st.markdown("**Chroma Collection**")
        st.code(settings.chroma_collection)

        st.markdown("**Upload Directory**")
        st.code(str(settings.upload_path))

        st.markdown("**Maximum Agent Steps**")
        st.write(settings.max_agent_steps)

        st.markdown("**Tool Calls Per Step**")
        st.write(settings.max_tool_calls_per_step)

    st.info(
        "Changing the embedding model or chunk size affects "
        "newly processed documents. Reprocess existing documents "
        "to update their embeddings."
    )


# ---------- USER PREFERENCES ----------

def user_preferences():

    st.subheader("🎨 User Preferences")

    with st.container(border=True):

        st.toggle(
            "🛠️ Show Tool Execution Details",
            key="pref_show_tools",
            value=ui.get_pref("show_tools", True)
        )

        st.toggle(
            "📚 Show Document Sources",
            key="pref_show_sources",
            value=ui.get_pref("show_sources", True)
        )

        st.caption(
            "These preferences control what information "
            "is displayed in AI Assistant conversations."
        )


# ---------- SYSTEM HEALTH ----------

def system_health(settings):

    st.subheader("💚 System Health")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🗄️ Check Database",
            use_container_width=True
        ):

            with st.spinner("Checking database..."):
                ok, message = check_connection()

            if ok:
                st.success(message)
            else:
                st.error(message)

    with col2:

        if st.button(
            "🤖 Check AI Model",
            use_container_width=True
        ):

            with st.spinner("Checking AI model..."):
                ok, message = check_llm_connection()

            if ok:
                st.success(message)
            else:
                st.error(message)

    st.caption(
        "Connection tests run only when you click a button."
    )


# ---------- RELOAD CONFIGURATION ----------

def reload_configuration():

    st.subheader("🔄 Configuration Management")

    st.caption(
        "Settings are loaded from your project's .env file."
    )

    if st.button(
        "🔄 Reload Settings",
        type="primary"
    ):

        with st.spinner("Reloading application settings..."):

            reload_settings()

            embeddings.clear_embedding_cache()

            vector_store.reset_vector_store()

            configure_engine()

            from backend.agents.graph import get_compiled_graph

            get_compiled_graph.cache_clear()

        st.success("Settings reloaded successfully!")

        st.rerun()

    st.caption(
        f"Environment file: {PROJECT_ROOT / '.env'}"
    )


# ---------- MAIN PAGE ----------

def render():

    ui.page_header(
        "⚙️ Settings & Configuration",
        "Manage AI models, RAG settings and system preferences"
    )

    settings = get_settings()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 AI & Database",
        "📚 RAG Settings",
        "🎨 Preferences",
        "💚 System Health"
    ])

    with tab1:

        model_settings(settings)

        st.divider()

        database_settings(settings)

    with tab2:

        rag_settings(settings)

    with tab3:

        user_preferences()

    with tab4:

        system_health(settings)

        st.divider()

        reload_configuration()
