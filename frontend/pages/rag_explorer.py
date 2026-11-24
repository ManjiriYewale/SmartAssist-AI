
from __future__ import annotations

import streamlit as st

from backend import services
from backend.config.settings import ALLOWED_EXTENSIONS, get_settings
from backend.database import repository as repo
from backend.rag.retriever import RetrievedChunk, format_context
from frontend.components import ui


def render():

    # ---------- PAGE HEADER ----------

    ui.page_header(
        "🔎 RAG Explorer",
        "Search your documents and explore how AI retrieves information"
    )

    settings = get_settings()

    # Load processed documents
    docs = ui.call_backend(
        repo.list_documents,
        status="processed"
    )

    if docs is None:
        return

    if not docs:
        st.info(
            "📁 No processed documents available. "
            "Upload documents in the Document Library."
        )
        return

    # ---------- SEARCH SECTION ----------

    st.subheader("🔍 Semantic Document Search")

    with st.form("rag_search"):

        query = st.text_input(
            "Ask a question",
            placeholder="e.g. How many annual leave days do employees get?"
        )

        col1, col2 = st.columns(2)

        with col1:
            document = st.selectbox(
                "Select Document",
                [None] + [d["id"] for d in docs],
                format_func=lambda doc_id:
                    "All Documents" if doc_id is None
                    else next(
                        d["filename"]
                        for d in docs
                        if d["id"] == doc_id
                    )
            )

        with col2:
            file_type = st.selectbox(
                "File Type",
                ["any"] + sorted(ALLOWED_EXTENSIONS)
            )

        col3, col4 = st.columns(2)

        with col3:
            top_k = st.slider(
                "Number of Results (Top-K)",
                1,
                20,
                settings.top_k
            )

        with col4:
            min_score = st.slider(
                "Minimum Relevance Score",
                0.0,
                1.0,
                float(settings.min_relevance_score),
                0.05
            )

        search = st.form_submit_button(
            "🔎 Search Documents",
            type="primary",
            use_container_width=True
        )

    # ---------- EMBEDDING INFORMATION ----------

    with st.expander("⚙️ RAG Configuration"):

        st.write("**Embedding Model:**", settings.embedding_model)

        st.write("**Chunk Size:**", settings.chunk_size)

        st.write("**Chunk Overlap:**", settings.chunk_overlap)

        st.write("**Similarity Metric:** Cosine Similarity")

    if not search:
        return

    if not query.strip():
        st.warning("Please enter a question first.")
        return

    # ---------- RETRIEVE DOCUMENT CHUNKS ----------

    with st.spinner("🔍 Searching document knowledge base..."):

        result = ui.call_backend(
            services.explore_retrieval,
            query,
            top_k=top_k,
            doc_id=document,
            file_type=None if file_type == "any" else file_type,
            min_score=min_score
        )

    if result is None:
        return

    chunks = result["chunks"]

    st.divider()

    # ---------- SEARCH STATISTICS ----------

    st.subheader("📊 Search Results Overview")

    col1, col2 = st.columns(2)

    with col1:
        ui.metric_card(
            "📚 Chunks Retrieved",
            len(chunks)
        )

    with col2:
        ui.metric_card(
            "🚫 Below Threshold",
            result["below_threshold"]
        )

    if not chunks:
        st.warning(
            "No relevant information found. "
            "Try reducing the minimum relevance score."
        )
        return

    # ---------- RETRIEVED CHUNKS ----------

    st.subheader("📄 Retrieved Document Chunks")

    for index, chunk in enumerate(chunks, 1):

        item = RetrievedChunk(**chunk)

        with st.expander(
            f"📄 Result {index} | {item.filename} | Score: {item.score:.2f}",
            expanded=index == 1
        ):

            st.markdown(f"**📁 Document:** {item.filename}")

            st.markdown(
                f"**📄 Page:** {item.page or 'N/A'}"
            )

            st.markdown(
                f"**🔢 Chunk:** {item.chunk_index}"
            )

            if item.sheet:
                st.markdown(f"**📊 Sheet:** {item.sheet}")

            st.markdown("**🎯 Relevance Score**")

            st.progress(
                min(max(item.score, 0.0), 1.0)
            )

            st.caption(f"Similarity: {item.score:.2%}")

            st.markdown("**📝 Retrieved Content**")

            st.text(item.content)

    # ---------- LLM CONTEXT ----------

    st.divider()

    with st.expander("🧠 View Context Sent to LLM"):

        st.caption(
            "This is the retrieved information "
            "that will be provided to the AI model."
        )

        st.code(
            format_context(chunks),
            language="text"
        )
