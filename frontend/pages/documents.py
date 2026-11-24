
from __future__ import annotations

import pandas as pd
import streamlit as st

from backend import services
from backend.config.settings import ALLOWED_EXTENSIONS, get_settings
from backend.database import repository as repo
from backend.exceptions import SmartAssistError
from frontend.components import ui


# ---------- UPLOAD DOCUMENTS ----------

def upload_section():

    settings = get_settings()

    st.subheader("📤 Upload Documents")

    st.caption(
        "Upload PDF, Word, Text, CSV or Excel files "
        "to your AI knowledge base."
    )

    st.session_state.setdefault("uploader_key", 0)

    files = st.file_uploader(
        f"Choose files (Maximum {settings.max_upload_mb} MB each)",
        type=sorted(ALLOWED_EXTENSIONS),
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if files:

        st.info(f"📁 {len(files)} file(s) selected")

        if st.button("🚀 Upload & Process", type="primary"):

            progress = st.progress(0)

            for index, file in enumerate(files):

                try:
                    result = services.upload_document(
                        file.name,
                        file.getvalue()
                    )

                    if result["duplicate"]:
                        st.warning(f"{file.name} already exists.")

                    elif result["status"] == "processed":
                        st.success(f"✅ {file.name} processed successfully!")

                    else:
                        st.error(
                            f"{file.name}: {result.get('error_message', 'Processing failed')}"
                        )

                except SmartAssistError as error:
                    st.error(error.user_message)

                except Exception:
                    st.error(f"Unable to process {file.name}. Check logs.")

                progress.progress((index + 1) / len(files))

            st.session_state.uploader_key += 1

            if st.button("🔄 Refresh Documents"):
                st.rerun()


# ---------- DOCUMENT STATISTICS ----------

def document_statistics():

    stats = ui.call_backend(repo.get_dashboard_stats)

    if stats is None:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        ui.metric_card(
            "📁 Total Documents",
            stats["total_documents"]
        )

    with col2:
        ui.metric_card(
            "✅ Processed",
            stats["processed_documents"]
        )

    with col3:
        ui.metric_card(
            "📚 Indexed Chunks",
            stats["total_chunks"]
        )


# ---------- DOCUMENT MANAGEMENT ----------

def manage_documents(docs):

    st.subheader("📂 Manage Documents")

    for document in docs:

        with st.expander(
            f"📄 {document['filename']}  |  {document['status']}"
        ):

            st.markdown(
                ui.badge(document["status"]),
                unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Size",
                ui.human_size(document["size_bytes"])
            )

            col2.metric(
                "Pages",
                document["page_count"] or 0
            )

            col3.metric(
                "Chunks",
                document["chunk_count"]
            )

            col4.metric(
                "Characters",
                document["char_count"]
            )

            st.caption(
                f"Uploaded: {ui.fmt_time(document['created_at'])} UTC"
            )

            if document.get("error_message"):
                st.error(document["error_message"])

            # Reprocess failed documents
            if document["status"] in ["failed", "uploaded"]:

                if st.button(
                    "🔄 Reprocess",
                    key=f"reprocess_{document['id']}"
                ):

                    with st.spinner("Processing document..."):

                        result = ui.call_backend(
                            services.reprocess_document,
                            document["id"]
                        )

                    if result is not None:
                        st.rerun()

            # Delete document with confirmation
            if st.button(
                "🗑️ Delete Document",
                key=f"delete_{document['id']}"
            ):
                st.session_state["delete_doc_id"] = document["id"]

            if st.session_state.get("delete_doc_id") == document["id"]:

                st.warning("Are you sure you want to delete this document?")

                yes, no = st.columns(2)

                with yes:
                    if st.button(
                        "Yes, Delete",
                        key=f"yes_{document['id']}",
                        type="primary"
                    ):

                        result = ui.call_backend(
                            services.delete_document,
                            document["id"]
                        )

                        if result is not None:
                            st.session_state.pop("delete_doc_id", None)
                            st.rerun()

                with no:
                    if st.button(
                        "Cancel",
                        key=f"cancel_{document['id']}"
                    ):
                        st.session_state.pop("delete_doc_id", None)
                        st.rerun()


# ---------- MAIN PAGE ----------

def render():

    ui.page_header(
        "📚 Document Library",
        "Manage your AI knowledge base in one place"
    )

    # Document statistics
    document_statistics()

    st.divider()

    # Upload section
    upload_section()

    st.divider()

    # Search and filters
    st.subheader("🔎 Explore Documents")

    col1, col2, col3 = st.columns([3, 1.5, 1.5])

    with col1:
        search = st.text_input(
            "Search",
            placeholder="Search documents by filename..."
        )

    with col2:
        status = st.selectbox(
            "Status",
            ["all", "processed", "processing", "uploaded", "failed"]
        )

    with col3:
        file_type = st.selectbox(
            "File Type",
            ["all"] + sorted(ALLOWED_EXTENSIONS)
        )

    # Load documents
    docs = ui.call_backend(
        repo.list_documents,
        search or None,
        None if status == "all" else status,
        None if file_type == "all" else file_type
    )

    if docs is None:
        return

    if not docs:
        st.info("No documents found.")
        return

    # Document table
    table = pd.DataFrame([
        {
            "File Name": doc["filename"],
            "Type": doc["file_type"].upper(),
            "Size": ui.human_size(doc["size_bytes"]),
            "Pages": doc["page_count"],
            "Chunks": doc["chunk_count"],
            "Status": doc["status"]
        }
        for doc in docs
    ])

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # Manage documents
    manage_documents(docs)
