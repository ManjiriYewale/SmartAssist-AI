
from __future__ import annotations

import streamlit as st

from backend import services
from backend.database import repository as repo
from frontend.components import ui


# Suggested questions
EXAMPLES = [
    "📄 Summarize my uploaded document",
    "📁 Which documents have I uploaded?",
    "🧮 Calculate 18% GST on 2500",
    "📊 Find the highest revenue in my dataset",
    "🔍 Compare my two uploaded reports",
]


# Display AI response details
def show_details(message, index):

    sources = message.get("sources") or []
    tools = message.get("tool_calls") or []
    intent = message.get("intent")

    if intent:
        st.caption(f"🎯 Intent: {intent}")

    col1, col2, col3 = st.columns(3)

    # Document sources
    with col1:
        if sources and ui.get_pref("show_sources", True):
            with st.popover(f"📚 Sources ({len(sources)})"):
                for source in sources:
                    st.write(source.get("filename", "Document"))
                    st.caption(source.get("snippet", ""))

    # Tools used by AI
    with col2:
        if tools and ui.get_pref("show_tools", True):
            with st.popover(f"🛠️ Tools ({len(tools)})"):
                for tool in tools:
                    st.write(tool.get("name", "Tool"))
                    st.json(tool.get("args", {}))

    # Feedback
    with col3:
        st.feedback(
            "thumbs",
            key=f"feedback_{st.session_state.get('conv_id')}_{index}"
        )


def render():

    # ---------- PAGE HEADER ----------

    ui.page_header(
        "🤖 AI Workspace",
        "Your intelligent assistant for documents and data"
    )

    # ---------- CONVERSATIONS ----------

    conversations = ui.call_backend(repo.list_conversations)

    if conversations is None:
        return

    st.session_state.setdefault("conv_id", None)

    conversation_ids = [c["id"] for c in conversations]

    if st.session_state.conv_id not in conversation_ids:
        st.session_state.conv_id = None

    options = [None] + conversation_ids

    labels = {None: "➕ Start New Conversation"}

    for conversation in conversations:
        labels[conversation["id"]] = conversation["title"]

    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])

    with col1:
        conv_id = st.selectbox(
            "Chat History",
            options,
            index=options.index(st.session_state.conv_id),
            format_func=lambda x: labels[x],
            label_visibility="collapsed"
        )

        st.session_state.conv_id = conv_id

    with col2:
        if st.button("✨ New", use_container_width=True):
            st.session_state.conv_id = None
            st.rerun()

    with col3:
        if st.button(
            "🧹 Clear",
            disabled=conv_id is None,
            use_container_width=True
        ):
            repo.clear_messages(conv_id)
            st.rerun()

    with col4:
        if st.button(
            "🗑️ Delete",
            disabled=conv_id is None,
            use_container_width=True
        ):
            repo.delete_conversation(conv_id)
            st.session_state.conv_id = None
            st.rerun()

    st.divider()

    # ---------- CHAT HISTORY ----------

    messages = []

    if conv_id:
        messages = ui.call_backend(repo.get_messages, conv_id)

        if messages is None:
            return

    # Welcome screen
    if not messages:

        st.markdown("### 👋 Hello! How can I help you today?")

        st.caption(
            "Ask questions about documents, "
            "analyze datasets or use AI tools."
        )

        st.write("")

        st.markdown("#### ✨ Try asking")

        col1, col2 = st.columns(2)

        selected_prompt = None

        for index, question in enumerate(EXAMPLES):

            with col1 if index % 2 == 0 else col2:

                if st.button(
                    question,
                    key=f"example_{index}",
                    use_container_width=True
                ):
                    selected_prompt = question

    else:
        selected_prompt = None

    # Display previous messages
    for index, message in enumerate(messages):

        with st.chat_message(message["role"]):

            content = message["content"].replace(
                "<br>", "\n\n"
            )

            st.markdown(content)

            if message["role"] == "assistant":
                show_details(message, index)

    # ---------- CHAT INPUT ----------

    user_prompt = st.chat_input(
        "💬 Ask SmartAssist AI anything..."
    )

    prompt = user_prompt or selected_prompt

    # ---------- AI RESPONSE ----------

    if prompt:

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.status(
                "🤖 SmartAssist AI is thinking...",
                expanded=True
            ) as status:

                st.write("🔍 Understanding your question...")

                st.write("📚 Searching documents and tools...")

                result = ui.call_backend(
                    services.ask,
                    prompt,
                    conv_id
                )

                if result:
                    status.update(
                        label="✅ Response ready!",
                        state="complete",
                        expanded=False
                    )
                else:
                    status.update(
                        label="❌ Request failed",
                        state="error",
                        expanded=False
                    )

        if result:

            st.session_state.conv_id = result["conversation_id"]

            for warning in result.get("warnings", []):
                st.toast(warning, icon="⚠️")

            st.rerun()
