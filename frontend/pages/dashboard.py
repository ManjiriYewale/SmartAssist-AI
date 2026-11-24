
from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.database import repository as repo
from frontend.components import ui


# Icons for recent activity
ICON = {
    "document": "📄",
    "query": "💬",
    "tool": "🛠️"
}


def render() -> None:

    # ---------- PAGE HEADER ----------

    ui.page_header(
        "🚀 AI Command Center",
        "Welcome to SmartAssist AI - Your intelligent workspace"
    )

    # Get dashboard data from database
    stats = ui.call_backend(repo.get_dashboard_stats)

    if stats is None:
        return

    # ---------- STATISTICS CARDS ----------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ui.metric_card(
            "📁 Documents",
            stats["total_documents"],
            f"{stats['failed_documents']} failed"
        )

    with col2:
        ui.metric_card(
            "✅ Processed",
            stats["processed_documents"],
            f"{stats['total_chunks']} chunks indexed"
        )

    with col3:
        ui.metric_card(
            "💬 Queries",
            stats["total_queries"],
            f"{stats['successful_queries']} successful"
        )

    with col4:
        ui.metric_card(
            "⚡ Tool Runs",
            stats["total_tool_executions"],
            f"Avg latency {stats['avg_latency_ms']} ms"
        )

    st.divider()

    
    # ---------- QUICK ACTIONS ----------

    st.subheader("⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🤖 Ask AI Assistant", use_container_width=True):
            st.switch_page(st.session_state["app_pages"][1])

    with col2:
        if st.button("📁 Document Library", use_container_width=True):
            st.switch_page(st.session_state["app_pages"][2])

    with col3:
        if st.button("🔍 RAG Explorer", use_container_width=True):
            st.switch_page(st.session_state["app_pages"][3])


    st.divider()

    # ---------- TOOL USAGE CHART ----------

    st.subheader("📊 Tool Usage Analytics")

    tool_stats = ui.call_backend(repo.get_tool_stats)

    if tool_stats:

        df = pd.DataFrame(tool_stats)

        fig = px.bar(
            df,
            x="tool",
            y="calls",
            color="tool",
            title="AI Tool Executions",
            labels={
                "tool": "AI Tools",
                "calls": "Number of Executions"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info("No tool executions yet.")

    st.divider()

    # ---------- BOTTOM SECTION ----------

    left, right = st.columns([3, 2])

    # Recent activity
    with left:

        st.subheader("🕒 Recent Activity")

        events = ui.call_backend(
            repo.get_recent_activity,
            8
        )

        if events:

            rows = ""

            for event in events:

                icon = ICON.get(
                    event["type"],
                    "•"
                )

                text = html.escape(str(event["text"]))

                time = ui.fmt_time(event["time"])

                rows += (
                    f"<div>{icon} {text} "
                    f"<small>· {time} UTC</small></div>"
                )

            st.markdown(
                f'<div class="sa-timeline">{rows}</div>',
                unsafe_allow_html=True
            )

        else:
            st.info("No recent activity.")

    # Document status
    with right:

        st.subheader("📂 Document Status")

        counts = ui.call_backend(
            repo.get_document_status_counts
        )

        if counts:

            for status, count in counts.items():

                st.markdown(
                    ui.badge(
                        f"{status}: {count}",
                        status
                    ),
                    unsafe_allow_html=True
                )

        else:
            st.info("No documents uploaded yet.")

        st.write("")

        st.subheader("🟢 System Overview")

        st.success("SmartAssist AI Dashboard")
        st.caption(
            "Monitor documents, queries and AI tool activity."
        )
