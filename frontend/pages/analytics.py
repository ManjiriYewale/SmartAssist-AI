
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.database import repository as repo
from frontend.components import ui


# ---------- PERFORMANCE METRICS ----------

def performance_metrics(stats):

    total = stats["total_queries"]
    successful = stats["successful_queries"]

    success_rate = (
        f"{successful / total * 100:.1f}%"
        if total else "0%"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ui.metric_card("💬 Total Queries", total)

    with col2:
        ui.metric_card("✅ Success Rate", success_rate)

    with col3:
        ui.metric_card(
            "⏱️ Avg Response Time",
            f"{stats['avg_latency_ms']} ms"
        )

    with col4:
        ui.metric_card(
            "🗨️ Conversations",
            stats["total_conversations"],
            f"{stats['total_messages']} messages"
        )


# ---------- QUERY TRENDS ----------

def query_trends(days):

    st.subheader("📈 Query Activity")

    data = ui.call_backend(repo.get_queries_per_day, days)

    if data is None:
        return

    if not data:
        st.info("No query activity available.")
        return

    df = pd.DataFrame(data)

    if df.empty or "date" not in df.columns:
        st.info("No query activity available.")
        return

    # Use the same query counts returned by the backend
    fig = px.line(
        df,
        x="date",
        y=[col for col in df.columns if col != "date"],
        markers=True,
        title=f"Query Trends - Last {days} Days"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Queries",
        legend_title_text="Metric"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------- INTENT DISTRIBUTION ----------

def intent_distribution():

    st.subheader("🎯 Query Intent Distribution")

    intents = ui.call_backend(repo.get_intent_distribution)

    if not intents:
        st.info("No query intents recorded yet.")
        return

    df = pd.DataFrame(
        list(intents.items()),
        columns=["Intent", "Queries"]
    )

    fig = px.pie(
        df,
        names="Intent",
        values="Queries",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------- TOOL USAGE ----------

def tool_usage():

    st.subheader("🛠️ AI Tool Usage")

    tools = ui.call_backend(repo.get_tool_stats)

    if not tools:
        st.info("No tool executions recorded.")
        return

    df = pd.DataFrame(tools)

    fig = px.bar(
        df,
        x="tool",
        y=["calls", "errors"],
        barmode="group",
        labels={
            "tool": "AI Tool",
            "value": "Executions",
            "variable": "Metric"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Detailed Tool Statistics"):

        st.dataframe(
            df.rename(columns={"avg_ms": "Average Time (ms)"}),
            hide_index=True,
            width="stretch"
        )


# ---------- DOCUMENT ANALYTICS ----------

def document_analytics():

    st.subheader("📚 Document Analytics")

    status = ui.call_backend(repo.get_document_status_counts)
    file_types = ui.call_backend(repo.get_file_type_counts)

    if not status and not file_types:
        st.info("No document data available.")
        return

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Processing Status**")

        if status:

            df = pd.DataFrame(
                list(status.items()),
                columns=["Status", "Documents"]
            )

            fig = px.pie(
                df,
                names="Status",
                values="Documents",
                hole=0.4
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.markdown("**File Type Distribution**")

        if file_types:

            df = pd.DataFrame(
                list(file_types.items()),
                columns=["File Type", "Documents"]
            )

            fig = px.bar(
                df,
                x="File Type",
                y="Documents"
            )

            st.plotly_chart(fig, use_container_width=True)


# ---------- RECENT QUERIES ----------

def recent_queries():

    st.subheader("📝 Recent AI Queries")

    queries = ui.call_backend(repo.list_queries, 25)

    if not queries:
        st.info("No recent queries available.")
        return

    df = pd.DataFrame([
        {
            "Time (UTC)": ui.fmt_time(q["created_at"]),
            "Query": q["query"][:80],
            "Intent": q["intent"],
            "Action": q["selected_action"],
            "Tools": ", ".join(q["tools_used"] or []),
            "Sources": q["num_sources"],
            "Status": q["status"],
            "Latency (ms)": q["latency_ms"]
        }
        for q in queries
    ])

    st.dataframe(
        df,
        hide_index=True,
        width="stretch"
    )


# ---------- RECENT ACTIVITY ----------

def recent_activity():

    st.subheader("🕒 Recent Activity")

    activity = ui.call_backend(repo.get_recent_activity, 8)

    if not activity:
        st.info("No recent activity.")
        return

    for item in activity:

        with st.container(border=True):

            st.write(item["text"])

            st.caption(
                f"🕒 {ui.fmt_time(item['time'])} UTC"
            )


# ---------- MAIN PAGE ----------

def render():

    ui.page_header(
        "📊 Analytics Dashboard",
        "Monitor AI performance, document usage and system activity"
    )

    stats = ui.call_backend(repo.get_dashboard_stats)

    if stats is None:
        return

    # KPI cards
    performance_metrics(stats)

    st.divider()

    # Time range selection
    days = st.selectbox(
        "📅 Analytics Period",
        [7, 14, 30, 60, 90],
        index=1,
        format_func=lambda value: f"Last {value} Days"
    )

    st.divider()

    # Query and intent charts
    col1, col2 = st.columns(2)

    with col1:
        query_trends(days)

    with col2:
        intent_distribution()

    st.divider()

    # Tool usage
    tool_usage()

    st.divider()

    # Document analytics
    document_analytics()

    st.divider()

    # Query history
    recent_queries()

    st.divider()

    # Activity timeline
    recent_activity()
