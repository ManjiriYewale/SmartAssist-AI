
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from backend import services
from backend.database import repository as repo
from backend.tools import tool_catalog
from frontend.components import ui


# ---------- TOOL INFORMATION ----------

TOOL_INFO = {
    "calculator": ("🧮", "Calculator"),
    "document_search": ("🔍", "Document Search"),
    "csv_excel_analyzer": ("📊", "Data Analyzer"),
    "document_summarizer": ("📝", "Document Summarizer"),
    "document_comparator": ("📑", "Document Comparator"),
    "document_manager": ("📁", "Document Manager"),
}

EXAMPLE_ARGS = {
    "calculator": {
        "operation": "add_gst",
        "amount": 2500,
        "percent": 18
    },
    "document_search": {
        "query": "annual leave policy",
        "top_k": 3
    },
    "csv_excel_analyzer": {
        "operation": "overview"
    },
    "document_summarizer": {
        "mode": "key_points"
    },
    "document_comparator": {
        "document_a": "file_a.docx",
        "document_b": "file_b.docx"
    },
    "document_manager": {
        "action": "list"
    },
}


# ---------- AVAILABLE TOOLS ----------

def available_tools(catalog):

    st.subheader("🛠️ Available AI Tools")

    stats = ui.call_backend(repo.get_tool_stats)

    if stats is None:
        return

    usage = {item["tool"]: item for item in stats}

    for tool in catalog:

        name = tool["name"]

        icon, title = TOOL_INFO.get(
            name, ("🛠️", name.replace("_", " ").title())
        )

        tool_stats = usage.get(name, {})

        with st.container(border=True):

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"### {icon} {title}")
                st.write(tool["description"])

            with col2:
                st.metric(
                    "Executions",
                    tool_stats.get("calls", 0)
                )

                st.caption(
                    f"Errors: {tool_stats.get('errors', 0)}"
                )

            with st.expander("⚙️ View Tool Parameters"):

                rows = []

                for parameter, details in tool["parameters"].items():

                    param_type = details.get("type")

                    if not param_type:
                        param_type = "/".join(
                            str(item.get("type", "?"))
                            for item in details.get("anyOf", [])
                        )

                    rows.append({
                        "Parameter": parameter,
                        "Type": param_type,
                        "Required": parameter in tool["required"],
                        "Description": details.get("description", "")
                    })

                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        hide_index=True,
                        width="stretch"
                    )


# ---------- RUN TOOL ----------

def run_tool(catalog):

    st.subheader("▶️ Execute AI Tool")

    st.caption(
        "Select a tool, enter its arguments and run it manually."
    )

    names = [tool["name"] for tool in catalog]

    selected = st.selectbox(
        "Choose Tool",
        names
    )

    icon, title = TOOL_INFO.get(
        selected, ("🛠️", selected)
    )

    st.info(f"{icon} Selected Tool: {title}")

    arguments = st.text_area(
        "Tool Arguments (JSON)",
        value=json.dumps(
            EXAMPLE_ARGS.get(selected, {}),
            indent=2
        ),
        height=180,
        key=f"arguments_{selected}"
    )

    if st.button(
        "🚀 Run Selected Tool",
        type="primary",
        use_container_width=True
    ):

        try:
            args = json.loads(arguments)

            if not isinstance(args, dict):
                st.error("Arguments must be a JSON object.")
                return

        except json.JSONDecodeError:
            st.error("Please enter valid JSON arguments.")
            return

        with st.spinner(f"Running {selected}..."):

            result = ui.call_backend(
                services.run_tool_manually,
                selected,
                args
            )

        if result is None:
            return

        st.divider()

        st.subheader("📋 Execution Result")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                ui.badge(result["status"]),
                unsafe_allow_html=True
            )

        with col2:
            st.metric(
                "Execution Time",
                f"{result['duration_ms']} ms"
            )

        st.markdown("**Tool Output**")

        st.json(result["output"])


# ---------- USAGE HISTORY ----------

def usage_history(catalog):

    st.subheader("📊 Tool Execution History")

    names = ["all"] + [tool["name"] for tool in catalog]

    selected = st.selectbox(
        "Filter by Tool",
        names
    )

    rows = ui.call_backend(
        repo.list_tool_executions,
        200,
        None if selected == "all" else selected
    )

    if rows is None:
        return

    if not rows:
        st.info("No tool executions recorded yet.")
        return

    # Execution statistics
    total = len(rows)

    successful = sum(
        1 for item in rows
        if item["status"] == "success"
    )

    failed = total - successful

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Executions", total)
    col2.metric("Successful", successful)
    col3.metric("Other / Failed", failed)

    # History table
    table = pd.DataFrame([
        {
            "Time (UTC)": ui.fmt_time(item["created_at"]),
            "Tool": item["tool_name"],
            "Status": item["status"],
            "Duration (ms)": item["duration_ms"],
            "Source": item["source"]
        }
        for item in rows
    ])

    st.dataframe(
        table,
        hide_index=True,
        width="stretch"
    )

    st.divider()

    st.subheader("🔎 Execution Details")

    for item in rows[:15]:

        with st.expander(
            f"🛠️ {item['tool_name']} | {item['status']}"
        ):

            st.markdown("**Input Arguments**")
            st.json(item["input_args"], expanded=False)

            st.markdown("**Output**")
            st.code(
                str(item["output"] or "")[:3000],
                language="text"
            )

            if item.get("error_message"):
                st.error(item["error_message"])


# ---------- MAIN PAGE ----------

def render():

    ui.page_header(
        "🛠️ Tool Center",
        "Explore, execute and monitor your AI tools"
    )

    catalog = tool_catalog()

    tab1, tab2, tab3 = st.tabs([
        "🧰 Available Tools",
        "▶️ Run Tool",
        "📊 Usage History"
    ])

    with tab1:
        available_tools(catalog)

    with tab2:
        run_tool(catalog)

    with tab3:
        usage_history(catalog)
