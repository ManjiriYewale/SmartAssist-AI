"""Evaluation harness.

Offline (no API key needed):   python evaluation/evaluate.py
    - retrieval relevance (hit@k, MRR, mean top score), unanswerable handling, tool accuracy
Live (needs GROQ_API_KEY):     python evaluation/evaluate.py --live
    - additionally: intent-routing accuracy, tool-selection accuracy, answer groundedness proxy, refusal quality

Uses an isolated temporary database / vector store, so it never touches your real data.
Embeddings follow your .env (EMBEDDING_BACKEND=huggingface for real numbers).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def setup_isolated_env(tmp: str) -> None:
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/eval.db"
    os.environ["CHROMA_DIR"] = f"{tmp}/chroma"
    os.environ["UPLOAD_DIR"] = f"{tmp}/uploads"
    from backend.config.settings import reload_settings
    from backend.database import db
    from backend.rag import vector_store

    reload_settings(); vector_store.reset_vector_store(); db.configure_engine(); db.init_db()


def load_docs(names: list[str]) -> None:
    from backend.rag.ingestion import process_upload
    for n in names:
        rec, _ = process_upload(n, (ROOT / "sample_data" / n).read_bytes())
        assert rec["status"] == "processed", f"{n}: {rec.get('error_message')}"


def eval_retrieval(cases: list[dict], k: int = 5) -> dict:
    from backend.rag.retriever import retrieve
    hits, rr, top_scores, rows = 0, 0.0, [], []
    for c in cases:
        res = retrieve(c["q"], top_k=k, min_score=0.0)
        rank = next((i for i, ch in enumerate(res.chunks, 1)
                     if ch.filename == c["doc"] and c["text"].lower() in ch.content.lower()), None)
        hits += rank is not None
        rr += 1 / rank if rank else 0
        top_scores.append(res.chunks[0].score if res.chunks else 0)
        rows.append((c["q"], rank))
    n = len(cases)
    return {"cases": n, f"hit@{k}": round(hits / n, 3), "mrr": round(rr / n, 3),
            "mean_top_score": round(sum(top_scores) / n, 3), "details": rows}


def eval_unanswerable(questions: list[str]) -> dict:
    from backend.rag.retriever import retrieve
    from backend.config.settings import get_settings
    thr = get_settings().min_relevance_score
    refused = sum(0 if retrieve(q, top_k=5).chunks else 1 for q in questions)
    return {"cases": len(questions), "correctly_found_nothing": refused, "threshold": thr}


def eval_tools(cases: list[dict]) -> dict:
    import pandas as pd
    from backend.services import run_tool_manually
    ok, rows = 0, []
    for c in cases:
        out = run_tool_manually(c["tool"], c["args"])
        good = out["status"] == "success"
        res = out["output"]
        if good and "expect" in c:
            good = all(abs(float(res.get(k, 1e18)) - float(v)) < 1e-6 for k, v in c["expect"].items())
        if good and "expect_pandas" in c:
            op, col = c["expect_pandas"].split(":")
            truth = getattr(pd.read_csv(ROOT / "sample_data" / c["args"]["document"])[col], op)()
            good = abs(float(res["result"]) - float(truth)) < 1e-6
        ok += good
        rows.append((c["tool"], c["args"].get("operation"), good))
    return {"cases": len(cases), "accuracy": round(ok / len(cases), 3), "details": rows}


def eval_agent_live(cases: list[dict]) -> dict:
    from backend.agents.graph import run_agent
    intent_ok = tools_ok = ground_ok = 0
    rows = []
    for c in cases:
        st = run_agent(c["q"])
        used = {t["name"] for t in st.get("tool_calls", [])}
        i_ok = st.get("intent") == c["intent"]
        t_ok = set(c["tools"]) <= used and (c["tools"] or not used)
        ans = st.get("final_response", "")
        g_ok = all(re.search(alt, ans, re.I) is not None for alt in c["answer_contains"])
        intent_ok += i_ok; tools_ok += t_ok; ground_ok += g_ok
        rows.append((c["q"], st.get("intent"), sorted(used), i_ok, t_ok, g_ok, st.get("status")))
    n = len(cases)
    return {"cases": n, "intent_routing_accuracy": round(intent_ok / n, 3), "tool_selection_accuracy": round(tools_ok / n, 3),
            "answer_contains_expected_facts": round(ground_ok / n, 3), "details": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also evaluate the LLM agent (needs GROQ_API_KEY)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    data = json.loads((Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        setup_isolated_env(tmp)
        from backend.config.settings import get_settings
        s = get_settings()
        print(f"Embedding backend: {s.embedding_backend} ({s.embedding_model})")
        load_docs(data["documents"])
        report = {"retrieval": eval_retrieval(data["retrieval"], args.k), "unanswerable": eval_unanswerable(data["unanswerable"]),
                  "tools": eval_tools(data["tools"])}
        if args.live:
            if not s.llm_configured:
                print("GROQ_API_KEY is not set – cannot run --live evaluation."); return 2
            report["agent"] = eval_agent_live(data["agent"])
        from backend.database import db
        db._engine.dispose()
    for section, r in report.items():
        print(f"\n== {section} ==")
        print(json.dumps({k: v for k, v in r.items() if k != "details"}, indent=2))
        for d in r.get("details", []):
            print("  ", d)
    (Path(__file__).parent / "last_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
