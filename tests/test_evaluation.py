"""Runs the offline parts of the evaluation harness with the hash embedder (CI-friendly thresholds)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import evaluate  # noqa: E402

DATA = json.loads((ROOT / "evaluation" / "eval_set.json").read_text(encoding="utf-8"))


def test_offline_evaluation_metrics():
    evaluate.load_docs(DATA["documents"])
    r = evaluate.eval_retrieval(DATA["retrieval"], k=5)
    assert r["hit@5"] >= 0.8, r
    t = evaluate.eval_tools(DATA["tools"])
    assert t["accuracy"] == 1.0, t["details"]
