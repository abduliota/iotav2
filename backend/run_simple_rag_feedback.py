"""
Run test questions through simple RAG and produce a feedback report (not_found rate, per-question results).
Use for threshold/scope tuning. Loads questions from SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH if present.
Usage: from backend dir: python run_simple_rag_feedback.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Run from backend directory
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def load_test_questions(path: str | Path) -> list[dict]:
    """Load test questions from JSON: list of strings or list of {question, expected_not_found?}."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for item in data:
        if isinstance(item, str):
            out.append({"question": item, "expected_not_found": None})
        elif isinstance(item, dict) and "question" in item:
            out.append({
                "question": item["question"],
                "expected_not_found": item.get("expected_not_found"),
            })
        else:
            continue
    return out


def main() -> None:
    from config import SIMPLE_RAG_NOT_FOUND_MESSAGE, SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH
    from simple_rag import answer_query

    test_path = os.getenv("SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH", str(SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH))
    questions = load_test_questions(test_path)
    if not questions:
        print("No test questions found at", test_path, "- add a JSON list of questions or {question, expected_not_found}")
        sys.exit(0)

    results: list[dict] = []
    for i, row in enumerate(questions):
        q = row["question"]
        expected_nf = row.get("expected_not_found")
        print(f"[{i+1}/{len(questions)}] {q[:60]}...")
        try:
            out = answer_query(q)
            answer = out.get("answer") or ""
            is_not_found = (
                answer.strip().lower() == SIMPLE_RAG_NOT_FOUND_MESSAGE.lower()
                or "لم يتم العثور" in answer
            )
            results.append({
                "question": q,
                "answer_preview": answer[:200] + ("..." if len(answer) > 200 else ""),
                "not_found": is_not_found,
                "expected_not_found": expected_nf,
                "match": expected_nf is None or expected_nf == is_not_found,
            })
        except Exception as e:
            results.append({
                "question": q,
                "answer_preview": "",
                "not_found": True,
                "expected_not_found": expected_nf,
                "match": False,
                "error": str(e),
            })

    not_found_count = sum(1 for r in results if r["not_found"])
    match_count = sum(1 for r in results if r.get("match", True))
    report = {
        "total": len(results),
        "not_found_count": not_found_count,
        "not_found_rate": not_found_count / len(results) if results else 0,
        "expected_match_count": match_count,
        "results": results,
    }
    out_path = BACKEND_DIR / "output" / "simple_rag_feedback_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report written to {out_path}: not_found={not_found_count}/{len(results)}, expected_match={match_count}/{len(results)}")


if __name__ == "__main__":
    main()
