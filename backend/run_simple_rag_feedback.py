"""
Run test questions through simple RAG and produce a feedback report (not_found rate, per-question results).
Use for threshold/scope tuning. Loads questions from SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH if present.
Creates a run-specific log file each time with all log lines and a complete stats section (accuracy, etc.).
Usage: from backend dir: python run_simple_rag_feedback.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Run from backend directory
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _attach_feedback_run_log_handler(log_dir: Path) -> tuple[logging.FileHandler, Path]:
    """Create a timestamped log file for this feedback run; attach handler to root logger. Returns (handler, path)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"feedback_test_{ts}.log"
    path = path.resolve()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(fh)
    return fh, path


def _detach_feedback_run_log_handler(handler: logging.Handler) -> None:
    """Remove handler from root logger and close it."""
    logging.getLogger().removeHandler(handler)
    try:
        handler.close()
    except Exception:
        pass


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
    from config import (
        SIMPLE_RAG_NOT_FOUND_MESSAGE,
        SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH,
        SIMPLE_RAG_TEST_RUN_LOG_DIR,
    )
    from simple_rag import answer_query

    test_path = os.getenv("SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH", str(SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH))
    questions = load_test_questions(test_path)
    if not questions:
        print("No test questions found at", test_path, "- add a JSON list of questions or {question, expected_not_found}")
        sys.exit(0)

    log_dir = Path(SIMPLE_RAG_TEST_RUN_LOG_DIR)
    if not log_dir.is_absolute():
        log_dir = BACKEND_DIR / log_dir
    run_handler, run_log_path = _attach_feedback_run_log_handler(log_dir)
    start_time = time.perf_counter()

    results: list[dict] = []
    try:
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
                    "sources_count": len(out.get("sources") or []),
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "question": q,
                    "answer_preview": "",
                    "not_found": True,
                    "expected_not_found": expected_nf,
                    "match": False,
                    "sources_count": 0,
                    "error": str(e),
                })
    finally:
        elapsed = time.perf_counter() - start_time
        _detach_feedback_run_log_handler(run_handler)

    total = len(results)
    not_found_count = sum(1 for r in results if r["not_found"])
    match_count = sum(1 for r in results if r.get("match", True))
    with_expected = sum(1 for r in results if r.get("expected_not_found") is not None)
    accuracy = (match_count / with_expected) if with_expected else None

    report = {
        "total": total,
        "not_found_count": not_found_count,
        "not_found_rate": not_found_count / total if total else 0,
        "expected_match_count": match_count,
        "accuracy_questions_with_expected": with_expected,
        "accuracy": accuracy,
        "results": results,
    }
    out_path = BACKEND_DIR / "output" / "simple_rag_feedback_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    stats_lines = [
        "",
        "=" * 80,
        "FEEDBACK TEST RUN STATS",
        "=" * 80,
        f"Run log file: {run_log_path}",
        f"Finished: {datetime.now(timezone.utc).isoformat()}",
        f"Duration (seconds): {elapsed:.2f}",
        f"Total questions: {total}",
        f"Not found (count): {not_found_count}",
        f"Not found (rate): {not_found_count / total:.2%}" if total else "N/A",
        f"Expected match (count): {match_count}",
        f"Questions with expected_not_found set: {with_expected}",
        f"Accuracy (match / with_expected): {accuracy:.2%}" if accuracy is not None else "Accuracy: N/A (no expected labels)",
        "",
        "Per-question summary:",
    ]
    for i, r in enumerate(results, 1):
        nf = "NOT_FOUND" if r["not_found"] else "answer"
        match_str = "match" if r.get("match", True) else "MISMATCH"
        err = f", error={r.get('error')}" if r.get("error") else ""
        stats_lines.append(f"  [{i}] {nf}, {match_str}, sources={r.get('sources_count', 0)}{err}")
    stats_lines.append("=" * 80)

    try:
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(stats_lines) + "\n")
    except Exception as e:
        print(f"Warning: could not append stats to run log: {e}")

    print(f"Report written to {out_path}: not_found={not_found_count}/{total}, expected_match={match_count}/{total}")
    if accuracy is not None:
        print(f"Accuracy (where expected set): {accuracy:.1%}")
    print(f"Run log (all logs + stats): {run_log_path}")


if __name__ == "__main__":
    main()
