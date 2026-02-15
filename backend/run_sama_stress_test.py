"""
SAMA Rulebook 200-question stress test: run all questions through RAG, log to file and CLI, print final stats.
Usage: from backend dir: python run_sama_stress_test.py
       Optional: SAMA_STRESS_TEST_JSON=path/to.json python run_sama_stress_test.py
Output: CLI progress + timestamped log file (full answers + stats) + optional JSON report.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Run from backend directory
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Default path for 200-question set
DEFAULT_STRESS_TEST_JSON = BACKEND_DIR / "prompts" / "sama_stress_test_200.json"
PREVIEW_LEN = 52
LOG_LINE_SEP = "\n" + "-" * 80 + "\n"


def _attach_stress_log_handler(log_dir: Path) -> tuple[logging.FileHandler, Path]:
    """Create a timestamped log file for this run; attach handler to root logger. Returns (handler, path)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"sama_stress_test_{ts}.log"
    path = path.resolve()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(fh)
    return fh, path


def _detach_stress_log_handler(handler: logging.Handler) -> None:
    """Remove handler from root logger and close it."""
    logging.getLogger().removeHandler(handler)
    try:
        handler.close()
    except Exception:
        pass


def load_stress_questions(path: str | Path) -> list[dict]:
    """Load questions from JSON: list of {question, domain?, ...}. Returns list of {question, domain}."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            out.append({"index": i + 1, "question": item, "domain": "General"})
        elif isinstance(item, dict) and item.get("question"):
            out.append({
                "index": i + 1,
                "question": item["question"],
                "domain": item.get("domain") or "General",
            })
        else:
            continue
    # Ensure index 1..N
    for i, row in enumerate(out):
        row["index"] = i + 1
    return out


def _is_not_found(answer: str, not_found_en: str, not_found_ar: str) -> bool:
    """Normalized check: answer is the canonical not-found message (EN or AR)."""
    if not answer or not answer.strip():
        return True
    a = " ".join(answer.strip().lower().split())
    nf_en = " ".join((not_found_en or "").strip().lower().split())
    nf_ar = " ".join((not_found_ar or "").strip().lower().split())
    return a == nf_en or a == nf_ar


def main() -> None:
    from config import (
        SIMPLE_RAG_NOT_FOUND_MESSAGE,
        SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC,
        SIMPLE_RAG_TEST_RUN_LOG_DIR,
    )
    from simple_rag import answer_query

    json_path = os.getenv("SAMA_STRESS_TEST_JSON", str(DEFAULT_STRESS_TEST_JSON))
    questions = load_stress_questions(json_path)
    if not questions:
        print(f"No questions found at {json_path}. Expected JSON array of {{question, domain}}.", file=sys.stderr)
        sys.exit(1)

    log_dir = Path(SIMPLE_RAG_TEST_RUN_LOG_DIR)
    if not log_dir.is_absolute():
        log_dir = BACKEND_DIR / log_dir
    run_handler, run_log_path = _attach_stress_log_handler(log_dir)
    start_wall = time.perf_counter()

    results: list[dict] = []
    log_lines: list[str] = []

    print(f"SAMA Stress Test: {len(questions)} questions from {json_path}")
    print(f"Log file: {run_log_path}\n")

    try:
        for i, row in enumerate(questions):
            idx = row["index"]
            q = row["question"]
            domain = row.get("domain") or "General"
            t0 = time.perf_counter()
            try:
                out = answer_query(q)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                answer = out.get("answer") or ""
                sources = out.get("sources") or []
                n_sources = len(sources)
                not_found = _is_not_found(
                    answer, SIMPLE_RAG_NOT_FOUND_MESSAGE, SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC
                )
                preview = (q[:PREVIEW_LEN] + "…") if len(q) > PREVIEW_LEN else q
                answer_preview = (answer[:120] + "…") if len(answer) > 120 else answer
                status = "NOT_FOUND" if not_found else "OK"
                cli_line = f"[{idx:3d}/{len(questions)}] {domain[:28]:28s} | {status:9s} | src={n_sources:2d} | {elapsed_ms:6.0f}ms | {preview}"
                print(cli_line)

                r = {
                    "index": idx,
                    "domain": domain,
                    "question": q,
                    "answer": answer,
                    "answer_preview": answer_preview,
                    "not_found": not_found,
                    "sources_count": n_sources,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "error": None,
                }
                results.append(r)

                # Append to in-memory log block for file
                log_lines.append(f"Q{idx} [{domain}] {q}")
                log_lines.append(f"  → {status} | sources={n_sources} | {elapsed_ms:.0f}ms")
                log_lines.append(f"  Answer: {answer_preview}")
                log_lines.append("")

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                preview = (q[:PREVIEW_LEN] + "…") if len(q) > PREVIEW_LEN else q
                print(f"[{idx:3d}/{len(questions)}] {domain[:28]:28s} | ERROR     | {elapsed_ms:6.0f}ms | {preview} | {e}")
                results.append({
                    "index": idx,
                    "domain": domain,
                    "question": q,
                    "answer": "",
                    "answer_preview": "",
                    "not_found": True,
                    "sources_count": 0,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "error": str(e),
                })
                log_lines.append(f"Q{idx} [{domain}] {q}")
                log_lines.append(f"  → ERROR: {e}")
                log_lines.append("")

    finally:
        wall_elapsed = time.perf_counter() - start_wall
        _detach_stress_log_handler(run_handler)

    # --- Stats ---
    total = len(results)
    not_found_count = sum(1 for r in results if r["not_found"])
    error_count = sum(1 for r in results if r.get("error"))
    not_found_rate = (not_found_count / total) if total else 0
    avg_ms = sum(r["elapsed_ms"] for r in results) / total if total else 0
    total_sources = sum(r["sources_count"] for r in results)

    by_domain: dict[str, dict] = defaultdict(lambda: {"total": 0, "not_found": 0})
    for r in results:
        d = r["domain"]
        by_domain[d]["total"] += 1
        if r["not_found"]:
            by_domain[d]["not_found"] += 1

    stats_block = [
        "",
        "=" * 80,
        "SAMA STRESS TEST – FINAL STATS",
        "=" * 80,
        f"Log file: {run_log_path}",
        f"Finished: {datetime.now(timezone.utc).isoformat()}",
        f"Wall clock: {wall_elapsed:.2f}s",
        f"Total questions: {total}",
        f"Not found (count): {not_found_count}",
        f"Not found (rate): {not_found_rate:.1%}",
        f"Errors: {error_count}",
        f"Avg latency (ms): {avg_ms:.1f}",
        f"Total sources cited: {total_sources}",
        "",
        "By domain:",
    ]
    for domain in sorted(by_domain.keys()):
        d = by_domain[domain]
        rate = (d["not_found"] / d["total"] * 100) if d["total"] else 0
        stats_block.append(f"  {domain}: {d['not_found']}/{d['total']} not_found ({rate:.1f}%)")
    stats_block.append("")
    stats_block.append("Per-question:")
    for r in results:
        nf = "NOT_FOUND" if r["not_found"] else "OK"
        err = f" error={r.get('error')}" if r.get("error") else ""
        stats_block.append(f"  [{r['index']:3d}] {nf} src={r['sources_count']}{err}")
    stats_block.append("=" * 80)

    stats_text = "\n".join(stats_block)
    print(stats_text)

    # Write full log (answers + stats) to file
    try:
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write("\n--- STRESS TEST OUTPUT (full answers) ---\n")
            f.write(LOG_LINE_SEP.join(log_lines))
            f.write("\n")
            f.write(stats_text)
            f.write("\n")
    except Exception as e:
        print(f"Warning: could not append to run log: {e}", file=sys.stderr)

    # Optional JSON report
    report_path = BACKEND_DIR / "output" / "sama_stress_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total": total,
        "not_found_count": not_found_count,
        "not_found_rate": not_found_rate,
        "error_count": error_count,
        "avg_latency_ms": round(avg_ms, 2),
        "wall_seconds": round(wall_elapsed, 2),
        "total_sources_cited": total_sources,
        "by_domain": {k: dict(v) for k, v in by_domain.items()},
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport JSON: {report_path}")


if __name__ == "__main__":
    main()
