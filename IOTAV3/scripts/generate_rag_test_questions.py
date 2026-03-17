#!/usr/bin/env python3
"""
Generate RAG test questions from Supabase chunk content.

- Fetches rows from Supabase in batches of 1000 (configurable).
- Uses only the content column; samples N rows per batch to stay under OpenAI context.
- Calls OpenAI to generate 3 questions per batch grounded in that content.
- Appends all questions to a single Markdown file in IOTAV3/docs.

Usage (from repo root, with IOTAV3 as cwd or PYTHONPATH):
  cd IOTAV3 && python -m scripts.generate_rag_test_questions

Requires: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY in env or .env.
Optional: SUPABASE_CHUNKS_TABLE (default: sama_nora_chunks), SUPABASE_CONTENT_COLUMN (default: content), OPENAI_MODEL (default: gpt-4o-mini).
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Optional

def _load_dotenv(env_path: Optional[Path] = None) -> None:
    try:
        import dotenv
    except ImportError:
        return
    if env_path and env_path.is_file():
        dotenv.load_dotenv(env_path)
    else:
        script_dir = Path(__file__).resolve().parent
        backend_env = script_dir.parent / "backend" / ".env"
        if backend_env.is_file():
            dotenv.load_dotenv(backend_env)
        dotenv.load_dotenv()

def get_supabase_client():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or load from .env)")
    return create_client(url, key)

def fetch_batch(client, table: str, content_column: str, offset: int, batch_size: int) -> List[dict]:
    select_cols = "id," + content_column if content_column != "id" else "id"
    try:
        r = (
            client.table(table)
            .select(select_cols)
            .order("id")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
    except Exception:
        try:
            r = (
                client.table(table)
                .select(content_column)
                .range(offset, offset + batch_size - 1)
                .execute()
            )
        except Exception as e2:
            raise SystemExit("Supabase query failed: " + str(e2)) from e2
    return (r.data or []) if hasattr(r, "data") else []

def sample_rows(rows: List[dict], sample_size: int) -> List[dict]:
    if len(rows) <= sample_size:
        return rows
    return random.sample(rows, sample_size)

def truncate_content(content: str, max_chars: int = 500) -> str:
    if len(content) <= max_chars:
        return content
    return content[: max_chars - 3].rstrip() + "..."

def generate_questions_with_openai(sampled_content: List[str], model: str) -> List[str]:
    import openai
    combined = "\n\n---\n\n".join(truncate_content(c or "") for c in sampled_content)
    prompt = (
        "Based on the following text chunks from a RAG knowledge base, "
        "generate exactly 3 questions that a user could ask and that should be answerable from this content. "
        "Output only the 3 questions, one per line, no numbering or bullets."
    )
    user_content = prompt + "\n\n" + combined
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=300,
            )
            text = (resp.choices[0].message.content or "").strip()
            lines = []
            for line in text.splitlines():
                line = re.sub(r"^[\d\.\)\-\*]+\s*", "", line.strip()).strip()
                if line and len(lines) < 3:
                    lines.append(line)
            return lines[:3] if lines else ["(no questions generated)"]
        except Exception:
            if attempt == 2:
                raise
    return ["(openai request failed)"]

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAG test questions from Supabase chunks using OpenAI.")
    parser.add_argument("--env", type=Path, default=None, help="Path to .env file")
    parser.add_argument("--table", type=str, default=os.environ.get("SUPABASE_CHUNKS_TABLE", "sama_nora_chunks"), help="Supabase table name")
    parser.add_argument("--content-column", type=str, default=os.environ.get("SUPABASE_CONTENT_COLUMN", "content"), help="Column name for chunk text (default: content)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per batch")
    parser.add_argument("--sample-size", type=int, default=40, help="Rows to send to OpenAI per batch")
    parser.add_argument("--output", type=Path, default=None, help="Output .md file")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after this many batches")
    parser.add_argument("--model", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), help="OpenAI model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    _load_dotenv(args.env)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (or add to .env)")

    random.seed(args.seed)

    if args.output is None:
        script_dir = Path(__file__).resolve().parent
        args.output = script_dir.parent / "docs" / "rag_test_questions.md"
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    client = get_supabase_client()
    table = args.table
    content_column = args.content_column
    batch_size = args.batch_size
    sample_size = min(args.sample_size, batch_size)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# RAG test questions (auto-generated)\n\n")
        f.write("Questions are derived from Supabase chunk content for testing the LLM, RAG, and related modules.\n\n")

    offset = 0
    batch_num = 0
    total_questions = 0

    while True:
        if args.max_batches is not None and batch_num >= args.max_batches:
            break
        rows = fetch_batch(client, table, content_column, offset, batch_size)
        if not rows:
            break

        sampled = sample_rows(rows, sample_size)
        contents = [str(r.get(content_column) or "").strip() for r in sampled]
        if not any(contents):
            offset += len(rows)
            batch_num += 1
            if len(rows) < batch_size:
                break
            continue

        questions = generate_questions_with_openai(contents, args.model)
        batch_num += 1
        total_questions += len(questions)

        with open(args.output, "a", encoding="utf-8") as f:
            f.write("## Batch " + str(batch_num) + " (rows " + str(offset) + "-" + str(offset + len(rows) - 1) + ")\n\n")
            for q in questions:
                f.write("- " + q + "\n")
            f.write("\n")

        print("Batch " + str(batch_num) + ": rows " + str(offset) + "-" + str(offset + len(rows) - 1) + " -> " + str(len(questions)) + " questions", file=sys.stderr)
        offset += len(rows)
        if len(rows) < batch_size:
            break

    print("Done. Wrote " + str(total_questions) + " questions to " + str(args.output), file=sys.stderr)

if __name__ == "__main__":
    main()
