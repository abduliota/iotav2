"""
Backfill section_title for chunks where it is null or empty.
Fetches chunks from Supabase, generates titles from content using Qwen 1.8B (or OpenAI with --use-openai).

Before first run with Qwen, pre-download the model to avoid network issues during fetch:
    huggingface-cli download Qwen/Qwen1.5-1.8B-Chat

Usage (from backend/):
    python scripts/backfill_section_titles.py
    python scripts/backfill_section_titles.py --dry-run
    python scripts/backfill_section_titles.py --limit 10
    python scripts/backfill_section_titles.py --batch-size 50
    python scripts/backfill_section_titles.py --use-openai      # Use OpenAI gpt-4o-mini instead
    python scripts/backfill_section_titles.py --fallback-openai  # Use OpenAI when Qwen fails
    python scripts/backfill_section_titles.py --quiet          # Suppress per-batch progress
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Run from backend so config and imports resolve
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

# Load .env before imports
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(BACKEND_DIR) / ".env")
except ImportError:
    pass

from supabase_client import get_client

PAGE_SIZE = 1000  # Supabase default max
PROGRESS_INTERVAL = 5000  # Print fetch progress every N rows
DEFAULT_BATCH_SIZE = 50
OPENAI_DELAY_SEC = 0.2


def _extract_fallback_title(content: str) -> str | None:
    """Extract a title from content when model fails. Uses first meaningful line (max 80 chars)."""
    if not content or not content.strip():
        return None
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not lines:
        return None
    first = lines[0][:80].rstrip(".,;: ")
    return first if len(first) > 2 else None


def _generate_section_title_qwen(content: str) -> str | None:
    """Generate title using Qwen 1.8B (local). Falls back to first-line extraction if Qwen fails."""
    try:
        from qwen_model import generate_section_title
        title = generate_section_title(content)
        if title:
            return title
    except Exception:
        pass
    return _extract_fallback_title(content)


def _generate_section_title_openai(content: str) -> str | None:
    """Generate title using OpenAI gpt-4o-mini (requires OPENAI_API_KEY)."""
    if not content or not content.strip():
        return None
    truncated = content.strip()[:2000]
    try:
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a short section title (max 80 chars) for this regulatory text. "
                        "Output only the title, no quotes or explanation. Support Arabic and English."
                    ),
                },
                {"role": "user", "content": truncated},
            ],
            max_tokens=100,
        )
        if r.choices and r.choices[0].message and r.choices[0].message.content:
            title = r.choices[0].message.content.strip()
            return title[:80] if title else None
    except Exception:
        pass
    return None


def generate_section_title(
    content: str,
    use_openai: bool = False,
    fallback_openai: bool = False,
) -> str | None:
    """Generate a short section title from content. Uses Qwen by default, OpenAI if --use-openai.
    If fallback_openai, retries with OpenAI when Qwen returns None.
    When both fail, falls back to first-line extraction from content."""
    if use_openai:
        title = _generate_section_title_openai(content)
        return title or _extract_fallback_title(content)
    title = _generate_section_title_qwen(content)
    if title is None and fallback_openai:
        title = _generate_section_title_openai(content)
    return title or _extract_fallback_title(content)


def _needs_title(row: dict) -> bool:
    """True if chunk has no meaningful section_title."""
    st = row.get("section_title")
    return st is None or (isinstance(st, str) and not st.strip())


def fetch_chunks_needing_titles(
    client,
    limit: int | None = None,
    quiet: bool = False,
) -> list[dict]:
    """Fetch chunks where section_title is null or empty. Paginate and optionally cap."""
    rows: list[dict] = []
    start = 0
    use_filter = True
    while True:
        end = start + PAGE_SIZE - 1
        try:
            q = (
                client.table("sama_nora_chunks")
                .select("id, content, section_title")
                .range(start, end)
            )
            if use_filter:
                q = q.or_("section_title.is.null,section_title.eq.")  # noqa: E501
            result = q.execute()
        except Exception:
            if use_filter:
                use_filter = False
                continue
            raise
        page = result.data or []
        prev_count = len(rows)
        for row in page:
            if not _needs_title(row):
                continue
            if limit is not None and len(rows) >= limit:
                return rows
            rows.append(row)
        if len(page) < PAGE_SIZE or (limit is not None and len(rows) >= limit):
            break
        start += PAGE_SIZE
        if not quiet and len(rows) // PROGRESS_INTERVAL > prev_count // PROGRESS_INTERVAL:
            print(f"Fetched {len(rows)} rows needing titles so far...")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill section_title for chunks using Qwen 1.8B (or OpenAI with --use-openai)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and generate titles but do not update DB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Chunks per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap total chunks to process (for testing)",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI gpt-4o-mini instead of Qwen (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--fallback-openai",
        action="store_true",
        help="Use OpenAI when Qwen fails to generate a title (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-batch and fetch progress; show summary only",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw Qwen output for first 2 chunks (for debugging)",
    )
    args = parser.parse_args()

    if args.debug:
        os.environ["DEBUG_SECTION_TITLE"] = "1"

    client = get_client()
    if not args.quiet:
        print("Fetching chunks with empty section_title...")
    rows = fetch_chunks_needing_titles(client, limit=args.limit, quiet=args.quiet)
    total = len(rows)
    if total == 0:
        print("No chunks found with empty section_title.")
        return

    mode = "DRY RUN - " if args.dry_run else ""
    model_name = "OpenAI gpt-4o-mini" if args.use_openai else "Qwen 1.8B"
    if args.fallback_openai and not args.use_openai:
        model_name += " (OpenAI fallback)"
    if not args.quiet:
        print(f"{mode}Processing {total} chunks (batch size: {args.batch_size}, model: {model_name})...")

    updated = 0
    skipped = 0
    errors = 0
    batch_size = args.batch_size

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        for row in batch:
            chunk_id = row.get("id")
            content = row.get("content") or ""
            section_title = row.get("section_title")
            if section_title and str(section_title).strip():
                skipped += 1
                continue
            if not content.strip():
                skipped += 1
                continue
            try:
                title = generate_section_title(
                    content,
                    use_openai=args.use_openai,
                    fallback_openai=args.fallback_openai,
                )
                if title:
                    if not args.dry_run:
                        client.table("sama_nora_chunks").update(
                            {"section_title": title}
                        ).eq("id", chunk_id).execute()
                    updated += 1
                    if not args.quiet and (updated) % 50 == 0:
                        print(f"  Progress: {updated} updated...")
                else:
                    errors += 1
                    if not args.quiet:
                        print(f"  Skip id={chunk_id}: no title generated")
                if args.use_openai or (title and args.fallback_openai):
                    time.sleep(OPENAI_DELAY_SEC)
            except Exception as e:
                errors += 1
                if not args.quiet:
                    print(f"  Error id={chunk_id}: {e}")
        if not args.quiet:
            print(
                f"  Batch {i // batch_size + 1}: "
                f"{min(i + batch_size, total)}/{total} processed, "
                f"{updated} updated, {errors} errors"
            )

    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Total fetched: {total}")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    if args.dry_run:
        print("  (Dry run - no DB updates)")


if __name__ == "__main__":
    main()
