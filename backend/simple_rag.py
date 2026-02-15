"""Simple RAG: embed query -> vector search -> Qwen turns retrieved content into clear sentences.

All prompts and settings from config/env or prompt files. No hardcoded strings or magic numbers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import torch

from config import (
    BACKEND_DIR,
    SNIPPET_CHAR_LIMIT,
    SIMPLE_RAG_ANSWER_MARKER,
    SIMPLE_RAG_CONFABULATION_BLOCKLIST,
    SIMPLE_RAG_ECHO_PHRASES,
    SIMPLE_RAG_FILLER_PHRASES,
    SIMPLE_RAG_LOG_PATH,
    SIMPLE_RAG_MAX_CONTENT_CHARS,
    SIMPLE_RAG_MAX_NEW_TOKENS,
    SIMPLE_RAG_NOT_FOUND_MESSAGE,
    SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC,
    SIMPLE_RAG_OFF_TOPIC_PATTERNS,
    SIMPLE_RAG_OUT_OF_SCOPE_MESSAGE,
    SIMPLE_RAG_PROMPTS_JSON_PATH,
    SIMPLE_RAG_SCOPE_KEYWORDS,
    SIMPLE_RAG_STOP_PHRASES,
    SIMPLE_RAG_STRICT_CITATION,
    SIMPLE_RAG_SYSTEM_PROMPT_DEFAULT,
    SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH,
    SIMPLE_RAG_TOP_K,
    SIMPLE_RAG_USER_TEMPLATE_DEFAULT,
)
from embeddings import embed_query
from supabase_client import get_client
from qwen_model import _load_qwen
from transformers import GenerationConfig

# ---- Logging: file + console ----
_LOG: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _LOG
    if _LOG is not None:
        return _LOG
    _LOG = logging.getLogger("simple_rag")
    _LOG.setLevel(logging.DEBUG)
    _LOG.handlers.clear()
    path = Path(SIMPLE_RAG_LOG_PATH)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _LOG.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(ch)
    return _LOG

_CACHED_PROMPTS: dict[str, str] | None = None


def _load_prompts_json() -> dict[str, str]:
    """Load system and user_template from JSON file; fallback to env/config defaults."""
    global _CACHED_PROMPTS
    if _CACHED_PROMPTS is not None:
        return _CACHED_PROMPTS
    path = Path(SIMPLE_RAG_PROMPTS_JSON_PATH)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            system = (data.get("system") or "").strip() or os.getenv("SIMPLE_RAG_SYSTEM_PROMPT", SIMPLE_RAG_SYSTEM_PROMPT_DEFAULT)
            user_template = (data.get("user_template") or "").strip() or os.getenv("SIMPLE_RAG_USER_TEMPLATE", SIMPLE_RAG_USER_TEMPLATE_DEFAULT)
            _CACHED_PROMPTS = {"system": system, "user_template": user_template}
            return _CACHED_PROMPTS
        except Exception:
            pass
    _CACHED_PROMPTS = {
        "system": os.getenv("SIMPLE_RAG_SYSTEM_PROMPT", SIMPLE_RAG_SYSTEM_PROMPT_DEFAULT),
        "user_template": os.getenv("SIMPLE_RAG_USER_TEMPLATE", SIMPLE_RAG_USER_TEMPLATE_DEFAULT),
    }
    return _CACHED_PROMPTS


def _get_system_prompt() -> str:
    return _load_prompts_json()["system"]


def _get_user_template() -> str:
    return _load_prompts_json()["user_template"]


def _load_test_questions_json() -> dict[str, list[str]]:
    """Load test questions from JSON (keys: sama, arabic, generic)."""
    path = Path(SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    if not path.exists():
        return {"sama": [], "arabic": [], "generic": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "sama": list(data.get("sama") or []),
            "arabic": list(data.get("arabic") or []),
            "generic": list(data.get("generic") or []),
        }
    except Exception:
        return {"sama": [], "arabic": [], "generic": []}


def fetch_chunks(query_embedding: list[float], limit: int | None = None) -> list[dict[str, Any]]:
    """Fetch top-k chunks by vector similarity via Supabase match_chunks RPC."""
    if limit is None:
        limit = SIMPLE_RAG_TOP_K
    client = get_client()
    result = client.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": limit,
            "snippet_char_limit": SNIPPET_CHAR_LIMIT,
        },
    ).execute()
    return result.data or []


def build_context(chunks: list[dict[str, Any]], max_content_chars: int | None = None) -> str:
    """Turn retrieved chunks into one context string (Document, Pages, Content)."""
    if max_content_chars is None:
        max_content_chars = SIMPLE_RAG_MAX_CONTENT_CHARS
    parts: list[str] = []
    for i, row in enumerate(chunks, 1):
        doc = row.get("document_name", "")
        start = row.get("page_start", 0)
        end = row.get("page_end", 0)
        content = (row.get("content") or "").strip()
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "..."
        parts.append(
            f"[Passage {i}] Document: {doc}, Pages: {start}–{end}\nContent:\n{content}"
        )
    return "\n\n".join(parts)


def _truncate_instruction_echo(decoded: str) -> str:
    """Return decoded text up to (not including) the first occurrence of any echo phrase (instruction leakage)."""
    if not decoded or not SIMPLE_RAG_ECHO_PHRASES:
        return decoded.strip() if decoded else ""
    out = decoded
    earliest = len(out)
    for phrase in SIMPLE_RAG_ECHO_PHRASES:
        if not phrase:
            continue
        idx = out.lower().find(phrase.lower())
        if idx >= 0 and idx < earliest:
            earliest = idx
    if earliest < len(out):
        out = out[:earliest]
    return out.strip()


def _truncate_conversation_leakage(decoded: str) -> str:
    """Return decoded text up to (not including) the first occurrence of any stop phrase (conversation-turn leakage)."""
    if not decoded or not SIMPLE_RAG_STOP_PHRASES:
        return decoded.strip() if decoded else ""
    out = decoded
    earliest = len(out)
    for phrase in SIMPLE_RAG_STOP_PHRASES:
        if not phrase:
            continue
        idx = out.lower().find(phrase.lower())
        if idx >= 0 and idx < earliest:
            earliest = idx
    if earliest < len(out):
        out = out[:earliest]
    return out.strip()


def _strip_filler_phrases(text: str) -> str:
    """Remove filler phrases (e.g. 'As mentioned earlier') from model output."""
    if not text or not text.strip():
        return text
    out = text.strip()
    for phrase in SIMPLE_RAG_FILLER_PHRASES:
        if not phrase:
            continue
        while phrase.lower() in out.lower():
            idx = out.lower().find(phrase.lower())
            out = (out[:idx] + out[idx + len(phrase) :]).strip()
            # Collapse double spaces
            out = " ".join(out.split())
    return out


def _is_in_scope(query: str) -> bool:
    """True if the question is about SAMA/NORA/banking/regulations; False for off-topic (e.g. US president)."""
    if not query or not query.strip():
        return False
    q = query.strip().lower()
    # Explicit off-topic patterns → refuse
    for pattern in SIMPLE_RAG_OFF_TOPIC_PATTERNS:
        if pattern and pattern in q:
            return False
    # Must contain at least one in-scope keyword (English or Arabic)
    for keyword in SIMPLE_RAG_SCOPE_KEYWORDS:
        if keyword and keyword in q:
            return True
    return False


def _is_arabic_query(text: str) -> bool:
    """True if text contains Arabic script (U+0600–U+06FF)."""
    if not text or not text.strip():
        return False
    return any("\u0600" <= c <= "\u06FF" for c in text)


def _get_arabic_instruction_prefix() -> str:
    """Instruction to prepend when the question is in Arabic so the model answers in Arabic."""
    return "أجب بالعربية فقط. لا تستخدم الإنجليزية.\n\nAnswer only in Arabic. Do not use English.\n\n"


def _is_answer_mostly_english(text: str) -> bool:
    """True if majority of non-space chars are ASCII (Latin)."""
    if not text or not text.strip():
        return True
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True
    ascii_count = sum(1 for c in chars if ord(c) < 128)
    return ascii_count / len(chars) >= 0.7


# Patterns for definition-style questions (what is X, define X, etc.); group 1 = term
_DEFINITION_QUERY_PATTERNS = [
    re.compile(r"(?i)what\s+is\s+(.+?)\s*\??\s*$"),
    re.compile(r"(?i)define\s+(.+?)\s*\??\s*$"),
    re.compile(r"(?i)what\s+does\s+(.+?)\s+mean\s*\??\s*$"),
    re.compile(r"(?i)meaning\s+of\s+(.+?)\s*\??\s*$"),
]
_DEFINING_PHRASES = (" is ", " stands for ", " refers to ", " means ", " defined as ", " is defined as ")


def _is_definition_style_query(query: str) -> tuple[bool, str]:
    """Return (True, term) if query asks for a definition (e.g. 'What is NORA?'); else (False, '')."""
    if not query or not query.strip():
        return False, ""
    q = query.strip()
    for pat in _DEFINITION_QUERY_PATTERNS:
        m = pat.search(q)
        if m:
            term = m.group(1).strip()
            if len(term) > 60:
                term = term[:60].strip()
            if len(term) >= 2:
                return True, term
    return False, ""


def _context_explicitly_defines_term(context: str, term: str) -> bool:
    """True only if context contains the term and at least one defining phrase (e.g. 'is', 'stands for') in same or adjacent sentence."""
    if not context or not term or not term.strip():
        return False
    term_lower = term.strip().lower()
    context_lower = context.lower()
    if term_lower not in context_lower:
        return False
    segments = re.split(r"[.!?\n]+", context)
    for seg in segments:
        seg_lower = seg.lower()
        if term_lower not in seg_lower:
            continue
        if any(phrase in seg_lower for phrase in _DEFINING_PHRASES):
            return True
    return False


def _get_blocklisted_confabulation_terms(answer: str, context: str) -> list[str]:
    """Return list of blocklist terms that appear in answer but not in context."""
    if not answer or not context:
        return []
    answer_lower = answer.lower()
    context_lower = context.lower()
    found: list[str] = []
    for term in SIMPLE_RAG_CONFABULATION_BLOCKLIST:
        if not term:
            continue
        if term in answer_lower and term not in context_lower:
            found.append(term)
    return found


def _contains_blocklisted_confabulation(answer: str, context: str) -> bool:
    """True if answer contains a blocklist term that does not appear in context."""
    return len(_get_blocklisted_confabulation_terms(answer, context)) > 0


def _remove_sentences_containing_terms(text: str, terms: list[str], min_kept_len: int = 40) -> str | None:
    """Remove sentences that contain any of the given terms (case-insensitive). Return cleaned text or None if too short."""
    if not text or not text.strip() or not terms:
        return text
    text_lower = text.lower()
    # Split on sentence boundaries: . ! ? newline, or bullet
    parts = re.split(r"(?<=[.!?\n])\s+|\s*[\n•]\s*", text)
    kept: list[str] = []
    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue
        part_lower = part_stripped.lower()
        if any(term in part_lower for term in terms if term):
            continue
        kept.append(part_stripped)
    result = " ".join(kept).strip()
    if len(result) < min_kept_len:
        return None
    return result


def _extract_cited_sentences(answer: str) -> list[dict[str, str]]:
    """Extract (sentence or quoted text, citation) pairs from answer for 'exact sentences used' listing."""
    if not answer or not answer.strip():
        return []
    cited = []
    # Find all citations: (Page 1), (Pages 2–3), (Pages 2-3)
    pattern = r"(\(Page\s+\d+\)|\(Pages\s+\d+\s*[–\-]\s*\d+\))"
    last_end = 0
    for m in re.finditer(pattern, answer):
        citation = m.group(1)
        start = m.start()
        # Text from last citation (or start) up to this citation
        segment = answer[last_end:start].strip().rstrip(".").strip()
        # Take last sentence or line if multiple
        if "\n" in segment:
            segment = segment.split("\n")[-1].strip().rstrip(".").strip()
        if len(segment) > 250:
            segment = segment[-247:] + "..."
        if segment:
            cited.append({"text": segment, "citation": citation})
        last_end = m.end()
    return cited


# Minimum length to treat an answer as substantive when it has no (Page X); shorter -> replace with not_found
_SUBSTANTIVE_ANSWER_MIN_LEN = 50


def _ensure_citation_fallback(answer: str, not_found_message: str) -> tuple[str, bool]:
    """If no (Page X) or (Pages X–Y): substantive answers get (Source: provided context); short/non-substantive in strict mode -> not_found. Returns (answer, citation_fallback_applied)."""
    applied = False
    if not answer or not answer.strip():
        return answer, applied
    if answer.strip().lower() == not_found_message.lower():
        return answer, applied
    if "(Page " in answer or "(Pages " in answer:
        return answer, applied
    # No citation: soft fallback for substantive answers (synthesis without proper citation format)
    substantive = len(answer.strip()) >= _SUBSTANTIVE_ANSWER_MIN_LEN
    if substantive:
        applied = True
        return answer.rstrip(".").rstrip() + " (Source: provided context).", applied
    if SIMPLE_RAG_STRICT_CITATION:
        return not_found_message, True
    applied = True
    return answer.rstrip(".").rstrip() + " (Source: provided context).", applied


def generate_answer_simple(context_text: str, user_query: str) -> str:
    """Use Qwen to turn context + question into a detailed, well-formed answer (DB-only)."""
    tokenizer, model = _load_qwen()
    system_prompt = _get_system_prompt()
    user_template = _get_user_template()
    question_for_prompt = user_query
    if _is_arabic_query(user_query):
        question_for_prompt = _get_arabic_instruction_prefix() + user_query
    user_block = user_template.format(context=context_text, question=question_for_prompt)
    prompt = f"{system_prompt}\n\n{user_block}\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    gen_config = GenerationConfig(
        max_new_tokens=SIMPLE_RAG_MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.2,
    )
    # Stop at conversation-turn markers to avoid training-data leakage (post-process truncation is the main fix)
    stop_strings = [p for p in SIMPLE_RAG_STOP_PHRASES if p and "\n" not in p] or None
    with torch.no_grad():
        if stop_strings:
            try:
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_config,
                    stop_strings=stop_strings,
                    tokenizer=tokenizer,
                )
            except Exception:
                outputs = model.generate(**inputs, generation_config=gen_config)
        else:
            outputs = model.generate(**inputs, generation_config=gen_config)

    # Decode only the newly generated tokens (exclude the prompt)
    generated_ids = outputs[0][input_length:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    decoded = _truncate_instruction_echo(decoded)
    decoded = _truncate_conversation_leakage(decoded)
    marker = SIMPLE_RAG_ANSWER_MARKER
    if marker in decoded:
        answer = decoded.rsplit(marker, 1)[-1].strip()
    else:
        answer = decoded.strip()
    answer = _strip_filler_phrases(answer)
    return answer


def answer_query(user_query: str, top_k: int | None = None, category: str | None = None) -> dict[str, Any]:
    """Run simple RAG: embed -> fetch -> build context -> Qwen -> return answer and sources."""
    log = _get_logger()
    if not user_query or not user_query.strip():
        log.info("answer_query: empty query")
        return {"answer": "Please provide a question.", "sources": []}

    q = user_query.strip()
    if category:
        log.debug("query [%s]: %s", category, q)
    else:
        log.debug("query: %s", q)

    if not _is_in_scope(q):
        log.info("out_of_scope; returning SIMPLE_RAG_OUT_OF_SCOPE_MESSAGE")
        return {"answer": SIMPLE_RAG_OUT_OF_SCOPE_MESSAGE, "sources": []}

    k = top_k if top_k is not None else SIMPLE_RAG_TOP_K
    try:
        query_embedding = embed_query(q)
    except Exception as e:
        log.exception("embed_query failed: %s", e)
        raise
    chunks = fetch_chunks(query_embedding, limit=k)
    log.debug("retrieval: %d chunks", len(chunks))
    if chunks:
        top_docs = [(r.get("document_name"), r.get("page_start"), r.get("page_end")) for r in chunks[:3]]
        log.debug("top chunks: %s", top_docs)

    if not chunks:
        log.info("no chunks; returning not_found")
        return {
            "answer": SIMPLE_RAG_NOT_FOUND_MESSAGE,
            "sources": [],
        }

    context_text = build_context(chunks)
    try:
        answer = generate_answer_simple(context_text, q)
    except Exception as e:
        log.exception("generate_answer_simple failed: %s", e)
        raise
    raw_preview = (answer or "")[:300] + ("..." if len(answer or "") > 300 else "")
    log.debug("raw_answer (preview): %s", raw_preview)

    answer, citation_applied = _ensure_citation_fallback(answer, SIMPLE_RAG_NOT_FOUND_MESSAGE)
    if citation_applied:
        log.info("citation_fallback_applied (no Page/Pages in answer)")

    # Definition guard: for "what is X?" / "define X?" if context does not explicitly define X, force not found
    is_def, term = _is_definition_style_query(q)
    if is_def and term and not _is_not_found_answer(answer):
        if not _context_explicitly_defines_term(context_text, term):
            log.info("definition_guard: context does not explicitly define '%s'; overriding to not_found", term)
            answer = SIMPLE_RAG_NOT_FOUND_MESSAGE

    # Grounding check: answer must be supported by context (at least one segment appears in context)
    if not _is_not_found_answer(answer) and not _is_answer_grounded_in_context(
        answer, context_text, SIMPLE_RAG_NOT_FOUND_MESSAGE
    ):
        log.info("grounding_check: answer not supported by context; overriding to not_found")
        answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE

    # Confabulation: blocklist terms in answer but not in context — remove offending sentences; replace whole answer only if nothing left
    offending = _get_blocklisted_confabulation_terms(answer, context_text)
    if offending:
        log.info("blocklisted confabulation detected: %s", ", ".join(offending))
        cleaned = _remove_sentences_containing_terms(answer, offending, min_kept_len=40)
        if cleaned is not None and len(cleaned) >= 40:
            answer = cleaned
            log.debug("kept answer after removing sentences containing blocklist terms")
        else:
            answer = SIMPLE_RAG_NOT_FOUND_MESSAGE
            log.info("answer empty or too short after removing confabulation; replacing with not_found")

    log.debug("final_answer (preview): %s", (answer or "")[:300])
    cited_sentences = _extract_cited_sentences(answer) if answer else []
    snippet_limit = SNIPPET_CHAR_LIMIT
    sources = [
        {
            "document_name": row.get("document_name"),
            "page_start": row.get("page_start"),
            "page_end": row.get("page_end"),
            "snippet": (row.get("snippet") or (row.get("content") or "")[:snippet_limit]),
        }
        for row in chunks
    ]
    return {"answer": answer, "sources": sources, "cited_sentences": cited_sentences}


# Max characters of snippet to print per source in CLI (readability)
_SNIPPET_DISPLAY_CHARS = 280

# Minimum length of a contiguous answer segment that must appear in context for grounding check
_GROUNDING_MIN_SEGMENT_LEN = 25


def _is_answer_grounded_in_context(answer: str, context: str, not_found_message: str) -> bool:
    """True if answer is empty, not_found, or at least one contiguous segment of the answer appears in context."""
    if not answer or not answer.strip():
        return True
    a = answer.strip()
    if a == not_found_message or a == SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC:
        return True
    if not context or not context.strip():
        return False
    # Strip citation patterns and "(Source: provided context)" to get answer body
    body = re.sub(r"\(Page\s+\d+\)|\(Pages\s+\d+\s*[–\-]\s*\d+\)", "", a, flags=re.IGNORECASE)
    body = re.sub(r"\(Source:\s*provided context\.?\)", "", body, flags=re.IGNORECASE)
    body = re.sub(r"\s+", " ", body).strip().lower()
    ctx_norm = re.sub(r"\s+", " ", context.strip()).lower()
    if len(body) < _GROUNDING_MIN_SEGMENT_LEN:
        return body in ctx_norm if body else True
    for i in range(len(body) - _GROUNDING_MIN_SEGMENT_LEN + 1):
        segment = body[i : i + _GROUNDING_MIN_SEGMENT_LEN]
        if segment in ctx_norm:
            return True
    # Also check longer segments (full sentences or 50 chars)
    for n in (_GROUNDING_MIN_SEGMENT_LEN + 1, 50, 80, 120):
        for i in range(len(body) - n + 1):
            if body[i : i + n] in ctx_norm:
                return True
    return False


def _is_not_found_answer(answer: str) -> bool:
    """True if answer is the standard not-found message (English or Arabic)."""
    if not answer or not answer.strip():
        return False
    a = answer.strip()
    return a == SIMPLE_RAG_NOT_FOUND_MESSAGE or a == SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC


def _print_sources(result: dict[str, Any]) -> None:
    """Print sources list; if answer is not-found, add a note and print each source's snippet."""
    sources = result.get("sources") or []
    answer = (result.get("answer") or "").strip()
    if _is_not_found_answer(answer) and sources:
        print("No cited answer could be generated. Here are the passages we retrieved:")
    print("Sources:", len(sources))
    for i, s in enumerate(sources, 1):
        print(f"  {i}. {s.get('document_name')} (pages {s.get('page_start')}-{s.get('page_end')})")
        snippet = (s.get("snippet") or "").strip()
        if snippet:
            display = snippet[: _SNIPPET_DISPLAY_CHARS] + ("..." if len(snippet) > _SNIPPET_DISPLAY_CHARS else "")
            print(f"      {display}")


def run_test_questions() -> None:
    """Load test questions from JSON and run answer_query for each; print and log results."""
    log = _get_logger()
    questions = _load_test_questions_json()
    total = sum(len(q) for q in questions.values())
    log.info("run_test_questions: %d questions (SAMA=%d, Arabic=%d, Generic=%d)", total, len(questions["sama"]), len(questions["arabic"]), len(questions["generic"]))
    n = 0
    for category, qlist in [("SAMA", questions["sama"]), ("Arabic", questions["arabic"]), ("Generic", questions["generic"])]:
        for query in qlist:
            n += 1
            print(f"\n[{n}/{total}] [{category}] {query}")
            print("-" * 60)
            log.debug("[%d/%d] [%s] %s", n, total, category, query)
            try:
                result = answer_query(query, category=category)
            except Exception as e:
                log.exception("answer_query failed: %s", e)
                raise
            print("Answer:", result["answer"])
            if result.get("cited_sentences"):
                print("Sentences used from sources:")
                for i, cs in enumerate(result["cited_sentences"], 1):
                    print(f"  {i}. {cs.get('text', '')} {cs.get('citation', '')}")
            _print_sources(result)
            log.debug("answer (preview): %s", (result["answer"] or "")[:200])
            log.debug("sources_count: %d", len(result["sources"]))
    log.info("run_test_questions done.")
    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].strip() == "--test":
        run_test_questions()
    else:
        query = (
            sys.argv[1]
            if len(sys.argv) > 1
            else "What are the minimum criteria for obtaining a banking license in Saudi Arabia?"
        )
        print("Query:", query)
        print("-" * 60)
        result = answer_query(query)
        print("Answer:", result["answer"])
        if result.get("cited_sentences"):
            print("-" * 60)
            print("Sentences used from sources:")
            for i, cs in enumerate(result["cited_sentences"], 1):
                print(f"  {i}. {cs.get('text', '')} {cs.get('citation', '')}")
        print("-" * 60)
        _print_sources(result)
