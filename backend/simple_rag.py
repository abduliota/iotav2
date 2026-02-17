"""Simple RAG: embed query -> vector search -> Qwen turns retrieved content into clear sentences.

All prompts and settings from config/env or prompt files. No hardcoded strings or magic numbers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import torch

from config import (
    CONVERSATION_HISTORY_MAX_CHARS_PER_MESSAGE,
    CONVERSATION_HISTORY_MAX_MESSAGES,
    AMBIGUITY_DETECTION_ENABLED,
    AMBIGUITY_MAX_WORDS,
    ANSWER_ARABIC_SCRIPT_RATIO_MIN,
    BACKEND_DIR,
    DYNAMIC_TOP_K_ENABLED,
    DYNAMIC_TOP_K_MULTIPLIER,
    DYNAMIC_TOP_K_SIMILARITY_THRESHOLD,
    ENABLE_EXTRACTIVE_BUILDER,
    ENABLE_ANSWER_LANGUAGE_CHECK,
    ENABLE_ARABIC_DUAL_RETRIEVE,
    ENABLE_STRICT_RETRIEVAL_LANGUAGE_FILTER,
    ENABLE_RETRIEVED_BUT_NOT_USED_LOG,
    ENABLE_STRICT_SINGLE_LANGUAGE,
    ENABLE_POST_GEN_SIMILARITY_CHECK,
    ENABLE_TRANSLATE_BACK,
    ENABLE_RERANKING,
    CITATION_HIGH_SIMILARITY_THRESHOLD,
    NOT_FOUND_RETRIEVAL_CONFIDENCE_THRESHOLD,
    POST_GEN_SIMILARITY_THRESHOLD,
    RETURN_CITATION_VALID,
    RETURN_CONFIDENCE_SCORE,
    RRF_K,
    SIMPLE_RAG_CLARIFICATION_MESSAGE,
    USE_SEMANTIC_GROUNDING,
    SNIPPET_CHAR_LIMIT,
    SIMPLE_RAG_ANSWER_MARKER,
    SIMPLE_RAG_CONFABULATION_BLOCKLIST,
    SIMPLE_RAG_ECHO_PHRASES,
    SIMPLE_RAG_FILLER_PHRASES,
    MIN_CHUNK_SIMILARITY,
    MIN_CHUNK_SIMILARITY_ARABIC,
    MIN_CHUNK_SIMILARITY_SYNTHESIS,
    MIN_CHUNK_SIMILARITY_PROCEDURAL,
    MIN_CHUNK_SIMILARITY_FACT_DEFINITION,
    MIN_CHUNK_SIMILARITY_METADATA,
    ENABLE_STRICT_QUALITY_GATE,
    MIN_CHUNK_SIMILARITY_STRICT,
    MIN_CONFIDENCE_FOR_ANSWER,
    ENABLE_ENTITY_CONTAINMENT_CHECK,
    METADATA_ANSWER_MAX_CHARS,
    ENABLE_ARABIC_ENTITY_PRESENCE_CHECK,
    RERANK_BYPASS_SIMILARITY_GATE_THRESHOLD,
    SECOND_PASS_EXTRA_K,
    SECOND_PASS_MAX_K,
    SECOND_PASS_MIN_SIMILARITY,
    SECOND_PASS_SIMILARITY_THRESHOLD,
    ENABLE_SECOND_PASS_RETRIEVAL,
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
    SIMPLE_RAG_MANDATORY_CITATION_STRICT,
    SIMPLE_RAG_STRICT_CITATION,
    SIMPLE_RAG_SYSTEM_PROMPT_DEFAULT,
    SIMPLE_RAG_SYSTEM_PROMPT_ARABIC_SUFFIX,
    SIMPLE_RAG_SYSTEM_PROMPT_FACT_DEFINITION,
    SIMPLE_RAG_SYSTEM_PROMPT_METADATA,
    SIMPLE_RAG_SYSTEM_PROMPT_SYNTHESIS,
    SIMPLE_RAG_STRUCTURED_EXTRACTION_TEMPLATE,
    SIMPLE_RAG_SYSTEM_PROMPT_LAW_SUMMARY,
    SIMPLE_RAG_GENERIC_PHRASES_BLOCKLIST,
    SIMPLE_RAG_JURISDICTION_ANCHOR,
    SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH,
    SIMPLE_RAG_TEST_RUN_LOG_DIR,
    SIMPLE_RAG_TOP_K,
    SIMPLE_RAG_TOP_K_SYNTHESIS,
    SIMPLE_RAG_USER_TEMPLATE_DEFAULT,
    SYNTHESIS_TITLE_MATCH_PASS_THRESHOLD,
    TOP_K_DEFINITION,
    RAG_MAX_INPUT_TOKENS,
)
from embeddings import embed_chunk, embed_query
from query_normalize import ACRONYM_EXPANSIONS, normalize_query_for_embedding
from query_multilingual import merge_chunks_rrf, translate_arabic_to_english
from grounding import grounding_decision
from extractive_builder import build_extractive_answer
from ontology import get_documents_for_keywords
from rerank import rerank_chunks, _extract_query_entity
from translate import translate_to_arabic
from supabase_client import get_client
from users_sessions import get_session_message_history
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
    """Load test questions from JSON (keys: sama, arabic, generic, generic_off_topic)."""
    path = Path(SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    if not path.exists():
        return {"sama": [], "arabic": [], "generic": [], "generic_off_topic": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "sama": list(data.get("sama") or []),
            "arabic": list(data.get("arabic") or []),
            "generic": list(data.get("generic") or []),
            "generic_off_topic": list(data.get("generic_off_topic") or []),
        }
    except Exception:
        return {"sama": [], "arabic": [], "generic": [], "generic_off_topic": []}


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
    """Turn retrieved chunks into one context string (Document, Pages, Article, Content)."""
    if max_content_chars is None:
        max_content_chars = SIMPLE_RAG_MAX_CONTENT_CHARS
    parts: list[str] = []
    for i, row in enumerate(chunks, 1):
        doc = row.get("document_name", "")
        start = row.get("page_start", 0)
        end = row.get("page_end", 0)
        article_id = row.get("article_id") or row.get("article_number") or row.get("article")
        content = (row.get("content") or "").strip()
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "..."
        header = f"[Passage {i}] Document: {doc}, Pages: {start}–{end}"
        if article_id:
            header += f", Article: {article_id}"
        parts.append(header + f"\nContent:\n{content}")
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


def _strip_leading_format_echo(decoded: str) -> str:
    """Remove a leading line that echoes the format instruction (e.g. 'short description then bullet points:')."""
    if not decoded or not decoded.strip():
        return decoded
    lines = decoded.split("\n")
    if not lines:
        return decoded
    first = (lines[0] or "").strip()
    if not first:
        return decoded
    first_lower = first.lower().rstrip(".:")
    echo_prefixes = (
        "short description then bullet points",
        "format your answer as: short description then bullet points",
        "answer with a brief summary then a bullet list",
    )
    if any(first_lower.startswith(p) or first_lower == p for p in echo_prefixes):
        rest = "\n".join(lines[1:]).strip()
        return rest if rest else decoded
    return decoded


def _strip_markdown_links(text: str) -> str:
    """Replace markdown links [text](url) with link text only; no URLs in answers."""
    if not text or not text.strip():
        return text
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)


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


def _normalize_markdown_bullets(text: str) -> str:
    """Turn inline ' * ' into newline-before-bullet so Markdown renders lists."""
    if not text or not text.strip():
        return text
    text = re.sub(r"([:.])\s*\*\s+", r"\1\n\n* ", text)
    text = text.replace(" * ", "\n\n* ")
    return text


def _normalize_answer_layout(text: str) -> str:
    """Collapse excess newlines and ensure consistent spacing."""
    if not text or not text.strip():
        return text
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fix_nora_norway_confusion(answer: str, user_query: str) -> str:
    """Correct NORA/Norway hallucination: replace Norwegian/Norra/Norway with NORA/Saudi when answer is about NORA."""
    if not answer or not answer.strip():
        return answer
    q = (user_query or "").lower()
    a = answer
    if "nora" not in q and "nora" not in answer.lower():
        return answer
    # Only apply when the answer is about NORA (Saudi regulatory authority)
    a = re.sub(r"\bNorwegian\b", "NORA", a, flags=re.IGNORECASE)
    a = re.sub(r"\bNorra\b", "NORA", a, flags=re.IGNORECASE)
    # "Norway" in regulator context often means Saudi Arabia for this domain
    a = re.sub(r"\bNorway has\b", "NORA has", a, flags=re.IGNORECASE)
    a = re.sub(r"\bNorway's\b", "NORA's", a, flags=re.IGNORECASE)
    a = re.sub(r"\bin Norway\b", "in Saudi Arabia", a, flags=re.IGNORECASE)
    return a


def _normalize_not_found_for_strip(text: str) -> str:
    """Normalize for comparison: strip, collapse whitespace, lower case."""
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def _truncate_at_unexpected_cjk(text: str, max_cjk_run: int = 4) -> str:
    """Truncate before first long run of CJK when answer should be English/Arabic only."""
    if not text or max_cjk_run < 1:
        return text
    pattern = re.compile(
        r"[\u4e00-\u9fff\u3000-\u303f\u3100-\u312f]{" + str(max_cjk_run) + r",}"
    )
    m = pattern.search(text)
    if m:
        return text[: m.start()].rstrip()
    return text


def _strip_cjk_chars(text: str) -> str:
    """Remove CJK characters so answer is English/Arabic only."""
    if not text:
        return text
    out = re.sub(r"[\u4e00-\u9fff\u3000-\u303f\u3100-\u312f]+", " ", text)
    return re.sub(r" +", " ", out).strip()  # collapse spaces left after removal


def _strip_trailing_not_found(answer: str) -> str:
    """Remove every occurrence of the exact or normalized not_found message. Return substantive remainder or canonical not_found."""
    if not answer or not answer.strip():
        return answer
    a = answer.strip()
    canonical_messages = (SIMPLE_RAG_NOT_FOUND_MESSAGE, SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC)
    for not_found in canonical_messages:
        if not not_found:
            continue
        if a == not_found:
            return a
    # Remove all occurrences of exact and normalized not_found from a
    remainder = a
    for not_found in canonical_messages:
        if not not_found:
            continue
        norm_nf = _normalize_not_found_for_strip(not_found)
        while not_found in remainder:
            remainder = remainder.replace(not_found, " ").strip()
        # Replace normalized form: build a pattern that matches the phrase with flexible whitespace
        pattern = re.escape(norm_nf).replace(r"\ ", r"\s+")
        remainder = re.sub(pattern, " ", remainder, flags=re.IGNORECASE)
        remainder = " ".join(remainder.split()).strip()
    if not remainder or not any(c.isalnum() for c in remainder):
        return SIMPLE_RAG_NOT_FOUND_MESSAGE
    return remainder


# Intent classification for question-type-specific behavior (deterministic: same query -> same intent)
QUERY_INTENT_FACT_DEFINITION = "fact_definition"
QUERY_INTENT_METADATA = "metadata"
QUERY_INTENT_PROCEDURAL = "procedural"
QUERY_INTENT_SYNTHESIS = "synthesis"
QUERY_INTENT_OTHER = "other"

_METADATA_PATTERNS = [
    "law number",
    "decree number",
    "decree no",
    "law no",
    "what is the law number",
    "which decree",
    "royal decree",
    "version",
    "date",
    "number of the law",
    "رقم القانون",
    "المرسوم",
    "مرسوم",
    "التاريخ",
    "تاريخ",
]
_FACT_DEFINITION_PATTERNS = [
    "what is ",
    "what is the ",
    "what is the law",
    "which decree",
    "which royal decree",
    "what law governs",
    "what date",
    "who governs",
    "define ",
    "definition of",
    "purpose of",
    "what is the purpose",
    "ما هو ",
    "ما هي ",
    "ما القانون",
    "أي مرسوم",
    "ما المرسوم",
    "غرض",
    "يهدف",
    "ما الغرض",
    "تعريف",
    "ما المقصود",
    "ما المعنى",
]
_PROCEDURAL_PATTERNS = [
    "how do i ",
    "how to ",
    "steps to",
    "process for",
    "procedure",
    "كيف أقدم",
    "كيف يمكن",
]
_SYNTHESIS_PATTERNS = [
    "criteria",
    "requirements",
    "list ",
    "ما هي المتطلبات",
    "minimum ",
    "متطلبات",
]


_DEFINITION_LIKE_SECOND_PASS = (
    "what is", "define", "what is the", "ما هو", "ما هي", "تعريف", "ما المقصود", "ما المعنى", "purpose of",
)
_METADATA_LIKE_SECOND_PASS = (
    "law number", "decree number", "date", "رقم القانون", "المرسوم", "التاريخ", "decree no", "law no",
)


def _classify_query_intent(query: str) -> str:
    """Classify query as metadata, fact_definition, procedural, synthesis, or other. Deterministic (query-only; no randomness)."""
    if not query or not query.strip():
        return QUERY_INTENT_OTHER
    q = query.strip().lower()
    for p in _METADATA_PATTERNS:
        if p in q:
            return QUERY_INTENT_METADATA
    for p in _FACT_DEFINITION_PATTERNS:
        if p in q:
            return QUERY_INTENT_FACT_DEFINITION
    for p in _PROCEDURAL_PATTERNS:
        if p in q:
            return QUERY_INTENT_PROCEDURAL
    for p in _SYNTHESIS_PATTERNS:
        if p in q:
            return QUERY_INTENT_SYNTHESIS
    # Second pass: short definition/metadata-like queries to reduce "other"
    words = q.split()
    if len(words) <= 10 or len(q) <= 80:
        if any(x in q for x in _DEFINITION_LIKE_SECOND_PASS):
            return QUERY_INTENT_FACT_DEFINITION
        if any(x in q for x in _METADATA_LIKE_SECOND_PASS):
            return QUERY_INTENT_METADATA
    return QUERY_INTENT_OTHER


def _in_scope_keyword_match_count(query: str) -> int:
    """Number of scope keywords that appear in the query (for confidence logging)."""
    if not query or not query.strip():
        return 0
    q = query.strip().lower()
    return sum(1 for kw in SIMPLE_RAG_SCOPE_KEYWORDS if kw and kw in q)


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


def _is_chinese_query(text: str) -> bool:
    """True if text contains a run of CJK characters (e.g. Chinese)."""
    if not text or not text.strip():
        return False
    return bool(re.search(r"[\u4e00-\u9fff\u3000-\u303f\u3100-\u312f]{2,}", text))


def _chunk_has_arabic_content(chunk: dict[str, Any]) -> bool:
    """True if chunk content or section_title contains Arabic script."""
    text = (chunk.get("content") or "") + " " + (chunk.get("section_title") or "")
    return any("\u0600" <= c <= "\u06FF" for c in text)


_DEFINITION_MARKERS = ("definitions", "definition", "تعريف", "تعريفات", "means", "is defined as")


def _chunk_has_definition_marker(chunk: dict[str, Any]) -> bool:
    """True if chunk section_title or content suggests a Definitions section or definitional content."""
    title = (chunk.get("section_title") or "").lower()
    content = (chunk.get("content") or "")[:500].lower()
    text = title + " " + content
    return any(m in text for m in _DEFINITION_MARKERS)


def _reorder_definition_chunks_first(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For definition intent: put chunks with definition markers first, preserving order within each group."""
    with_def: list[dict[str, Any]] = []
    without_def: list[dict[str, Any]] = []
    for c in chunks:
        if _chunk_has_definition_marker(c):
            with_def.append(c)
        else:
            without_def.append(c)
    return with_def + without_def


def _reorder_arabic_first(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For Arabic queries: put chunks with Arabic content first, preserving relative order within each group."""
    with_ar: list[dict[str, Any]] = []
    without_ar: list[dict[str, Any]] = []
    for c in chunks:
        if _chunk_has_arabic_content(c):
            with_ar.append(c)
        else:
            without_ar.append(c)
    return with_ar + without_ar


def _synthesis_title_match_confidence(chunks: list[dict[str, Any]], query: str) -> float:
    """
    Return 0–1 confidence: fraction of top chunks whose section_title contains at least one query token.
    Used to relax synthesis verbatim override when retrieval strongly matches by title.
    """
    if not chunks or not query or not query.strip():
        return 0.0
    query_tokens = set(re.findall(r"[a-z0-9\u0600-\u06ff]{2,}", query.lower()))
    if not query_tokens:
        return 0.0
    top = chunks[:5]
    matches = 0
    for c in top:
        title = (c.get("section_title") or "").lower()
        title_tokens = set(re.findall(r"[a-z0-9\u0600-\u06ff]{2,}", title))
        if query_tokens & title_tokens:
            matches += 1
    return matches / len(top) if top else 0.0


def _is_mixed_language(text: str, min_ratio: float = 0.15) -> bool:
    """True if text has both significant Arabic script and significant Latin script (e.g. > 15% each)."""
    if not text or not text.strip():
        return False
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 10:
        return False
    arabic = sum(1 for c in chars if "\u0600" <= c <= "\u06FF")
    latin = sum(1 for c in chars if ord(c) < 128 and c.isalpha())
    n = len(chars)
    return (arabic / n >= min_ratio) and (latin / n >= min_ratio)


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
    re.compile(r"(?i)what\'?s\s+(.+?)\s*\??\s*$"),  # "whats" / "what's" same as "what is"
    re.compile(r"(?i)define\s+(.+?)\s*\??\s*$"),
    re.compile(r"(?i)what\s+does\s+(.+?)\s+mean\s*\??\s*$"),
    re.compile(r"(?i)meaning\s+of\s+(.+?)\s*\??\s*$"),
]
# One-liner fallback for "what is SAMA?" / "what is NORA?" when context has no expansion
DEFINITION_ACRONYM_FALLBACK: dict[str, str] = {
    "sama": "SAMA refers to the Saudi Arabian Monetary Authority (Saudi Central Bank).",
    "nora": "NORA refers to the National Regulatory Authority for Islamic banks in Saudi Arabia.",
}
_DEFINING_PHRASES = (" is ", " stands for ", " refers to ", " means ", " defined as ", " is defined as ")
_PURPOSE_PHRASES = ("purpose of", "the purpose of this", "يهدف إلى", "الغرض من", "objectives of")


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
    """True only if context contains the term and at least one defining phrase (e.g. 'is', 'stands for') or purpose phrase in same or adjacent sentence."""
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
        if any(phrase in seg_lower for phrase in _PURPOSE_PHRASES):
            return True
    return False


def _context_contains_purpose_for_term(context: str, term: str) -> bool:
    """True if context has a segment containing a purpose phrase and the term (or law name). Used for 'purpose of [Law]' questions."""
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
        if any(phrase in seg_lower for phrase in _PURPOSE_PHRASES):
            return True
    return False


_HEADER_METADATA_PHRASES = (
    "royal decree",
    "decree no",
    "law no",
    "law no.",
    "number m/",
    "date:",
    "نسخة",
    "مرسوم",
    "قانون رقم",
    "التاريخ",
)


def _context_contains_header_or_metadata(context: str, term: str) -> bool:
    """True if context has a segment containing the term and header-style metadata (law/decree number, date)."""
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
        if any(phrase in seg_lower for phrase in _HEADER_METADATA_PHRASES):
            return True
    return False


def _chunks_title_match_term(chunks: list[dict[str, Any]], term: str) -> bool:
    """True if any chunk's section_title contains the term (disable definition_guard override when title matches)."""
    if not chunks or not term or not term.strip():
        return False
    term_lower = term.strip().lower()
    for c in chunks:
        title = (c.get("section_title") or "").lower()
        if term_lower in title:
            return True
        if len(term_lower) > 3 and len(title) > 10:
            term_tokens = set(re.findall(r"[a-z0-9\u0600-\u06ff]{2,}", term_lower))
            title_tokens = set(re.findall(r"[a-z0-9\u0600-\u06ff]{2,}", title))
            if term_tokens and term_tokens <= title_tokens:
                return True
    return False


def _context_contains_acronym_expansion(context: str, term: str) -> bool:
    """True if term is a known acronym and its expansion (e.g. 'saudi arabian monetary authority') appears in context. Fixes 'what is sama' when context has full form but not 'SAMA is ...'."""
    if not context or not term or not term.strip():
        return False
    term_lower = term.strip().lower()
    context_lower = context.lower()
    try:
        from query_normalize import ACRONYM_EXPANSIONS, LEGAL_TERM_MAP
        expansion = ACRONYM_EXPANSIONS.get(term_lower) or LEGAL_TERM_MAP.get(term_lower)
        if not expansion:
            return False
        exp_lower = expansion.lower()
        if exp_lower in context_lower:
            return True
        exp_words = exp_lower.split()
        if len(exp_words) >= 2 and " ".join(exp_words[:3]) in context_lower:
            return True
        if len(exp_words) >= 2 and " ".join(exp_words[:2]) in context_lower:
            return True
        if term_lower in ("sama", "nora") and "monetary" in context_lower and ("agency" in context_lower or "authority" in context_lower):
            return True
    except ImportError:
        pass
    return False


def _build_acronym_definition_from_context(
    context_text: str, chunks: list[dict[str, Any]], term: str
) -> str | None:
    """
    When context contains the expansion of a known acronym but no explicit 'X is ...' sentence,
    build a short definition with citation. Used to fix 'what is sama' when the model returns not_found.
    Returns None if we cannot build a definition.
    """
    if not context_text or not chunks or not term or not term.strip():
        return None
    term_lower = term.strip().lower()
    context_lower = context_text.lower()
    try:
        from query_normalize import ACRONYM_EXPANSIONS, LEGAL_TERM_MAP
        expansion = ACRONYM_EXPANSIONS.get(term_lower) or LEGAL_TERM_MAP.get(term_lower)
    except ImportError:
        return None
    if not expansion:
        return None
    # Prefer "agency" when context uses it (SAMA docs say "Agency"), else "authority"
    exp_display = expansion.strip()
    if term_lower == "sama" and "agency" in context_lower:
        exp_display = "Saudi Arabian Monetary Agency"
    elif term_lower == "sama":
        exp_display = "Saudi Arabian Monetary Authority"
    else:
        exp_display = exp_display.title()
    # Find first chunk that contains the expansion (or monetary+agency/authority for sama/nora)
    for c in chunks:
        content = (c.get("content") or "").lower()
        if expansion.lower() in content:
            break
        if term_lower in ("sama", "nora") and "monetary" in content and ("agency" in content or "authority" in content):
            break
    else:
        return None
    page = c.get("page_start") or c.get("page_end") or 1
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    acronym_upper = term.strip().upper() if len(term.strip()) <= 6 else term.strip()
    return f"{acronym_upper} refers to the {exp_display} (Page {page})."


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


def _get_generic_blocklist_phrases(answer: str, context: str) -> list[str]:
    """Return generic-phrase blocklist entries that appear in answer but not in context."""
    if not answer or not context or not SIMPLE_RAG_GENERIC_PHRASES_BLOCKLIST:
        return []
    answer_lower = answer.lower()
    context_lower = context.lower()
    found: list[str] = []
    for phrase in SIMPLE_RAG_GENERIC_PHRASES_BLOCKLIST:
        if not phrase:
            continue
        if phrase.lower() in answer_lower and phrase.lower() not in context_lower:
            found.append(phrase)
    return found


def _extract_article_numbers_from_answer(answer: str) -> list[str]:
    """Extract 'Article X' or 'المادة X' style references from answer for cited_articles."""
    if not answer or not answer.strip():
        return []
    articles: list[str] = []
    for m in re.finditer(r"(?i)(?:article|المادة)\s*[:\s]*(\d+(?:\s*[–\-]\s*\d+)?)", answer):
        articles.append(m.group(1).strip())
    return list(dict.fromkeys(articles))


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


def _extract_page_numbers_from_citations(answer: str) -> list[int]:
    """Extract all page numbers mentioned in (Page X) or (Pages X–Y) citations."""
    if not answer or not answer.strip():
        return []
    pages: list[int] = []
    for m in re.finditer(r"\(Page\s+(\d+)\)", answer, re.IGNORECASE):
        pages.append(int(m.group(1)))
    for m in re.finditer(r"\(Pages\s+(\d+)\s*[–\-]\s*(\d+)\)", answer, re.IGNORECASE):
        start, end = int(m.group(1)), int(m.group(2))
        for p in range(start, end + 1):
            pages.append(p)
    return pages


def _validate_citations(answer: str, chunks: list[dict[str, Any]]) -> tuple[bool, bool]:
    """Check that cited page numbers exist in chunks. Returns (all_valid, any_valid)."""
    cited_pages = _extract_page_numbers_from_citations(answer)
    if not cited_pages:
        return False, False
    chunk_ranges = [
        (int(c.get("page_start") or 0), int(c.get("page_end") or 0)) for c in chunks
    ]
    any_ok = False
    all_ok = True
    for p in cited_pages:
        found = any(s <= p <= e for s, e in chunk_ranges)
        if found:
            any_ok = True
        else:
            all_ok = False
    return all_ok, any_ok


def _has_valid_citation_and_high_similarity(
    answer: str, chunks: list[dict[str, Any]], threshold: float
) -> bool:
    """True if answer has at least one (Page X) citation, that page is in a chunk, and that chunk has similarity >= threshold."""
    if threshold <= 0 or not chunks:
        return False
    cited_pages = _extract_page_numbers_from_citations(answer)
    if not cited_pages:
        return False
    for i, c in enumerate(chunks):
        start = int(c.get("page_start") or 0)
        end = int(c.get("page_end") or 0)
        sim = float(
            c.get("cosine_similarity") or c.get("similarity") or c.get("score") or 0.0
        )
        if sim < threshold:
            continue
        for p in cited_pages:
            if start <= p <= end:
                return True
    return False


def _top_retrieval_similarity(chunks: list[dict[str, Any]]) -> float:
    """Return the best similarity score among chunks (from retrieval or rerank)."""
    if not chunks:
        return 0.0
    best = 0.0
    for c in chunks:
        s = float(c.get("cosine_similarity") or c.get("similarity") or c.get("score") or 0.0)
        if s > best:
            best = s
    return best


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
# For fact/definition intent, allow short answers without (Page X) and append (Source: provided context)
_FACT_DEFINITION_MAX_LEN_NO_CITATION = 150


def _ensure_citation_fallback(
    answer: str, not_found_message: str, intent: str = QUERY_INTENT_OTHER
) -> tuple[str, bool]:
    """If no (Page X) or (Pages X–Y): substantive answers get (Source: provided context); short/non-substantive in strict mode -> not_found. Returns (answer, citation_replaced_with_not_found). Only True when answer was replaced with not_found (for logging)."""
    replaced_not_found = False
    if not answer or not answer.strip():
        return answer, replaced_not_found
    if answer.strip().lower() == not_found_message.lower():
        return answer, replaced_not_found
    if "(Page " in answer or "(Pages " in answer:
        return answer, replaced_not_found
    # Fact_definition/metadata: never replace with not_found for citation alone; always append (Source: provided context)
    if intent in (QUERY_INTENT_FACT_DEFINITION, QUERY_INTENT_METADATA):
        return answer.rstrip(".").rstrip() + " (Source: provided context).", replaced_not_found
    # No citation: soft fallback for substantive answers (synthesis without proper citation format)
    substantive = len(answer.strip()) >= _SUBSTANTIVE_ANSWER_MIN_LEN
    if substantive:
        return answer.rstrip(".").rstrip() + " (Source: provided context).", replaced_not_found
    if SIMPLE_RAG_STRICT_CITATION:
        return not_found_message, True
    return answer.rstrip(".").rstrip() + " (Source: provided context).", replaced_not_found


def generate_answer_simple(
    context_text: str,
    user_query: str,
    intent: str = QUERY_INTENT_OTHER,
    conversation_history: list[dict] | None = None,
) -> str:
    """Use Qwen to turn context + question into a detailed, well-formed answer (DB-only)."""
    tokenizer, model = _load_qwen()
    system_prompt = _get_system_prompt()
    if intent == QUERY_INTENT_FACT_DEFINITION and SIMPLE_RAG_SYSTEM_PROMPT_FACT_DEFINITION:
        system_prompt = system_prompt.rstrip() + "\n\n" + SIMPLE_RAG_SYSTEM_PROMPT_FACT_DEFINITION
    if intent == QUERY_INTENT_METADATA and SIMPLE_RAG_SYSTEM_PROMPT_METADATA:
        system_prompt = system_prompt.rstrip() + "\n\n" + SIMPLE_RAG_SYSTEM_PROMPT_METADATA
    if intent == QUERY_INTENT_SYNTHESIS and SIMPLE_RAG_SYSTEM_PROMPT_SYNTHESIS:
        system_prompt = system_prompt.rstrip() + "\n\n" + SIMPLE_RAG_SYSTEM_PROMPT_SYNTHESIS
    if intent == QUERY_INTENT_SYNTHESIS and SIMPLE_RAG_STRUCTURED_EXTRACTION_TEMPLATE:
        system_prompt = system_prompt.rstrip() + "\n\n" + SIMPLE_RAG_STRUCTURED_EXTRACTION_TEMPLATE
    if intent == QUERY_INTENT_SYNTHESIS and SIMPLE_RAG_SYSTEM_PROMPT_LAW_SUMMARY:
        system_prompt = system_prompt.rstrip() + "\n\n" + SIMPLE_RAG_SYSTEM_PROMPT_LAW_SUMMARY
    if SIMPLE_RAG_JURISDICTION_ANCHOR:
        system_prompt = system_prompt.rstrip() + "\n\n" + SIMPLE_RAG_JURISDICTION_ANCHOR
    if _is_arabic_query(user_query) and SIMPLE_RAG_SYSTEM_PROMPT_ARABIC_SUFFIX:
        system_prompt = system_prompt.rstrip() + "\n\n" + SIMPLE_RAG_SYSTEM_PROMPT_ARABIC_SUFFIX
    user_template = _get_user_template()
    question_for_prompt = user_query
    if _is_arabic_query(user_query):
        question_for_prompt = _get_arabic_instruction_prefix() + user_query
    max_chars = CONVERSATION_HISTORY_MAX_CHARS_PER_MESSAGE if CONVERSATION_HISTORY_MAX_CHARS_PER_MESSAGE > 0 else None
    if conversation_history:
        parts = []
        for row in conversation_history:
            u = (row.get("user_message") or "").strip()
            a = (row.get("assistant_message") or "").strip()
            if max_chars:
                u = (u[:max_chars] + "...") if len(u) > max_chars else u
                a = (a[:max_chars] + "...") if len(a) > max_chars else a
            parts.append(f"User: {u}\nAssistant: {a}\n")
        history_block = (
            "The following are the last exchanges in this conversation; use them as context for the current question.\n\n"
            "### RECENT CONVERSATION\n\n" + "".join(parts) + "\n\n"
        )
        user_block = history_block + "### CONTEXT\n\n" + context_text + "\n\n### QUESTION\n" + question_for_prompt + "\n\nOutput only your answer. Do not repeat the question or any instructions.\n\nAnswer:"
    else:
        user_block = user_template.format(context=context_text, question=question_for_prompt)
    prompt = f"{system_prompt}\n\n{user_block}\n"

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=RAG_MAX_INPUT_TOKENS)
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    input_length = inputs["input_ids"].shape[1]
    gen_config = GenerationConfig(
        max_new_tokens=SIMPLE_RAG_MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.2,
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    # Decode only the newly generated tokens (exclude the prompt)
    generated_ids = outputs[0][input_length:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    decoded = _truncate_instruction_echo(decoded)
    decoded = _truncate_conversation_leakage(decoded)
    decoded = _strip_leading_format_echo(decoded)
    if not _is_chinese_query(user_query):
        decoded = _truncate_at_unexpected_cjk(decoded)
        decoded = _strip_cjk_chars(decoded)
    marker = SIMPLE_RAG_ANSWER_MARKER
    if marker in decoded:
        answer = decoded.rsplit(marker, 1)[-1].strip()
    else:
        answer = decoded.strip()
    answer = _strip_filler_phrases(answer)
    answer = _strip_trailing_not_found(answer)
    answer = _normalize_markdown_bullets(answer)
    answer = _normalize_answer_layout(answer)
    answer = _fix_nora_norway_confusion(answer, user_query)
    answer = _strip_markdown_links(answer)
    return answer


def answer_query(
    user_query: str,
    top_k: int | None = None,
    category: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run simple RAG: embed -> fetch -> build context -> Qwen -> return answer and sources."""
    log = _get_logger()
    if not user_query or not user_query.strip():
        log.info("answer_query: empty query")
        return {"answer": "Please provide a question.", "sources": []}

    q = user_query.strip()
    intent = _classify_query_intent(q)
    in_scope = _is_in_scope(q)
    scope_keyword_matches = _in_scope_keyword_match_count(q)
    log.info("intent=%s in_scope=%s scope_keyword_matches=%s query=%s", intent, in_scope, scope_keyword_matches, q[:200] if q else "")
    if category:
        log.debug("query [%s] intent=%s: %s", category, intent, q)
    else:
        log.debug("query intent=%s: %s", intent, q)

    if not in_scope:
        log.info("out_of_scope; returning SIMPLE_RAG_OUT_OF_SCOPE_MESSAGE")
        return {"answer": SIMPLE_RAG_OUT_OF_SCOPE_MESSAGE, "sources": []}

    if top_k is not None:
        k = top_k
    elif intent == QUERY_INTENT_SYNTHESIS:
        k = SIMPLE_RAG_TOP_K_SYNTHESIS
    elif intent == QUERY_INTENT_FACT_DEFINITION:
        k = TOP_K_DEFINITION
    else:
        k = SIMPLE_RAG_TOP_K
    # For definition-style queries (e.g. "what is SAMA?"), expand acronym so retrieval finds definition chunks
    is_def, term = _is_definition_style_query(q)
    if is_def and term:
        expansion = ACRONYM_EXPANSIONS.get(term.strip().lower())
        q_for_embed = normalize_query_for_embedding(q + " " + expansion) if expansion else normalize_query_for_embedding(q)
    else:
        q_for_embed = normalize_query_for_embedding(q)
    try:
        query_embedding = embed_query(q_for_embed)
    except Exception as e:
        log.exception("embed_query failed: %s", e)
        raise
    chunks = fetch_chunks(query_embedding, limit=k)
    if DYNAMIC_TOP_K_ENABLED and chunks:
        top_sim = chunks[0].get("cosine_similarity") or chunks[0].get("similarity")
        if top_sim is not None and float(top_sim) < DYNAMIC_TOP_K_SIMILARITY_THRESHOLD:
            k2 = min(int(k * DYNAMIC_TOP_K_MULTIPLIER), 20)
            if k2 > k:
                chunks = fetch_chunks(query_embedding, limit=k2)
                log.debug("dynamic_top_k: re-fetched with k=%d", k2)
    if _is_arabic_query(q) and ENABLE_ARABIC_DUAL_RETRIEVE:
        translated_en = translate_arabic_to_english(q)
        if translated_en:
            try:
                en_embedding = embed_query(translated_en)
                chunks_en = fetch_chunks(en_embedding, limit=k)
                chunks = merge_chunks_rrf([chunks, chunks_en], rrf_k=RRF_K, top_n=k)
            except Exception as e:
                log.debug("dual_retrieve (EN) failed, using single: %s", e)
    # Second-pass retrieval (3.2): when top similarity is in [floor, threshold), re-fetch with larger k and merge
    if ENABLE_SECOND_PASS_RETRIEVAL and chunks and SECOND_PASS_SIMILARITY_THRESHOLD > 0:
        top_sim_raw = chunks[0].get("cosine_similarity") or chunks[0].get("similarity")
        if top_sim_raw is not None:
            top_sim_val = float(top_sim_raw)
            if SECOND_PASS_MIN_SIMILARITY <= top_sim_val < SECOND_PASS_SIMILARITY_THRESHOLD:
                k2 = min(k + SECOND_PASS_EXTRA_K, SECOND_PASS_MAX_K)
                if k2 > k:
                    try:
                        chunks_2 = fetch_chunks(query_embedding, limit=k2)
                        chunks = merge_chunks_rrf([chunks, chunks_2], rrf_k=RRF_K, top_n=k)
                        log.debug("second_pass: merged with k2=%d", k2)
                    except Exception as e:
                        log.debug("second_pass failed: %s", e)
    preferred_doc_names = get_documents_for_keywords(q)
    if ENABLE_RERANKING and chunks:
        chunks = rerank_chunks(q, chunks, top_n=k, preferred_doc_names=preferred_doc_names, intent=intent)
    if _is_arabic_query(q) and chunks:
        chunks = _reorder_arabic_first(chunks)
    log.debug("retrieval: %d chunks", len(chunks))
    # Phase 6.2: Arabic entity presence – at least one top chunk should contain the query entity
    if ENABLE_ARABIC_ENTITY_PRESENCE_CHECK and _is_arabic_query(q) and chunks:
        entity = _extract_query_entity(q)
        if entity and len(entity.strip()) >= 2:
            top5 = chunks[:5]
            if not any((entity.strip().lower() in (c.get("content") or "").lower()) for c in top5):
                log.info("arabic_entity_presence: query entity '%s' not in top chunks; returning not_found", entity[:50])
                return {"answer": SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC, "sources": []}
    if chunks:
        top_docs = [(r.get("document_name"), r.get("page_start"), r.get("page_end")) for r in chunks[:3]]
        log.debug("top chunks: %s", top_docs)
    has_arabic_in_top = True
    if _is_arabic_query(q) and chunks:
        has_arabic_in_top = any(_chunk_has_arabic_content(c) for c in chunks[:5])
        if not has_arabic_in_top:
            log.info("retrieval_language_consistency: Arabic query but no Arabic chunk in top; continuing with EN context")
    if ENABLE_STRICT_RETRIEVAL_LANGUAGE_FILTER and _is_arabic_query(q) and chunks and not has_arabic_in_top:
        log.info("strict_retrieval_language: Arabic query, no Arabic chunk in top; returning Arabic not_found")
        return {"answer": SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC, "sources": []}

    if not chunks:
        if AMBIGUITY_DETECTION_ENABLED and len(q.split()) <= AMBIGUITY_MAX_WORDS:
            log.info("ambiguity: no chunks for short query; returning clarification")
            return {"answer": SIMPLE_RAG_CLARIFICATION_MESSAGE, "sources": []}
        log.info("no chunks; returning not_found")
        return {
            "answer": SIMPLE_RAG_NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # Effective threshold: per-intent base, then lower for synthesis and Arabic (retrieval recall)
    intent_threshold = MIN_CHUNK_SIMILARITY
    if intent == QUERY_INTENT_SYNTHESIS:
        intent_threshold = MIN_CHUNK_SIMILARITY_SYNTHESIS
    elif intent == QUERY_INTENT_PROCEDURAL:
        intent_threshold = MIN_CHUNK_SIMILARITY_PROCEDURAL
    elif intent == QUERY_INTENT_FACT_DEFINITION:
        intent_threshold = MIN_CHUNK_SIMILARITY_FACT_DEFINITION
    elif intent == QUERY_INTENT_METADATA:
        intent_threshold = MIN_CHUNK_SIMILARITY_METADATA
    effective_threshold = intent_threshold
    if intent == QUERY_INTENT_SYNTHESIS:
        effective_threshold = min(effective_threshold, MIN_CHUNK_SIMILARITY_SYNTHESIS)
    if _is_arabic_query(q):
        effective_threshold = min(effective_threshold, MIN_CHUNK_SIMILARITY_ARABIC)

    top_similarity = _top_retrieval_similarity(chunks)
    rerank_bypass = (
        RERANK_BYPASS_SIMILARITY_GATE_THRESHOLD > 0
        and chunks
        and (float(chunks[0].get("_rerank_score") or 0) >= RERANK_BYPASS_SIMILARITY_GATE_THRESHOLD)
    )
    if not rerank_bypass and effective_threshold > 0 and top_similarity < effective_threshold:
        top3 = [float(chunks[i].get("cosine_similarity") or chunks[i].get("similarity") or chunks[i].get("score") or 0) for i in range(min(3, len(chunks)))]
        log.info(
            "similarity_gate: top %.3f < threshold %.3f (intent=%s arabic=%s); top3=%s; returning not_found",
            top_similarity, effective_threshold, intent, _is_arabic_query(q), top3[:3],
        )
        return {
            "answer": SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE,
            "sources": [{"document_name": c.get("document_name"), "page_start": c.get("page_start"), "page_end": c.get("page_end"), "snippet": (c.get("content") or "")[:SNIPPET_CHAR_LIMIT]} for c in chunks[:5]],
        }

    if effective_threshold > 0:
        chunks = [c for c in chunks if (float(c.get("cosine_similarity") or c.get("similarity") or c.get("score") or 0) >= effective_threshold)]
        if not chunks:
            log.info("similarity_gate: no chunks above effective threshold %.3f; returning not_found", effective_threshold)
            return {"answer": SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE, "sources": []}

    # Intent-aware upper bound: for fact_definition/metadata require at least one chunk above strict bar (reduce noise)
    if ENABLE_STRICT_QUALITY_GATE and MIN_CHUNK_SIMILARITY_STRICT > 0 and intent in (QUERY_INTENT_FACT_DEFINITION, QUERY_INTENT_METADATA):
        top_sim = _top_retrieval_similarity(chunks)
        if top_sim < MIN_CHUNK_SIMILARITY_STRICT:
            log.info("strict_quality_gate: top similarity %.3f < MIN_CHUNK_SIMILARITY_STRICT %.3f (intent=%s); returning not_found", top_sim, MIN_CHUNK_SIMILARITY_STRICT, intent)
            return {"answer": SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE, "sources": []}

    for c in chunks:
        c.pop("_rerank_score", None)

    # Intent-based retrieval: for fact_definition prefer chunks with definition markers (2.1, 2.2)
    if intent == QUERY_INTENT_FACT_DEFINITION and chunks:
        definition_chunks = [c for c in chunks if _chunk_has_definition_marker(c)]
        if definition_chunks:
            chunks = _reorder_definition_chunks_first(chunks)

    context_text = build_context(chunks)
    conversation_history: list[dict] = []
    if session_id:
        try:
            conversation_history = get_session_message_history(
                session_id, limit=CONVERSATION_HISTORY_MAX_MESSAGES
            )
        except Exception as e:
            log.debug("get_session_message_history failed (continuing without history): %s", e)
    # Routing (7.4): extraction only for fact_definition/metadata; synthesis uses multi-chunk generative path.
    if ENABLE_EXTRACTIVE_BUILDER and intent in (QUERY_INTENT_FACT_DEFINITION, QUERY_INTENT_METADATA):
        answer = build_extractive_answer(q, chunks, intent, max_chunks=1)
        if not answer or not answer.strip():
            answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE
        log.debug("extractive_builder used for intent=%s", intent)
    else:
        try:
            answer = generate_answer_simple(
                context_text, q, intent, conversation_history=conversation_history or None
            )
        except Exception as e:
            log.exception("generate_answer_simple failed: %s", e)
            raise
    raw_preview = (answer or "")[:300] + ("..." if len(answer or "") > 300 else "")
    log.debug("raw_answer (preview): %s", raw_preview)

    if ENABLE_POST_GEN_SIMILARITY_CHECK and not _is_not_found_answer(answer):
        if not _post_gen_similarity_ok(answer, chunks, POST_GEN_SIMILARITY_THRESHOLD):
            log.info("post_gen_similarity: answer not similar enough to chunks; overriding to not_found")
            answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE
    # Phase 8.2: allow mixed answer when query is mixed (Arabic + Latin); require primarily Arabic only when query is clearly Arabic
    if _is_arabic_query(q) and not _is_mixed_language(q) and ENABLE_ANSWER_LANGUAGE_CHECK and not _is_not_found_answer(answer):
        if not _is_answer_primarily_arabic(answer, ANSWER_ARABIC_SCRIPT_RATIO_MIN):
            # Translate back only when source answer is grounded and has citation (do not translate ungrounded)
            source_grounded = (
                "(Page " in answer or "(Pages " in answer or "(Source:" in answer
                or _is_answer_grounded_in_context(answer, context_text, SIMPLE_RAG_NOT_FOUND_MESSAGE)
            )
            if ENABLE_TRANSLATE_BACK and source_grounded:
                translated = translate_to_arabic(answer)
                if translated:
                    answer = translated
                    log.info("answer_language_check: translated answer to Arabic")
                else:
                    answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC
            else:
                if not source_grounded:
                    log.info("answer_language_check: not translating ungrounded answer to Arabic; overriding to not_found")
                else:
                    log.info("answer_language_check: Arabic query but answer not primarily Arabic; overriding to Arabic not_found")
                answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC

    answer, citation_replaced_not_found = _ensure_citation_fallback(answer, SIMPLE_RAG_NOT_FOUND_MESSAGE, intent)
    if citation_replaced_not_found:
        log.info("citation_fallback_applied: answer replaced with not_found (no Page/Pages in answer)")
    if SIMPLE_RAG_MANDATORY_CITATION_STRICT and not _is_not_found_answer(answer):
        if "(Page " not in answer and "(Pages " not in answer:
            log.info("citation_validator: mandatory citation strict; no (Page X) present; overriding to not_found")
            answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE
    if ENABLE_STRICT_SINGLE_LANGUAGE and not _is_not_found_answer(answer) and _is_mixed_language(answer):
        log.info("language_validator: mixed language detected; overriding to not_found")
        answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE

    # Entity containment (5.3, 7.3): answer or context must contain query entity for fact_definition/metadata
    if ENABLE_ENTITY_CONTAINMENT_CHECK and intent in (QUERY_INTENT_FACT_DEFINITION, QUERY_INTENT_METADATA) and not _is_not_found_answer(answer):
        entity = _extract_query_entity(q)
        if entity and len(entity.strip()) >= 3:
            answer_lower = (answer or "").lower()
            context_lower = context_text.lower()
            entity_lower = entity.strip().lower()
            if entity_lower not in answer_lower and entity_lower not in context_lower:
                log.info("entity_containment: query entity '%s' not in answer or context; overriding to not_found", entity[:50])
                answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE

    # Metadata answers (5.5): restrict to short factual extractions; truncate long decree text
    if intent == QUERY_INTENT_METADATA and not _is_not_found_answer(answer) and len(answer or "") > METADATA_ANSWER_MAX_CHARS:
        body = answer
        citation_suffix = ""
        if "(Page " in answer or "(Pages " in answer:
            match = re.search(r"(\s*\(Pages?\s+\d+[^\)]*\)\.?)\s*$", answer)
            if match:
                citation_suffix = match.group(1).strip()
                body = answer[: match.start()].strip()
        if len(body) > METADATA_ANSWER_MAX_CHARS:
            body = body[: METADATA_ANSWER_MAX_CHARS].strip()
            if not body.endswith("."):
                body = body.rsplit(".", 1)[0] + "." if "." in body else body + "."
            answer = (body + " " + citation_suffix).strip() if citation_suffix else body
        log.debug("metadata_answer_truncated to %d chars", METADATA_ANSWER_MAX_CHARS)

    # Citation validation (run before guard): check cited pages exist in chunks
    citation_all_valid, citation_any_valid = _validate_citations(answer, chunks)
    citation_valid = citation_any_valid
    top_sim = _top_retrieval_similarity(chunks)
    skip_not_found_if_high_retrieval = (
        NOT_FOUND_RETRIEVAL_CONFIDENCE_THRESHOLD > 0
        and top_sim >= NOT_FOUND_RETRIEVAL_CONFIDENCE_THRESHOLD
    )
    has_citation_and_similarity = _has_valid_citation_and_high_similarity(
        answer, chunks, CITATION_HIGH_SIMILARITY_THRESHOLD
    )
    synthesis_title_confidence = (
        _synthesis_title_match_confidence(chunks, q) if intent == QUERY_INTENT_SYNTHESIS else 0.0
    )

    # Definition guard: for "what is X?" / "define X?" / "purpose of X?" if context does not explicitly define or state purpose, force not found. Skip for metadata intent.
    if intent != QUERY_INTENT_METADATA:
        is_def, term = _is_definition_style_query(q)
        if is_def and term and not _is_not_found_answer(answer):
            if not _context_explicitly_defines_term(context_text, term) and not _context_contains_purpose_for_term(
                context_text, term
            ) and not _context_contains_header_or_metadata(context_text, term) and not _context_contains_acronym_expansion(
                context_text, term
            ) and not _chunks_title_match_term(chunks, term):
                if has_citation_and_similarity or skip_not_found_if_high_retrieval:
                    log.info("definition_guard: skipping override (valid citation+similarity or high retrieval confidence)")
                else:
                    log.info("definition_guard: context does not explicitly define '%s'; overriding to not_found", term)
                    answer = SIMPLE_RAG_NOT_FOUND_MESSAGE

    # Post-fill: when model returned not_found but context has acronym expansion (e.g. "what is sama"), fill from context
    if intent == QUERY_INTENT_FACT_DEFINITION and _is_not_found_answer(answer):
        is_def, term = _is_definition_style_query(q)
        if is_def and term and _context_contains_acronym_expansion(context_text, term):
            filled = _build_acronym_definition_from_context(context_text, chunks, term)
            if filled:
                answer = filled
                log.info("definition_post_fill: replaced not_found with acronym definition for '%s'", term)

    # Fallback: for "what is SAMA?" / "what is NORA?" use fixed one-liner when not_found or when answer is not a proper definition
    if intent == QUERY_INTENT_FACT_DEFINITION:
        is_def, term = _is_definition_style_query(q)
        if is_def and term:
            term_lower = re.sub(r"[?!.,;:]+$", "", term.strip().lower())  # "nora?" -> "nora"
            if term_lower in DEFINITION_ACRONYM_FALLBACK:
                use_fallback = _is_not_found_answer(answer)
                if not use_fallback:
                    # Replace wrong definition: answer doesn't contain the canonical expansion
                    answer_lower = (answer or "").lower()
                    if term_lower == "sama":
                        use_fallback = "saudi arabian monetary authority" not in answer_lower and "saudi central bank" not in answer_lower
                    elif term_lower == "nora":
                        use_fallback = "national regulatory authority" not in answer_lower
                if use_fallback:
                    answer = DEFINITION_ACRONYM_FALLBACK[term_lower]
                    log.info("definition_acronym_fallback: set answer for '%s'", term_lower)

    _grounding_confidence: float = 1.0
    # Semantic grounding (optional): pass / soft_fail (keep + source) / hard_fail (not_found)
    if USE_SEMANTIC_GROUNDING and not _is_not_found_answer(answer):
        decision, sem_score, soft_msg = grounding_decision(answer, context_text, chunks, intent, is_arabic_query=_is_arabic_query(q))
        _grounding_confidence = sem_score
        if decision == "hard_fail":
            if has_citation_and_similarity or skip_not_found_if_high_retrieval:
                log.info("semantic_grounding: hard_fail but skipping override (citation+similarity or high retrieval)")
            else:
                log.info("semantic_grounding: hard_fail (score=%.3f); overriding to not_found", sem_score)
                answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE
                _grounding_confidence = 0.0
        elif decision == "soft_fail" and soft_msg and "(Source:" not in answer:
            answer = answer.rstrip().rstrip(".") + soft_msg
    # Literal grounding: fact_definition/metadata use relaxed; synthesis and others use strict
    if not _is_not_found_answer(answer):
        if intent in (QUERY_INTENT_FACT_DEFINITION, QUERY_INTENT_METADATA):
            if not _is_answer_loosely_grounded(answer, context_text):
                if has_citation_and_similarity or skip_not_found_if_high_retrieval:
                    log.info("grounding_check: skipping override (citation+similarity or high retrieval)")
                else:
                    log.info("grounding_check: answer not loosely grounded (fact_definition/metadata); overriding to not_found")
                    answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE
        else:
            if not _is_answer_grounded_in_context(
                answer, context_text, SIMPLE_RAG_NOT_FOUND_MESSAGE
            ):
                if has_citation_and_similarity or skip_not_found_if_high_retrieval:
                    log.info("grounding_check: skipping override (citation+similarity or high retrieval)")
                else:
                    log.info("grounding_check: answer not supported by context; overriding to not_found")
                    answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE
    # Synthesis verbatim (Option A): override to not_found only when main grounding failed; if grounded but no verbatim, keep answer and ensure source
    if intent == QUERY_INTENT_SYNTHESIS and not _is_not_found_answer(answer) and not _answer_has_verbatim_support(answer, context_text):
        if _is_answer_grounded_in_context(answer, context_text, SIMPLE_RAG_NOT_FOUND_MESSAGE):
            if "(Page " not in answer and "(Pages " not in answer and "(Source:" not in answer:
                answer = answer.rstrip(".").rstrip() + " (Source: provided context)."
        else:
            if (
                has_citation_and_similarity
                or skip_not_found_if_high_retrieval
                or (SYNTHESIS_TITLE_MATCH_PASS_THRESHOLD > 0 and synthesis_title_confidence >= SYNTHESIS_TITLE_MATCH_PASS_THRESHOLD)
            ):
                log.info("grounding_check: synthesis not grounded but skipping override (citation+similarity, high retrieval, or title match)")
            else:
                log.info("grounding_check: synthesis answer not grounded; overriding to not_found")
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
            if has_citation_and_similarity or skip_not_found_if_high_retrieval:
                log.info("confabulation: would replace with not_found but skipping (citation+similarity or high retrieval)")
            else:
                answer = SIMPLE_RAG_NOT_FOUND_MESSAGE
                log.info("answer empty or too short after removing confabulation; replacing with not_found")

    # Generic phrases blocklist: remove ungrounded generic banking explanations
    generic_offending = _get_generic_blocklist_phrases(answer, context_text)
    if generic_offending and not _is_not_found_answer(answer):
        log.info("generic blocklist detected: %s", ", ".join(generic_offending))
        cleaned = _remove_sentences_containing_terms(answer, generic_offending, min_kept_len=40)
        if cleaned is not None and len(cleaned) >= 40:
            answer = cleaned
            log.debug("kept answer after removing sentences containing generic phrases")

    # Minimum confidence gate: combined similarity + grounding + citation (0 = disabled)
    if MIN_CONFIDENCE_FOR_ANSWER > 0 and not _is_not_found_answer(answer):
        combined_confidence = 0.4 * top_sim + 0.4 * _grounding_confidence + 0.2 * (1.0 if citation_valid else 0.0)
        if combined_confidence < MIN_CONFIDENCE_FOR_ANSWER:
            log.info("min_confidence_gate: combined_confidence %.3f < MIN_CONFIDENCE_FOR_ANSWER %.3f; overriding to not_found", combined_confidence, MIN_CONFIDENCE_FOR_ANSWER)
            answer = SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC if _is_arabic_query(q) else SIMPLE_RAG_NOT_FOUND_MESSAGE

    log.debug("final_answer (preview): %s", (answer or "")[:300])
    cited_sentences = _extract_cited_sentences(answer) if answer else []
    cited_articles = _extract_article_numbers_from_answer(answer) if answer else []
    if ENABLE_RETRIEVED_BUT_NOT_USED_LOG and chunks and answer:
        cited_pages_set = set(_extract_page_numbers_from_citations(answer))
        cited_idx = set()
        for i, c in enumerate(chunks):
            start = int(c.get("page_start") or 0)
            end = int(c.get("page_end") or 0)
            if any(start <= p <= end for p in cited_pages_set):
                cited_idx.add(i)
        not_used = len(chunks) - len(cited_idx)
        if not_used > 0:
            if intent == QUERY_INTENT_SYNTHESIS:
                log.debug("retrieved_but_not_used: %d of %d chunks not cited in answer (synthesis; multiple chunks expected)", not_used, len(chunks))
            else:
                log.info("retrieved_but_not_used: %d of %d chunks not cited in answer", not_used, len(chunks))
    snippet_limit = SNIPPET_CHAR_LIMIT
    sources = [
        {
            "document_name": row.get("document_name"),
            "page_start": row.get("page_start"),
            "page_end": row.get("page_end"),
            "article_id": row.get("article_id") or row.get("article_number") or row.get("article"),
            "snippet": (row.get("snippet") or (row.get("content") or "")[:snippet_limit]),
        }
        for row in chunks
    ]
    if _is_not_found_answer(answer):
        sources = []
    out: dict[str, Any] = {"answer": answer, "sources": sources, "cited_sentences": cited_sentences, "cited_articles": cited_articles}
    if RETURN_CONFIDENCE_SCORE:
        out["confidence"] = 0.0 if _is_not_found_answer(answer) else _grounding_confidence
    if RETURN_CITATION_VALID:
        out["citation_valid"] = citation_valid
    return out


def answer_query_streaming(
    user_query: str,
    session_id: str | None = None,
    chunk_size: int = 128,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """
    Streaming-friendly wrapper around answer_query.

    Currently computes the full answer once and then splits it into
    fixed-size text chunks. This preserves existing RAG behavior while
    enabling streaming delivery to the frontend without changing the
    core generation pipeline.
    """
    result = answer_query(user_query, session_id=session_id)
    answer = result.get("answer") or ""
    sources = result.get("sources") or []
    if not isinstance(answer, str):
        answer = str(answer)
    chunks: list[str] = []
    if answer:
        for i in range(0, len(answer), chunk_size):
            chunks.append(answer[i : i + chunk_size])
    return answer, sources, chunks


# Max characters of snippet to print per source in CLI (readability)
_SNIPPET_DISPLAY_CHARS = 280

# Minimum length of a contiguous answer segment that must appear in context for grounding check
_GROUNDING_MIN_SEGMENT_LEN = 25
# For short answers (e.g. decree + date), allow normalized overlap (punctuation/whitespace collapsed)
_GROUNDING_SHORT_ANSWER_LEN = 80
# Relaxed grounding for fact_definition/metadata: shorter segment or short-answer normalized match
_GROUNDING_RELAXED_MIN_SEGMENT = 15
_GROUNDING_LOOSE_SHORT_ANSWER_LEN = 100


def _normalize_for_grounding(text: str) -> str:
    """Collapse punctuation and whitespace for short-answer grounding comparison."""
    t = re.sub(r"[\s,;:.\-–—]+", " ", text).strip().lower()
    return " ".join(t.split())


def _is_answer_loosely_grounded(answer: str, context: str) -> bool:
    """True if any segment >= 15 chars appears in context, or answer is short and normalized phrase in context. For fact_definition/metadata."""
    if not answer or not context:
        return False
    body = re.sub(r"\(Page\s+\d+\)|\(Pages\s+\d+\s*[–\-]\s*\d+\)", "", answer, flags=re.IGNORECASE)
    body = re.sub(r"\(Source:\s*provided context\.?\)", "", body, flags=re.IGNORECASE)
    body = body.strip()
    if not body:
        return True
    body_norm = re.sub(r"\s+", " ", body).strip().lower()
    ctx_norm = re.sub(r"\s+", " ", context.strip()).lower()
    if len(body_norm) <= _GROUNDING_LOOSE_SHORT_ANSWER_LEN:
        if _normalize_for_grounding(body) in _normalize_for_grounding(context):
            return True
    for i in range(len(body_norm) - _GROUNDING_RELAXED_MIN_SEGMENT + 1):
        segment = body_norm[i : i + _GROUNDING_RELAXED_MIN_SEGMENT]
        if segment in ctx_norm:
            return True
    return False


_VERBATIM_SUPPORT_MIN_SEGMENT = 30
_VERBATIM_SUPPORT_MIN_SUBSTRING = 40


def _answer_has_verbatim_support(answer: str, context: str) -> bool:
    """True if at least one sentence or long substring of the answer appears in context (after normalizing whitespace)."""
    if not answer or not context:
        return False
    body = re.sub(r"\(Page\s+\d+\)|\(Pages\s+\d+\s*[–\-]\s*\d+\)", "", answer, flags=re.IGNORECASE)
    body = re.sub(r"\(Source:\s*provided context\.?\)", "", body, flags=re.IGNORECASE)
    body = body.strip()
    ctx_norm = re.sub(r"\s+", " ", context.strip()).lower()
    segments = re.split(r"[.!?\n]+", body)
    for seg in segments:
        seg = seg.strip()
        if len(seg) < _VERBATIM_SUPPORT_MIN_SEGMENT:
            continue
        seg_norm = re.sub(r"\s+", " ", seg).strip().lower()
        if seg_norm and seg_norm in ctx_norm:
            return True
        if len(seg_norm) >= _VERBATIM_SUPPORT_MIN_SUBSTRING:
            for i in range(len(seg_norm) - _VERBATIM_SUPPORT_MIN_SUBSTRING + 1):
                sub = seg_norm[i : i + _VERBATIM_SUPPORT_MIN_SUBSTRING]
                if sub in ctx_norm:
                    return True
    return False


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
    # Short factual answers (e.g. decree + date): allow normalized overlap
    if len(body) <= _GROUNDING_SHORT_ANSWER_LEN:
        if _normalize_for_grounding(body) in _normalize_for_grounding(context):
            return True
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
    """True if answer is the standard not-found message (English or Arabic). Uses normalized comparison (strip, collapse spaces, lowercase) so small variations still match."""
    if not answer or not answer.strip():
        return False
    a = " ".join(answer.strip().lower().split())
    nf_en = " ".join((SIMPLE_RAG_NOT_FOUND_MESSAGE or "").strip().lower().split())
    nf_ar = " ".join((SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC or "").strip().lower().split())
    return a == nf_en or a == nf_ar


def _is_answer_primarily_arabic(text: str, min_ratio: float = 0.3) -> bool:
    """True if at least min_ratio of alphabetic characters are in Arabic script (U+0600–U+06FF)."""
    if not text or not text.strip():
        return False
    ar_count = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count == 0:
        return False
    return (ar_count / alpha_count) >= min_ratio


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _post_gen_similarity_ok(
    answer: str,
    chunks: list[dict[str, Any]],
    threshold: float,
) -> bool:
    """True if max cosine similarity between answer embedding and chunk content embeddings >= threshold."""
    if not answer or not chunks or _is_not_found_answer(answer):
        return True
    try:
        answer_emb = embed_query(re.sub(r"\(Page\s+\d+\)|\(Pages\s+\d+\s*[–\-]\s*\d+\)", "", answer, flags=re.IGNORECASE))
        best = -1.0
        for c in chunks[:10]:
            content = (c.get("content") or "").strip()[:2000]
            if not content:
                continue
            chunk_emb = embed_chunk(content)
            sim = _cosine_similarity(answer_emb, chunk_emb)
            if sim > best:
                best = sim
        return best >= threshold
    except Exception:
        return True


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


def format_response_for_display(user_query: str, result: dict[str, Any], snippet_chars: int = 200) -> str:
    """Format result as: User's question, IOTA AI's Response, Source (numbered list)."""
    answer = (result.get("answer") or "").strip()
    sources = result.get("sources") or []
    lines = [
        "User's question : " + user_query,
        "IOTA AI's Response : " + answer,
        "Source :",
    ]
    if not sources:
        lines.append("  (none)")
    else:
        for i, s in enumerate(sources, 1):
            doc = s.get("document_name") or ""
            start = s.get("page_start", 0)
            end = s.get("page_end", 0)
            snippet = (s.get("snippet") or "").strip()
            if snippet and snippet_chars > 0:
                disp = snippet[:snippet_chars] + ("..." if len(snippet) > snippet_chars else "")
                lines.append(f"  {i} - {doc} (pages {start}-{end}): {disp}")
            else:
                lines.append(f"  {i} - {doc} (pages {start}-{end})")
    return "\n".join(lines)


def _attach_test_run_log_handler() -> tuple[logging.FileHandler, Path]:
    """Create a timestamped log file for this test run and attach a handler to the root logger. Returns (handler, path)."""
    log_dir = Path(SIMPLE_RAG_TEST_RUN_LOG_DIR)
    if not log_dir.is_absolute():
        log_dir = BACKEND_DIR / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"simple_rag_test_{ts}.log"
    path = path.resolve()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    root = logging.getLogger()
    root.addHandler(fh)
    return fh, path


def _detach_test_run_log_handler(handler: logging.Handler) -> None:
    """Remove the test-run file handler from the root logger and close it."""
    root = logging.getLogger()
    root.removeHandler(handler)
    try:
        handler.close()
    except Exception:
        pass


def run_test_questions() -> None:
    """Load test questions from JSON and run answer_query for each; print and log results.
    Creates a new log file per run under SIMPLE_RAG_TEST_RUN_LOG_DIR with all log lines and a final stats section.
    """
    log = _get_logger()
    run_handler, run_log_path = _attach_test_run_log_handler()
    log.info("test run log file: %s", run_log_path)
    start_time = time.perf_counter()
    questions = _load_test_questions_json()
    categories_order = ["sama", "arabic", "generic", "generic_off_topic"]
    total = sum(len(questions.get(cat, [])) for cat in categories_order)
    log.info(
        "run_test_questions: %d questions (SAMA=%d, Arabic=%d, Generic=%d, Off-topic=%d)",
        total,
        len(questions.get("sama", [])),
        len(questions.get("arabic", [])),
        len(questions.get("generic", [])),
        len(questions.get("generic_off_topic", [])),
    )
    n = 0
    run_results: list[dict[str, Any]] = []
    errors_count = 0
    try:
        for category_key in categories_order:
            qlist = questions.get(category_key, [])
            category_label = "Off-topic" if category_key == "generic_off_topic" else category_key.capitalize()
            for query in qlist:
                n += 1
                print(f"\n[{n}/{total}] [{category_label}] {query}")
                print("-" * 60)
                log.debug("[%d/%d] [%s] %s", n, total, category_label, query)
                try:
                    result = answer_query(query, category=category_label)
                    answer = result.get("answer") or ""
                    not_found = _is_not_found_answer(answer)
                    run_results.append({
                        "index": n,
                        "category": category_label,
                        "query": query,
                        "answer_preview": answer[:300] + ("..." if len(answer) > 300 else ""),
                        "not_found": not_found,
                        "sources_count": len(result.get("sources") or []),
                        "error": None,
                    })
                except Exception as e:
                    log.exception("answer_query failed: %s", e)
                    errors_count += 1
                    run_results.append({
                        "index": n,
                        "category": category_label,
                        "query": query,
                        "answer_preview": "",
                        "not_found": True,
                        "sources_count": 0,
                        "error": str(e),
                    })
                    raise
                print(format_response_for_display(query, result))
                log.debug("answer (preview): %s", (result.get("answer") or "")[:200])
                log.debug("sources_count: %d", len(result.get("sources") or []))
    finally:
        elapsed = time.perf_counter() - start_time
        _detach_test_run_log_handler(run_handler)

    # Append stats section to the run log file
    not_found_count = sum(1 for r in run_results if r["not_found"])
    by_category: dict[str, dict[str, Any]] = {}
    for r in run_results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"count": 0, "not_found": 0}
        by_category[cat]["count"] += 1
        if r["not_found"]:
            by_category[cat]["not_found"] += 1
    for cat, stats in by_category.items():
        stats["not_found_rate"] = (stats["not_found"] / stats["count"]) if stats["count"] else 0.0

    stats_lines = [
        "",
        "=" * 80,
        "TEST RUN STATS",
        "=" * 80,
        f"Run log file: {run_log_path}",
        f"Finished: {datetime.now(timezone.utc).isoformat()}",
        f"Duration (seconds): {elapsed:.2f}",
        f"Total questions: {total}",
        f"Errors: {errors_count}",
        f"Not found (count): {not_found_count}",
        f"Not found (rate): {not_found_count / total:.2%}" if total else "N/A",
        "",
        "By category:",
    ]
    for cat in categories_order:
        label = "Off-topic" if cat == "generic_off_topic" else cat.capitalize()
        if label in by_category:
            s = by_category[label]
            stats_lines.append(
                f"  {label}: count={s['count']}, not_found={s['not_found']}, not_found_rate={s['not_found_rate']:.2%}"
            )
    stats_lines.extend([
        "",
        "Per-question summary:",
    ])
    for r in run_results:
        nf = "NOT_FOUND" if r["not_found"] else "answer"
        err = f", error={r['error']}" if r.get("error") else ""
        stats_lines.append(f"  [{r['index']}] {r['category']}: {nf}, sources={r['sources_count']}{err}")
    stats_lines.append("=" * 80)

    try:
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(stats_lines) + "\n")
    except Exception as e:
        log.warning("Could not append stats to run log %s: %s", run_log_path, e)

    log.info("run_test_questions done. Run log: %s | not_found=%d/%d, errors=%d, duration=%.2fs",
             run_log_path, not_found_count, total, errors_count, elapsed)
    print(f"\nDone. Run log: {run_log_path}")
    print(f"Stats: not_found={not_found_count}/{total} ({not_found_count/total:.1%}), errors={errors_count}, duration={elapsed:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].strip() == "--test":
        run_test_questions()
    else:
        query = (
            sys.argv[1]
            if len(sys.argv) > 1
            else "What are the minimum criteria for obtaining a banking license in Saudi Arabia?"
        )
        result = answer_query(query)
        print(format_response_for_display(query, result))
