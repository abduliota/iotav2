from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from IOTAV3.backend import config_iotav3 as cfg
from IOTAV3.backend.cache.prompt_cache import (
    BasePromptCache,
    build_cache_payload,
    get_prompt_cache,
)
from IOTAV3.backend.cag.cached_knowledge import get_cag_text_block
from IOTAV3.backend.context.context_builder import build_context_payload
from IOTAV3.backend.context_engine.persona import (
    build_system_prompt,
    build_user_prompt,
)
from IOTAV3.backend.llm.client import generate as llm_generate
from IOTAV3.backend.retrieval.embeddings import embed_query
from IOTAV3.backend.retrieval.retriever import retrieve_chunks
from IOTAV3.backend.memory.memory import (  # type: ignore
    get_session_summary,
    get_user_profile,
)


AnswerResult = Dict[str, Any]


def _classify_intent(query: str) -> str:
    q = query.strip().lower()
    if q.startswith("what is ") or q.startswith("define "):
        return "definition"
    if q.startswith("how ") or q.startswith("why ") or q.startswith("compare "):
        return "synthesis"
    return "generic"


def _is_off_topic(query: str) -> bool:
    q = query.strip().lower()
    return any(pat in q for pat in cfg.OFF_TOPIC_PATTERNS)


def _is_in_scope(query: str) -> bool:
    q = query.strip().lower()
    return any(kw in q for kw in cfg.IN_SCOPE_KEYWORDS)


def _query_for_embedding(query: str) -> str:
    """
    Optionally expand definition-style queries before embedding so retrieval
    ranks definition chunks higher. Only affects the string passed to embed_query.
    """
    q = query.strip()
    if not q:
        return q
    q_lower = q.lower()
    term: Optional[str] = None
    if q_lower.startswith("what is "):
        rest = q[8:].strip()
        term = rest.split("?")[0].strip().rstrip("?!.,;:").lower()
    elif q_lower.startswith("what's "):
        rest = q[7:].strip()
        term = rest.split("?")[0].strip().rstrip("?!.,;:").lower()
    elif q_lower.startswith("define "):
        rest = q[7:].strip()
        term = rest.split("?")[0].strip().rstrip("?!.,;:").lower()
    elif q_lower.startswith("meaning of "):
        rest = q[11:].strip()
        term = rest.split("?")[0].strip().rstrip("?!.,;:").lower()
    if term and term in cfg.QUERY_EXPANSION_FOR_EMBEDDING:
        expansion = cfg.QUERY_EXPANSION_FOR_EMBEDDING[term]
        return (q + " " + expansion).strip()
    return q


def _build_doc_keys(chunks: List[dict]) -> List[Tuple[str, int, int]]:
    keys: List[Tuple[str, int, int]] = []
    for row in chunks:
        doc = str(row.get("document_name") or "").strip()
        page_start = int(row.get("page_start") or 0)
        page_end = int(row.get("page_end") or page_start)
        keys.append((doc, page_start, page_end))
    return keys


def answer_query_iotav3(
    query: str,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> AnswerResult:
    """
    End-to-end RAG pipeline for IOTAV3.

    This mirrors the contract of the legacy `simple_rag.answer_query` but
    uses the IOTAV3-local components (config, cache, context engine).
    """

    intent = _classify_intent(query)

    if _is_off_topic(query):
        return {
            "answer": cfg.OUT_OF_SCOPE_MESSAGE,
            "sources": [],
            "intent": intent,
            "in_scope": False,
        }

    # Keyword-based scope check; this may later be augmented by retrieval
    # evidence to form an effective in-scope flag.
    keyword_in_scope = _is_in_scope(query)

    # Cache handles are initialised lazily after retrieval so that the key
    # can incorporate document and memory context.
    cache: Optional[BasePromptCache] = None
    cache_key: Optional[str] = None

    # Retrieval: expand definition-style query for embedding; use larger top_k for definitions.
    query_for_embed = _query_for_embedding(query)
    query_embedding = embed_query(query_for_embed)
    top_k = cfg.TOP_K_DEFINITION if intent == "definition" else cfg.TOP_K
    chunks = retrieve_chunks(query_embedding, top_k=top_k)

    # Memory: use existing helpers to fetch high-level summaries.
    session_summary = None
    user_profile_summary = None
    if session_id is not None:
        summary = get_session_summary(session_id)
        if summary and isinstance(summary, dict):
            session_summary = summary.get("summary") or summary.get("content")
    if user_id is not None:
        profile = get_user_profile(user_id)
        if profile and isinstance(profile, dict):
            user_profile_summary = profile.get("summary") or profile.get("content")

    cag_block = get_cag_text_block({})

    context_payload = build_context_payload(
        query=query,
        chunks=chunks,
        session_summary=session_summary,
        user_profile_summary=user_profile_summary,
        cag_block=cag_block,
    )

    # Flatten context payload into text for the LLM.
    context_lines: List[str] = []
    if context_payload.user_profile_summary:
        context_lines.append(
            "USER PROFILE SUMMARY:\n" + context_payload.user_profile_summary.strip()
        )
    if context_payload.session_summary:
        context_lines.append(
            "SESSION SUMMARY:\n" + context_payload.session_summary.strip()
        )
    if context_payload.cag_block:
        context_lines.append(context_payload.cag_block.strip())
    if context_payload.passages:
        for idx, p in enumerate(context_payload.passages, start=1):
            context_lines.append(
                f"[Passage {idx}] Document: {p.document_name}, "
                f"Pages: {p.page_start}–{p.page_end}\nContent:\n{p.snippet}"
            )
    context_text = "\n\n".join(context_lines).strip()

    # Compute retrieval-derived signals for scoping and grounding.
    top_similarity = 0.0
    if context_payload.passages:
        top_similarity = max(p.similarity for p in context_payload.passages)
    has_passages = bool(context_payload.passages)
    has_usable_passage = any(
        p.similarity >= cfg.RETRIEVAL_MIN_CONTEXT_SIM for p in context_payload.passages
    )

    # Retrieval-first effective in-scope flag: either we saw explicit
    # scope keywords or the top retrieved passage is similar enough.
    effective_in_scope = bool(
        keyword_in_scope
        or top_similarity >= cfg.RETRIEVAL_IN_SCOPE_MIN_SIM
    )

    # Cache lookup based on the full logical context (query, intent,
    # effective scope flag, top documents, and coarse memory/profile
    # identifiers).
    cache = get_prompt_cache()
    if cache is not None:
        doc_keys = _build_doc_keys(chunks[: cfg.MAX_SOURCES])
        memory_keys: List[str] = []
        if session_id is not None:
            memory_keys.append(f"session:{session_id}")
        if user_id is not None:
            memory_keys.append(f"user:{user_id}")
        profile_fingerprint = user_id

        payload = build_cache_payload(
            query=query,
            intent=intent,
            in_scope=effective_in_scope,
            route=None,
            doc_keys=doc_keys,
            memory_keys=memory_keys,
            profile_fingerprint=profile_fingerprint,
        )
        cache_key = cache.build_key(payload)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    if (not has_passages or not context_text) and not effective_in_scope:
        # No useful context and retrieval does not suggest we are in the
        # domain -> treat as out-of-scope.
        answer_text = cfg.OUT_OF_SCOPE_MESSAGE
        sources: List[dict] = []
    elif not has_passages and effective_in_scope:
        # Clearly in-domain but retrieval returned no passages at all.
        answer_text = cfg.NOT_FOUND_MESSAGE
        sources = []
    elif not has_usable_passage and not effective_in_scope:
        # Retrieval returned only very low-similarity passages and we do
        # not consider the query in scope -> out-of-scope.
        answer_text = cfg.OUT_OF_SCOPE_MESSAGE
        sources = []
    elif not has_usable_passage and effective_in_scope:
        # We believe the query is in-domain but have no sufficiently
        # strong passages to ground an answer. For definition-style
        # questions, be strict and return NOT_FOUND; for more
        # synthesis-oriented questions, allow the model to answer from
        # the best available (but weak) context.
        if intent == "definition":
            answer_text = cfg.NOT_FOUND_MESSAGE
            sources = []
        else:
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(query=query, context_text=context_text)
            answer_text = llm_generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stream=on_chunk is not None,
                on_chunk=on_chunk,
            )
            top_passages = context_payload.passages[: cfg.MAX_SOURCES]
            sources = [
                {
                    "document_name": p.document_name,
                    "page_start": p.page_start,
                    "page_end": p.page_end,
                    "snippet": p.snippet,
                }
                for p in top_passages
            ]
    else:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(query=query, context_text=context_text)
        answer_text = llm_generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stream=on_chunk is not None,
            on_chunk=on_chunk,
        )
        # Simple source extraction: map top N passages to sources.
        top_passages = context_payload.passages[: cfg.MAX_SOURCES]
        sources = [
            {
                "document_name": p.document_name,
                "page_start": p.page_start,
                "page_end": p.page_end,
                "snippet": p.snippet,
            }
            for p in top_passages
        ]

    result: AnswerResult = {
        "answer": answer_text,
        "sources": sources,
        "intent": intent,
        "in_scope": effective_in_scope,
    }

    # Populate cache after generation.
    if cache is not None and cache_key is not None:
        try:
            cache.set(cache_key, result)
        except Exception:
            # Cache failures must never break query handling.
            pass

    return result

