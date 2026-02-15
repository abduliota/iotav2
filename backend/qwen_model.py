"""Qwen 1.8B Instruct generation layer (Stage 4).

Loads Qwen1.5-1.8B-Chat (or QWEN_MODEL from config) in 4-bit and exposes a
deterministic generate_answer() function that follows the strict compliance
system prompt defined in Core/06_generation_layer.md.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import ALLOW_BASED_ON_CONTEXT_PROMPT_LINE, APP_BRAND_NAME, QWEN_MODEL

# HF token for gated models (set HF_TOKEN or HUGGING_FACE_HUB_TOKEN)
_HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None


def _load_qwen() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Lazy-load Qwen 1.8B Instruct model and tokenizer in 4-bit on single GPU (persistent)."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL,
        trust_remote_code=True,
        token=_HF_TOKEN,
    )

    # 4-bit quantization; 1.8B fits easily in ~6GB VRAM (e.g. RTX 4050 Laptop)
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for quantization.")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    # Load quantized model - workaround: patch PreTrainedModel.to() to skip for quantized models
    # bitsandbytes already places quantized models on GPU, so .to() calls should be no-ops
    import transformers.modeling_utils as modeling_utils
    original_to = modeling_utils.PreTrainedModel.to
    
    # Flag to track that we're loading a quantized model
    _loading_quantized = [True]  # Use list to allow modification in nested scope
    
    def patched_to(self, *args, **kwargs):
        # During quantized model loading, always skip .to() calls
        # This prevents errors when dispatch_model tries to call .to() before
        # quantization attributes are fully set
        if _loading_quantized[0]:
            return self
        
        # After loading, check if model is quantized using multiple indicators
        if (hasattr(self, 'hf_quantizer') and self.hf_quantizer is not None) or \
           (hasattr(self, 'quantization_config') and self.quantization_config is not None) or \
           (hasattr(self, '_hf_quantizer') and self._hf_quantizer is not None):
            return self
        
        # Check for BitsAndBytes modules as fallback indicator
        if hasattr(self, 'named_modules'):
            try:
                for name, module in self.named_modules():
                    module_type = str(type(module))
                    if 'BitsAndBytes' in module_type or 'bnb' in module_type.lower():
                        return self
            except Exception:
                pass  # If check fails, proceed with original .to()
        
        # Non-quantized model - use original .to()
        return original_to(self, *args, **kwargs)
    
    modeling_utils.PreTrainedModel.to = patched_to
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            trust_remote_code=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",  # Use device_map="auto" to handle quantization properly
            token=_HF_TOKEN,
        )
    finally:
        # Restore original .to() method and reset flag
        modeling_utils.PreTrainedModel.to = original_to
        _loading_quantized[0] = False

    # RoPE (cos/sin) and other buffers were created on CPU and never moved because
    # patched .to() is a no-op for quantized models. Move each module's buffers to
    # that module's parameter device so cos[position_ids] in Qwen2 doesn't crash.
    _move_buffers_to_module_device(model)

    _tokenizer = tokenizer
    _model = model
    return tokenizer, model


def _move_buffers_to_module_device(module: torch.nn.Module) -> None:
    """Move each submodule's buffers to the same device as that submodule's parameters.
    Modules with only buffers (e.g. Qwen2RotaryEmbedding) are moved to the root model's
    device. Fixes RoPE (cos_cached/sin_cached) left on CPU when PreTrainedModel.to()
    is patched during quantized load."""
    root_device = next(module.parameters()).device
    for m in module.modules():
        bufs = list(m.buffers(recurse=False))
        if not bufs:
            continue
        params = list(m.parameters(recurse=False))
        device = next(m.parameters(recurse=False)).device if params else root_device
        for buf in bufs:
            if buf.device != device:
                buf.data = buf.data.to(device)


def _build_system_prompt() -> str:
    """Return the strict system prompt for the compliance assistant."""
    return f"""
You are the official AI compliance assistant of {APP_BRAND_NAME}.

Your role is to provide accurate, regulation-based answers
strictly from SAMA and NORA documents.

You MUST follow these rules:

1. Use ONLY the provided context.
2. Do NOT use prior knowledge.
3. Do NOT infer beyond the text.
4. Do NOT speculate.
5. Do NOT fabricate missing details.
6. If the answer is not explicitly supported by the context,
   respond exactly:
   "Information not found in SAMA/NORA documents."
   Only use this when the context truly does not contain the answer.
   When the context contains the answer (e.g. decree number, definition,
   or clear statement), you MUST provide it and cite the pages.
7. If the question is outside SAMA/NORA scope,
   respond exactly:
   "This assistant only supports SAMA and NORA related queries."
8. Always cite page numbers.
9. Never mention internal system logic.
10. Never mention embeddings, Supabase, or retrieval mechanisms.

11. Only state information that is explicitly supported by the context.
12. Do NOT invent or list categories that are not clearly mentioned in the context.
13. Do NOT repeat the same item more than once in the answer.
14. If the context does not clearly list who is prohibited or eligible,
    respond exactly: "Information not found in SAMA/NORA documents."
15. When listing items, use at most 5 bullet points.
16. In this assistant, "SAMA" ALWAYS refers to the Saudi Central Bank
    (formerly the Saudi Arabian Monetary Authority) and to Saudi regulations,
    NOT to any organization or association in other countries.
17. Never interpret "SAMA" as referring to Singapore or any other non-Saudi entity.
18. Do NOT expand acronyms (including "SAMA" and "NORA") unless the exact
    expansion appears explicitly in the provided context.
19. If the expansion or definition of any acronym or term is not clearly present
    in the context, respond exactly: "Information not found in SAMA/NORA documents."
20. You MUST only refer to SAMA (Saudi Central Bank) and Saudi regulations.
    Do NOT mention regulators, central banks, or banks of other countries
    (e.g. UAE, Pakistan, Abu Dhabi, etc.). If the context does not clearly
    support a SAMA/Saudi-only answer, respond exactly:
    "Information not found in SAMA/NORA documents."
21. You MUST cite the source for your answer using the format (Pages X–Y) or
    (Page X) at the end of relevant statements. If the context does not allow
    you to cite specific pages, say so explicitly.
22. For numbers, percentages, or definitions (e.g. ratios, limits), state only
    what appears in the context. If the context does not give a specific number
    or definition, respond exactly: "Information not found in SAMA/NORA documents."
    Do not invent figures or definitions.
23. If the user's question is in Arabic, respond only in Arabic. If in English,
    respond only in English. Do not repeat this instruction in your response.
24. Do not repeat the same phrase, clause, or bullet point; never output the
    same sentence more than once. State each point once.
25. Base your answer only on the provided context. Do not add general knowledge.
    Prefer quoting or paraphrasing the context; if the context does not state
    something, do not state it.
26. When the context contains a direct answer (e.g. decree number, definition),
    quote or paraphrase it and cite the pages. Include specific names and
    numbers from the context. Do not say "no specific content" or "no specific
    content was given" if that information is present in the context.
27. Do NOT output URLs, hyperlinks, or say that you cannot access links.
    Answer only from the provided context text. Never mention external websites.

Citation format:

- Single page → (Page X)
- Page range → (Pages X–Y)

Citations must appear at the end of relevant statements.

Answer format:

- Start with a direct answer.
- Use bullet points if multiple conditions exist.
- End with page references.
- Be precise and formal.
""".strip()


def _is_arabic_query(text: str) -> bool:
    """Return True if the text is predominantly Arabic (by script)."""
    if not text or not text.strip():
        return False
    # Arabic Unicode: \u0600-\u06FF (Arabic), \u0750-\u077F (Arabic Supplement)
    arabic_count = sum(
        1 for c in text
        if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F"
    )
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters == 0:
        return False
    return arabic_count / total_letters >= 0.3


def _strip_leakage(text: str) -> str:
    """Remove prompt/instruction leakage from the start of the model output."""
    if not text or not text.strip():
        return text
    leakage_prefixes = [
        "the user's question is",
        "answer:",
        "the user's question is in english; respond in that language.",
        "the user's question is in arabic; respond in that language.",
        "you must answer this exact question:",
    ]
    out = text.strip()
    lower = out.lower()
    for prefix in leakage_prefixes:
        if lower.startswith(prefix):
            out = out[len(prefix) :].strip()
            lower = out.lower()
            break
    return out


def _strip_trailing_leakage(text: str) -> str:
    """Remove instruction/prompt leakage from the end of the model output (repeated markers, trailing prompts)."""
    if not text or not text.strip():
        return text
    trailing_phrases = [
        "your response (cite pages):",
        "your response (answer directly):",
        "please summarize the main purpose of the article below",
        "please write down the full title of the article where the information needs to be found",
        "please write down the full title of the article where the information is found",
    ]
    out = text.strip()
    while True:
        lower = out.lower()
        stripped_any = False
        for phrase in trailing_phrases:
            if lower.endswith(phrase):
                out = out[: -len(phrase)].strip()
                stripped_any = True
                break
            # Strip when phrase appears anywhere: remove from last occurrence onward
            if phrase in lower:
                idx = lower.rfind(phrase)
                out = out[:idx].strip()
                stripped_any = True
                break
            # Also strip a line that is exactly the phrase (with optional trailing whitespace)
            if "\n" in out:
                lines = out.split("\n")
                last_line = lines[-1].strip().lower()
                if last_line == phrase or last_line.rstrip(".:") == phrase.rstrip(".:"):
                    out = "\n".join(lines[:-1]).strip()
                    stripped_any = True
                    break
        if not stripped_any:
            break
    return out


def _keep_last_substantive_block(text: str) -> str:
    """If text has multiple blocks (e.g. separated by '---' or 'Your response'), return the last substantive one."""
    if not text or not text.strip():
        return text
    citation_pattern = re.compile(r"\(Pages?\s*\d+(\s*[–-]\s*\d+)?\)", re.IGNORECASE)
    short_no_info = 80  # "no information" style lines are short
    blocks = re.split(r"\n\s*---\s*\n|\n\nYour response\b", text, flags=re.IGNORECASE)
    blocks = [b.strip() for b in blocks if b.strip()]
    if len(blocks) <= 1:
        return text.strip()
    # Prefer last block that has citation or is long enough
    for b in reversed(blocks):
        if citation_pattern.search(b) or len(b) > short_no_info:
            return b
    return blocks[-1]


def _is_garbled_answer(text: str) -> bool:
    """Return True if the answer looks like garbage (repeated symbols, almost no words)."""
    if not text or not text.strip():
        return True
    s = text.strip()
    # Long run of same character (e.g. dashes, spaces, dots) -> garbled
    if len(s) > 50:
        for ch in ("-", ".", " ", "_", "="):
            run = ch * 30
            if run in s:
                return True
    # Meaningful content: letters (and digits) vs total non-space chars
    non_space = [c for c in s if not c.isspace()]
    if not non_space:
        return True
    letter_or_digit = sum(1 for c in non_space if c.isalnum() or c in ".,;:()–—''\"")
    ratio = letter_or_digit / len(non_space)
    if ratio < 0.25:
        return True
    # Very few words
    words = [w for w in s.split() if any(c.isalnum() for c in w)]
    if len(words) < 3 and len(s) > 40:
        return True
    return False


def generate_answer(context: str, user_query: str) -> str:
    """Generate a deterministic answer from Qwen 1.8B Instruct given context and query.

    Parameters (from Core/06_generation_layer.md):
    - temperature = 0
    - top_p = 0.1
    - do_sample = False
    - max_new_tokens = 350
    - repetition_penalty = 1.05
    """
    tokenizer, model = _load_qwen()

    system_prompt = _build_system_prompt()

    prompt = (
        f"{system_prompt}\n\n"
        "You must answer using ONLY the following context. Do not cite or mention any external URLs, "
        "websites, or sources outside this context. If the answer is in the context, state it and cite the document pages.\n\n"
        "Context:\n\n"
        f"{context}\n\n"
        "---\n\n"
        "User Question:\n"
        f"{user_query}\n\n"
        f"You must answer this exact question: {user_query}\n\n"
        "If the answer is not in the context above, respond exactly: Information not found in SAMA/NORA documents. "
        "Do not ask the user for more information or ask follow-up questions.\n\n"
        "If the context describes a related but different topic (e.g. what delays a process rather than what the minimum criteria are), "
        "do not answer as if the context supported the question. Respond that the context does not contain the requested information, "
        "or briefly state what the context does say and that it does not answer the question asked. Do not ask follow-up questions.\n\n"
        + (f"{ALLOW_BASED_ON_CONTEXT_PROMPT_LINE}\n\n" if ALLOW_BASED_ON_CONTEXT_PROMPT_LINE else "")
        + "---\n\n"
        "Your response (cite pages):\n"
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.2,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer using last occurrence of marker (avoids duplicate markers / wrong block)
    answer = decoded.strip()
    for marker in ("Your response (cite pages):", "Answer:"):
        if marker in decoded:
            answer = decoded.rsplit(marker, 1)[-1].strip()
            break

    answer = _strip_leakage(answer)
    answer = _strip_trailing_leakage(answer)
    # If multiple blocks remain (e.g. "---" or "\n\nYour response"), keep last substantive block
    answer = _keep_last_substantive_block(answer)
    # Replace garbled output (e.g. repeated dashes/symbols) with safe fallback
    if _is_garbled_answer(answer):
        answer = "Information not found in SAMA/NORA documents."
    stripped = answer.strip() if answer else ""
    if not stripped or len(stripped) < 30:
        low = stripped.lower()
        if "information not found" in low or "this assistant only supports" in low:
            pass
        else:
            answer = "Information not found in SAMA/NORA documents."

    # Fix repetition: detect and truncate repeated n-grams and sentences
    answer = _fix_repetition(answer)
    
    # Cap answer length (slightly higher for Arabic)
    max_chars = 1000 if _is_arabic_query(user_query) else 800
    answer = _cap_answer_length(answer, max_chars=max_chars)
    
    # Optional: Validate citations (if answer doesn't have citations and isn't a rejection message)
    answer = _validate_citations(answer)
    
    return answer


def _validate_citations(text: str) -> str:
    """Ensure answer has citations or is a valid rejection message.
    
    Args:
        text: Answer text
    
    Returns:
        Text with citations added if missing, or rejection message if invalid
    """
    if not text:
        return text
    
    # Check if it's a valid rejection message
    rejection_phrases = [
        "information not found in sama/nora documents",
        "this assistant only supports sama and nora related queries",
    ]
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in rejection_phrases):
        return text
    
    # Check for citation pattern: (Page X) or (Pages X–Y) or (Pages X-Y)
    citation_pattern = r"\(Pages?\s*\d+(\s*[–-]\s*\d+)?\)"
    if re.search(citation_pattern, text, re.IGNORECASE):
        return text
    
    # No citations found - append generic citation
    return text + " (Source: provided context.)"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (EN + Arabic punctuation)."""
    # Sentence endings: . ! ? and Arabic ؟ ۔
    pattern = re.compile(r"[.?!؟۔]\s*")
    parts = pattern.split(text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def _normalize_for_repeat(s: str) -> str:
    """Normalize string for repetition comparison (lowercase, collapse space, no punctuation)."""
    s = " ".join(s.split()).lower()
    return re.sub(r"[^\w\s]", "", s)


def _fix_repetition(text: str, ngram_size: int = 5, max_occurrences: int = 3) -> str:
    """Detect and truncate repeated n-grams and repeated sentences in the answer.
    
    Args:
        text: Answer text
        ngram_size: Size of n-gram to check for repetition
        max_occurrences: Maximum allowed consecutive occurrences
    
    Returns:
        Text truncated at first repeated phrase if repetition detected
    """
    if not text:
        return text

    # Sentence-level: truncate at first consecutive duplicate (normalized)
    sentences = _split_sentences(text)
    if len(sentences) >= 2:
        seen: list[str] = []
        for i, sent in enumerate(sentences):
            norm = _normalize_for_repeat(sent)
            if norm and norm in seen:
                return ". ".join(sentences[:i]).strip()
            if norm:
                seen.append(norm)

    # Overlap: same 8–12 word window appearing 3+ times anywhere
    words = text.split()
    window_size = min(10, max(8, len(words) // 3))
    if len(words) >= window_size * 3:
        for start in range(len(words) - window_size):
            window = tuple(words[start : start + window_size])
            norm_window = _normalize_for_repeat(" ".join(window))
            if not norm_window:
                continue
            count = 0
            pos = 0
            while pos <= len(words) - window_size:
                w = tuple(words[pos : pos + window_size])
                if _normalize_for_repeat(" ".join(w)) == norm_window:
                    count += 1
                    if count >= 3:
                        return " ".join(words[: start + window_size])
                pos += 1

    # Consecutive same-word run (e.g. "word word word")
    if len(words) >= 4:
        i = 0
        while i < len(words) - 3:
            w = words[i]
            run = 1
            for j in range(i + 1, len(words)):
                if words[j] == w:
                    run += 1
                else:
                    break
            if run >= 3:
                return " ".join(words[:i])
            i += 1

    if len(words) < ngram_size:
        return text

    # Smaller n-gram pass (3 and 4) then standard 5-gram
    for ng in (3, 4, ngram_size):
        if len(words) < ng * max_occurrences:
            continue
        for i in range(len(words) - ng * max_occurrences):
            ngram = tuple(words[i : i + ng])
            consecutive_count = 1
            for j in range(i + ng, len(words) - ng + 1, ng):
                next_ngram = tuple(words[j : j + ng])
                if next_ngram == ngram:
                    consecutive_count += 1
                    if consecutive_count >= max_occurrences:
                        return " ".join(words[: i + ng])
                else:
                    break

    return text


def _cap_answer_length(text: str, max_chars: int = 800) -> str:
    """Cap answer length, truncating at sentence boundary if possible.
    
    Args:
        text: Answer text
        max_chars: Maximum characters allowed
    
    Returns:
        Truncated text (at sentence boundary if possible)
    """
    if not text or len(text) <= max_chars:
        return text
    
    # Try to truncate at sentence boundary (EN + Arabic punctuation)
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    last_exclamation = truncated.rfind("!")
    last_question = truncated.rfind("?")
    last_question_ar = truncated.rfind("؟")
    last_question_ur = truncated.rfind("۔")
    
    last_sentence_end = max(
        last_period, last_exclamation, last_question,
        last_question_ar, last_question_ur,
        -1,
    )
    
    # Use sentence boundary if within reasonable range (relaxed from 0.7 to 0.5)
    if last_sentence_end > max_chars * 0.5:
        return truncated[:last_sentence_end + 1]
    
    # Otherwise truncate at word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.5:
        return truncated[:last_space] + "..."
    
    return truncated + "..."

