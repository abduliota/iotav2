"""Stage 1 config: paths and thresholds (no secrets)."""
import os
from pathlib import Path


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, str(default)).lower()
    return v in ("1", "true", "yes")


def _load_list_from_file_or_env(
    file_env_key: str,
    list_env_key: str,
    default_list: list[str],
) -> list[str]:
    """Load list from file (one per line) or comma-separated env, else return default_list."""
    file_path = os.getenv(file_env_key, "").strip()
    if file_path:
        path = Path(file_path)
        if not path.is_absolute():
            path = BACKEND_DIR / path
        try:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    out = [line.strip() for line in f if line.strip()]
                if out:
                    return out
        except Exception:
            pass
    env_val = os.getenv(list_env_key, "").strip()
    if env_val:
        return [s.strip() for s in env_val.split(",") if s.strip()]
    return default_list


# Paths: backend dir and project root (parent of backend)
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

PDF_DIR = PROJECT_ROOT / "pdfs"
OUTPUT_DIR = BACKEND_DIR / "output" / "pages"
CHUNKS_OUTPUT_DIR = BACKEND_DIR / "output" / "chunks"
PAGES_INPUT_DIR = BACKEND_DIR / "output" / "pages"

# Use OCR when native text has fewer than this many characters
NATIVE_TEXT_MIN_LEN = 100

# OCR: enable and languages (PaddleOCR runs en + ar, picks best result per page)
USE_OCR = True
OCR_LANGS = ["en", "ar"]
OCR_DPI = 200  # render PDF page at this DPI for OCR

# Remove a header/footer line if it appears on more than this fraction of pages (0–1)
HEADER_FOOTER_FREQUENCY_THRESHOLD = 0.6

# Sentence-ending characters (don't merge line after these)
SENTENCE_END_CHARS = ".?!:؟؛"

# ---- Stage 2: Chunking (Qwen tokenizer) ----
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
BACKTRACK_TOKENS = 50  # max tokens to backtrack for sentence boundary
SHORT_PAGE_TOKEN_THRESHOLD = 300  # merge page with next if below this
LONG_SECTION_TOKEN_THRESHOLD = 2000  # split by paragraphs first if above
QWEN_MODEL = "Qwen/Qwen2-7B-Instruct"

# ---- Stage 3: Embeddings and Supabase ----
# OpenAI embeddings (read from env: OPENAI_API_KEY)
# Using text-embedding-3-small for 1536-dim vectors.
AZURE_EMBEDDING_MODEL = "text-embedding-3-small"
AZURE_EMBEDDING_DIMENSION = 1536

# Supabase (read from env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
# Batch size for chunk inserts
CHUNK_BATCH_SIZE = 200  # 100-300 recommended

# ---- Stage 4: Query, Retrieval, and Generation ----
# Similarity threshold for accepting retrieval (cosine similarity in [0, 1]).
# Lowered for experimentation on dense regulatory text; tune back up after measuring.
SIMILARITY_THRESHOLD = 0.5
# Slightly lower threshold for Arabic queries (often lower embedding similarity). For future use when match_chunks RPC supports min_similarity.
SIMILARITY_THRESHOLD_ARABIC = 0.45
# When top_similarity >= this and model still returns "not found", use fallback/retry
HIGH_CONFIDENCE_SIMILARITY = 0.65
# Per-chunk filter: only pass chunks with cosine_similarity >= this to the LLM (avoids garbage in context)
MIN_CHUNK_SIMILARITY = _env_float("MIN_CHUNK_SIMILARITY", 0.65)

# How many chunks to retrieve from Supabase per query
# Set to 5 so the assistant returns at least 5 sources when available.
TOP_K = 5

# Adaptive top_k based on query intent
TOP_K_DEFINITION = 8  # For definition queries
TOP_K_ANALYSIS = 5    # For analysis/explanation queries
TOP_K_LOOKUP = 3       # For simple lookup queries

# Query expansion settings
ENABLE_QUERY_EXPANSION = True
MAX_QUERY_VARIATIONS = 3  # Including original query
# RRF merge when combining multi-query results (reciprocal rank fusion)
ENABLE_RRF_MERGE = _env_bool("ENABLE_RRF_MERGE", True)
RRF_K = int(os.getenv("RRF_K", "60"))  # RRF constant; higher k reduces rank sensitivity

# Hybrid search settings (keyword search uses section_title; keep True to search/retrieve by section_title)
ENABLE_HYBRID_SEARCH = True
KEYWORD_SEARCH_LIMIT = 5  # Max chunks from keyword search

# Reranking settings (reranker boosts by section_title keyword match; keep True to use section_title in ranking)
ENABLE_RERANKING = True

# How many characters of content to expose as snippet to the frontend
SNIPPET_CHAR_LIMIT = 300  # characters, not tokens

# Brand name injected into generation system prompt
APP_BRAND_NAME = "SAMA/NORA Compliance Assistant"

# ---- Domain Gate Settings ----
# Enable semantic domain gate (uses embeddings instead of keywords only)
ENABLE_SEMANTIC_DOMAIN_GATE = True

# Domain similarity threshold for semantic domain gate (cosine similarity in [0, 1])
# Queries must be semantically similar to domain anchors above this threshold
DOMAIN_SIMILARITY_THRESHOLD = 0.6  # Tune based on testing

# Domain anchor phrases (representative SAMA/NORA domain concepts)
# These are embedded once and cached, then compared against user queries
DOMAIN_ANCHOR_PHRASES = [
    "SAMA banking regulations and supervision",
    "Saudi Central Bank Banking Control Law",
    "banking license requirements in Saudi Arabia",
    "SAMA regulatory compliance requirements",
    "banking sector licensing provisions",
    "SAMA supervisory authority and oversight",
    "regulatory retail exposures and capital adequacy",
    "SAMA risk management and governance",
    "banking sector compliance and reporting",
    "SAMA statutory deposit requirements",
    # Arabic domain anchors for semantic gate
    "مكافحة غسل الأموال ساما",
    "الحوكمة البنوك السعودية",
    "لوائح ساما والترخيص",
]

# ---- Phase 1: RAG tuning and safety (config-driven, no hardcoding) ----
# Similarity threshold for replacing ungrounded answers (URLs, disclaimers) with snippet fallback
UNGROUNDED_REPLACE_SIMILARITY_THRESHOLD = _env_float("UNGROUNDED_REPLACE_SIMILARITY_THRESHOLD", 0.55)
# Reranker: cap on keyword boost so semantic relevance is not outweighed by section_title match
RERANKER_KEYWORD_BOOST_CAP = _env_float("RERANKER_KEYWORD_BOOST_CAP", 0.2)
# Only apply keyword boost when base_similarity >= this (0 = always apply)
RERANKER_MIN_SIMILARITY_FOR_BOOST = _env_float("RERANKER_MIN_SIMILARITY_FOR_BOOST", 0.0)
# When blocking for out-of-scope/banned, still return retrieved sources with a message
SAFETY_REJECT_RETURN_SOURCES = _env_bool("SAFETY_REJECT_RETURN_SOURCES", True)
SAFETY_REJECT_MESSAGE = os.getenv(
    "SAFETY_REJECT_MESSAGE",
    "The answer was blocked because it referred to out-of-scope content. Below are the retrieved passages that were considered.",
)
REJECT_REASON_LOGGING = _env_bool("REJECT_REASON_LOGGING", True)
# Substrings that mark boilerplate chunk starts (used to pick first answer-rich chunk)
BOILERPLATE_PREFIXES = [
    "Standards prior to completing",
    "can delay the Licensing process",
    "Table of Contents",
    "License Application for Banking Business",
    "Circular to all banks",
]
# Out-of-scope entities: from SAFETY_BANNED_TERMS_FILE (one per line) or SAFETY_BANNED_TERMS (comma-separated) or default
_DEFAULT_BANNED_TERMS = [
    "Singapore",
    "Singapore Association of Money Managers",
    "Abu Dhabi",
    "Pakistan",
    "NBD",
    "CBP",
    "Central Bank of Pakistan",
    "National Bank of Abu Dhabi",
    "UAE",
    "United Arab Emirates",
    "Qatar",
    "Dubai",
    "DFSA",
    "Central Bank of Qatar",
    "CBQ",
    "Dubai Financial Services",
    "Securities and Commodities Authority",
    "Financial Regulatory Authority",
    "FRA",
    "Federal Law",
    "FIMSA",
]
BANNED_TERMS = _load_list_from_file_or_env(
    "SAFETY_BANNED_TERMS_FILE",
    "SAFETY_BANNED_TERMS",
    _DEFAULT_BANNED_TERMS,
)
# Wrong or nonsensical Arabic phrases: from SAFETY_BANNED_ARABIC_FILE or SAFETY_BANNED_ARABIC_PHRASES or default
_DEFAULT_BANNED_ARABIC_PHRASES = [
    "الغسل الآمن",
]
BANNED_ARABIC_PHRASES = _load_list_from_file_or_env(
    "SAFETY_BANNED_ARABIC_FILE",
    "SAFETY_BANNED_ARABIC_PHRASES",
    _DEFAULT_BANNED_ARABIC_PHRASES,
)
# Optional prompt line: when to answer from context vs say "not found"
ALLOW_BASED_ON_CONTEXT_PROMPT_LINE = os.getenv(
    "ALLOW_BASED_ON_CONTEXT_PROMPT_LINE",
    "If the context contains relevant information even if not a perfect match, provide it with a citation; only say Information not found when the context is clearly irrelevant.",
)

# ---- Phase 2: Semantic cache ----
CACHE_ENABLED = _env_bool("CACHE_ENABLED", True)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
CACHE_SIMILARITY_THRESHOLD = _env_float("CACHE_SIMILARITY_THRESHOLD", 0.95)
CACHE_BACKEND = os.getenv("CACHE_BACKEND", "memory")
REDIS_URL = os.getenv("REDIS_URL", "")

# ---- Phase 3: Query rewriter ----
ENABLE_QUERY_REWRITE = _env_bool("ENABLE_QUERY_REWRITE", True)

# ---- Phase 4: Conversational RAG (optional) ----
ENABLE_CONVERSATION_MEMORY = _env_bool("ENABLE_CONVERSATION_MEMORY", True)
CONVERSATION_MEMORY_TTL_SECONDS = int(os.getenv("CONVERSATION_MEMORY_TTL_SECONDS", "3600"))
CONVERSATION_MAX_MESSAGES = int(os.getenv("CONVERSATION_MAX_MESSAGES", "20"))

# ---- Phase 5: Advanced retrieval (optional) ----
ENABLE_HYDE = _env_bool("ENABLE_HYDE", True)
HYDE_MAX_TOKENS = int(os.getenv("HYDE_MAX_TOKENS", "80"))
HYDE_PROMPT_PREFIX = os.getenv(
    "HYDE_PROMPT_PREFIX",
    "Write a short regulatory-style answer to the following question, in one or two sentences: ",
)
# Blend weights for query vs hypothetical embedding (retrieval only)
HYDE_QUERY_WEIGHT = _env_float("HYDE_QUERY_WEIGHT", 0.5)
HYDE_HYPO_WEIGHT = _env_float("HYDE_HYPO_WEIGHT", 0.5)
ENABLE_CROSS_ENCODER_RERANK = _env_bool("ENABLE_CROSS_ENCODER_RERANK", True)
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_SELF_RAG = _env_bool("ENABLE_SELF_RAG", True)
SELF_RAG_MAX_RETRIES = int(os.getenv("SELF_RAG_MAX_RETRIES", "1"))
SELF_RAG_EXTRA_TOP_K = int(os.getenv("SELF_RAG_EXTRA_TOP_K", "10"))
# Heuristic for "poor" answer: min length to consider, and phrases that count as citation
SELF_RAG_POOR_ANSWER_MIN_LEN = int(os.getenv("SELF_RAG_POOR_ANSWER_MIN_LEN", "30"))
SELF_RAG_CITATION_PHRASES = ["page", "according to", "source", "sama", "nora"]

# ---- Simple RAG (single-file pipeline): all settings from env/config, no hardcoding in code ----
SIMPLE_RAG_TOP_K = int(os.getenv("SIMPLE_RAG_TOP_K", "8"))
# Larger k for synthesis/criteria questions to pull deeper sections
SIMPLE_RAG_TOP_K_SYNTHESIS = int(os.getenv("SIMPLE_RAG_TOP_K_SYNTHESIS", "14"))
SIMPLE_RAG_MAX_CONTENT_CHARS = int(os.getenv("SIMPLE_RAG_MAX_CONTENT_CHARS", "1800"))
SIMPLE_RAG_MAX_NEW_TOKENS = int(os.getenv("SIMPLE_RAG_MAX_NEW_TOKENS", "500"))
SIMPLE_RAG_NOT_FOUND_MESSAGE = os.getenv(
    "SIMPLE_RAG_NOT_FOUND_MESSAGE",
    "Information not found in SAMA/NORA documents.",
)
SIMPLE_RAG_ANSWER_MARKER = os.getenv(
    "SIMPLE_RAG_ANSWER_MARKER",
    "Answer in clear sentences using only the context above:",
)
SIMPLE_RAG_PROMPTS_DIR = BACKEND_DIR / "prompts"
SIMPLE_RAG_LOG_DIR = BACKEND_DIR / "logs"
SIMPLE_RAG_LOG_PATH = os.getenv(
    "SIMPLE_RAG_LOG_PATH",
    str(SIMPLE_RAG_LOG_DIR / "simple_rag.log"),
)
SIMPLE_RAG_STRICT_CITATION = _env_bool("SIMPLE_RAG_STRICT_CITATION", True)
SIMPLE_RAG_OUT_OF_SCOPE_MESSAGE = os.getenv(
    "SIMPLE_RAG_OUT_OF_SCOPE_MESSAGE",
    "I only answer questions about SAMA and NORA documents. I don't have information on that.",
)
# Pipe-separated: query must contain at least one to be in-scope (SAMA/NORA/banking/regulations)
SIMPLE_RAG_SCOPE_KEYWORDS = [
    s.strip().lower()
    for s in os.getenv(
        "SIMPLE_RAG_SCOPE_KEYWORDS",
        "sama|nora|bank|banks|banking|license|licensing|regulation|regulatory|consumer|compliance|report|capital|requirement|cybersecurity|ساما|نورا|بنك|بنوك|ترخيص|قانون|مراقبة",
    ).split("|")
    if s.strip()
]
# Pipe-separated: if query contains any of these (case-insensitive), treat as off-topic and refuse
SIMPLE_RAG_OFF_TOPIC_PATTERNS = [
    s.strip().lower()
    for s in os.getenv(
        "SIMPLE_RAG_OFF_TOPIC_PATTERNS",
        "us president|who is the president|who's the president|weather|sports|football|recipe|movie|russia|where is|world cup|capital of france",
    ).split("|")
    if s.strip()
]
SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC = os.getenv(
    "SIMPLE_RAG_NOT_FOUND_MESSAGE_ARABIC",
    "لم يتم العثور على المعلومات في وثائق ساما/نورا.",
)
# Terms that indicate confabulation if present in answer but not in context (comma-separated env or default; override via SIMPLE_RAG_CONFABULATION_BLOCKLIST)
SIMPLE_RAG_CONFABULATION_BLOCKLIST = [
    s.strip().lower()
    for s in os.getenv(
        "SIMPLE_RAG_CONFABULATION_BLOCKLIST",
        "Kuwait,Dubai,ECRIS,IMF,Mint,YNAB,MEED,FMA,tourist visa,fit and proper,capital adequacy ratio",
    ).split(",")
    if s.strip()
]
SIMPLE_RAG_PROMPTS_JSON_PATH = os.getenv(
    "SIMPLE_RAG_PROMPTS_JSON_PATH",
    str(SIMPLE_RAG_PROMPTS_DIR / "simple_rag_prompts.json"),
)
SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH = os.getenv(
    "SIMPLE_RAG_TEST_QUESTIONS_JSON_PATH",
    str(SIMPLE_RAG_PROMPTS_DIR / "simple_rag_test_questions.json"),
)
# Fallback when prompts JSON is missing or key missing (env SIMPLE_RAG_SYSTEM_PROMPT / SIMPLE_RAG_USER_TEMPLATE overrides)
SIMPLE_RAG_SYSTEM_PROMPT_DEFAULT = (
    "If the question is not about SAMA/NORA regulations or the provided documents, respond only with: "
    "This system only answers questions related to the SAMA Banking Sector Rulebook.\n\n"
    "The retrieved context may contain English or Arabic text. If the user question is in Arabic: you may use information "
    "from both English and Arabic context; you must provide the final answer in Arabic; do not introduce information "
    "not present in the context; if the answer is not found in the context (in either language), respond exactly: "
    "لم يتم العثور على المعلومات في وثائق ساما/نورا.\n\n"
    "You must answer ONLY using sentences that appear in the retrieved context (quote or clear restatement). "
    "If the answer cannot be directly supported by quoting or clearly restating sentences from the context, respond exactly: "
    "Information not found in SAMA/NORA documents. Do NOT infer. Do NOT use general knowledge. Do NOT add qualifications or commentary. "
    "Your answer must include the exact sentence or phrase from the context that supports it. If you cannot find such a sentence, "
    "respond exactly: Information not found in SAMA/NORA documents. You are NOT allowed to: infer meaning, interpret abbreviations, "
    "expand acronyms, guess definitions, or use prior knowledge.\n\n"
    "If the question asks what X is or the definition of X (including acronyms like NORA), you may answer only if the context contains "
    "an explicit definition or clear explanation of X (e.g. \"X is …\", \"X stands for …\", \"X refers to …\"). "
    "If the context does not define or explain X, respond exactly: Information not found in SAMA/NORA documents. Do not infer or invent a definition.\n\n"
    "Before answering, check: Is the answer directly supported by a phrase in the context? If not, respond exactly: Information not found in SAMA/NORA documents.\n\n"
    "CRITICAL: Every sentence in your answer MUST end with (Page X) or (Pages X–Y). Where possible, support your answer with a direct quote from the context, "
    "followed by (Page X) or (Pages X–Y). Do not output any sentence without a citation.\n\n"
    "Write in full sentences, not only bullet headings. Each sentence must state the requirement or criterion clearly and end with (Page X) or (Pages X–Y). "
    "Bad: \"1. Types of licenses;\" Good: \"The guidelines describe the types of licenses available (Pages 1–2).\" "
    "For definition questions, use this format when the context defines the term: Definition: \"quoted sentence from context\" (Page X).\n\n"
    "Rules: Do not invent steps, numbers, or procedures. Do not mention entities or terms not in the context. Keep the answer concise: at most 5 points. "
    "Use the same language as the question (Arabic or English). Every sentence in your answer MUST end with (Page X) or (Pages X–Y). No exceptions."
)
SIMPLE_RAG_USER_TEMPLATE_DEFAULT = (
    "### CONTEXT\n\n{context}\n\n### QUESTION\n{question}\n\n"
    "Output only your answer. Do not repeat the question or any instructions.\n\nAnswer:"
)
# Optional system line for fact/definition questions: extract exact phrase and cite (Page X); one sentence enough
SIMPLE_RAG_SYSTEM_PROMPT_FACT_DEFINITION = os.getenv(
    "SIMPLE_RAG_SYSTEM_PROMPT_FACT_DEFINITION",
    "For questions asking for a single fact, decree name, law, or definition: extract the exact phrase from the context and cite (Page X). One sentence is enough. Do not add extra explanation.",
)
# Optional system line for criteria/requirements questions: require at least one direct quote
SIMPLE_RAG_SYSTEM_PROMPT_SYNTHESIS = os.getenv(
    "SIMPLE_RAG_SYSTEM_PROMPT_SYNTHESIS",
    "For criteria or requirements questions, at least one sentence in your answer must be a direct quote or near-verbatim restatement from the context, followed by (Page X).",
)
# Arabic queries: reinforce no translation beyond context (env-overridable)
SIMPLE_RAG_SYSTEM_PROMPT_ARABIC_SUFFIX = os.getenv(
    "SIMPLE_RAG_SYSTEM_PROMPT_ARABIC_SUFFIX",
    "Answer only in Arabic. Use only the exact information from the context; do not add or translate information that is not in the context. If the context does not contain the answer, respond exactly: لم يتم العثور على المعلومات في وثائق ساما/نورا.",
)
# Filler phrases to strip from model output (pipe-separated env so phrases can contain commas)
SIMPLE_RAG_FILLER_PHRASES = [
    s.strip()
    for s in os.getenv(
        "SIMPLE_RAG_FILLER_PHRASES",
        "As mentioned earlier|As mentioned earlier,|As stated above|As noted previously|As noted above",
    ).split("|")
    if s.strip()
]
# Phrases that signal instruction echo in model output; truncate decoded text before first occurrence (pipe-separated)
SIMPLE_RAG_ECHO_PHRASES = [
    s.strip()
    for s in os.getenv(
        "SIMPLE_RAG_ECHO_PHRASES",
        "Please write in English language|Answer only in Arabic|Question:|Answer in clear sentences using only the context above",
    ).split("|")
    if s.strip()
]
# Phrases that signal conversation-turn leakage (e.g. fake Human:/Assistant:); truncate decoded text before first occurrence (pipe-separated)
SIMPLE_RAG_STOP_PHRASES = [
    s.strip()
    for s in os.getenv(
        "SIMPLE_RAG_STOP_PHRASES",
        "Human:|Assistant:|\n\nHuman:|\n\nAssistant:",
    ).split("|")
    if s.strip()
]
