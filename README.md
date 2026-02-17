<div align="center">

# 💡 IOTA Techbologies : Regulation AI

**SAMA/NORA Compliance Assistant**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com/)
[![Qwen](https://img.shields.io/badge/Qwen-1.8B-FF6B6B?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Qwen)

[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-RegTech-FF6B6B?style=for-the-badge)](https://regtech.iotatechnologies.io)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen?style=for-the-badge)](https://regtech.iotatechnologies.io)

---

**LLM-Powered Regulatory Compliance Q&A System**

IOTA v2 is an intelligent compliance assistant designed specifically for SAMA (Saudi Arabian Monetary Authority) and NORA regulatory documents. Built on a RAG (Retrieval-Augmented Generation) architecture, it provides accurate, citation-backed answers with strict domain containment to ensure compliance-grade responses.

🌐 **Live Demo**: [https://regtech.iotatechnologies.io](https://regtech.iotatechnologies.io)

| English | [العربية](#) |

</div>

---

## 📌 Overview

IOTA v2 combines advanced document understanding, semantic retrieval, and controlled LLM generation to deliver precise regulatory compliance assistance. The system enforces strict containment at multiple levels, ensuring answers are **only** derived from authorized SAMA/NORA documents with full page-level traceability.

### Key Characteristics

- ✅ **Domain-Contained**: Answers ONLY from SAMA/NORA documents
- ✅ **Page-Traceable**: Every answer includes document name and page citations
- ✅ **Deterministic**: Identical queries produce identical results
- ✅ **Multilingual**: Supports Arabic and English
- ✅ **Streaming Responses**: Real-time answer generation
- ✅ **Production-Ready**: Optimized for performance and scalability

---

## ✨ Key Features

### 🤖 Intelligent Q&A

- **Streaming Responses**: Real-time answer generation with incremental text updates
- **Citation System**: Automatic source attribution with document names and page numbers (Page X or Pages X–Y)
- **Citation Validation**: Verifies that cited page numbers exist in retrieved chunks
- **Context-Aware**: Multi-turn conversations with conversation history (configurable max messages/chars)
- **Domain Gate**: Automatic rejection of non-regulatory queries (keyword-based filtering)
- **Intent-Based Generation**: Specialized prompts and processing for different query types
- **Extractive Builder**: For fact_definition/metadata queries, extracts direct answers from chunks
- **Semantic Grounding**: Post-generation similarity checks to ensure answers are grounded in context
- **Confabulation Detection**: Blocklist system to detect and remove ungrounded terms
- **Entity Containment Check**: Ensures fact_definition/metadata answers contain query entities
- **Answer Language Validation**: Ensures Arabic queries receive Arabic answers (with translation fallback)
- **Definition Guard**: Special handling for "what is X?" queries to prevent hallucination
- **Post-Generation Similarity Check**: Validates answer similarity to retrieved chunks
- **Translation Support**: Optional translation for Arabic queries to improve retrieval

### 📚 Document Processing

- **PDF Parsing**: Advanced PDF extraction with PyMuPDF
- **OCR Support**: Image-based document processing with PaddleOCR (en + ar, 200 DPI)
- **Multilingual Extraction**: Handles Arabic and English content
- **Structured Chunking**: Intelligent document segmentation (500 tokens, 120 overlap)
- **Header/Footer Removal**: Automatic detection and removal of repeated headers/footers
- **Sentence Boundary Preservation**: Smart chunking that respects sentence endings

### 🔍 Advanced Semantic Retrieval

- **Vector Search**: pgvector-based similarity search with configurable embeddings
  - **Local Model (Default)**: SentenceTransformer with multilingual-e5-small (384-dim) - runs locally on CPU/GPU
  - **Azure OpenAI (Optional)**: text-embedding-3-small (1536-dim) or text-embedding-3-large (3072-dim) - requires API key
- **Intent Classification**: Automatic query intent detection (fact_definition, metadata, procedural, synthesis, other)
- **Intent-Aware Retrieval**: Different top-k values per intent type (synthesis uses more chunks)
- **Dual Retrieval for Arabic**: Arabic queries trigger both Arabic and English embeddings, merged with RRF
- **Second-Pass Retrieval**: When similarity is borderline, automatically re-fetches with larger k
- **Dynamic Top-K**: Adjusts retrieval depth based on similarity scores
- **RRF Merging**: Reciprocal Rank Fusion to combine results from multiple queries
- **Reranking**: Cross-encoder reranking with keyword boosting and definition section prioritization
- **Ontology-Based Selection**: Preferred document selection based on keyword matching
- **Query Normalization**: Acronym expansion (SAMA → "Saudi Arabian Monetary Authority"), legal term mapping
- **Multiple Similarity Thresholds**: Different thresholds per intent (synthesis, procedural, fact_definition, metadata)
- **Strict Quality Gates**: Additional validation layers for fact_definition and metadata queries

### 🔐 Authentication & Security

- **WebAuthn Support**: Fingerprint-based authentication for secure access
- **Prompt Limits**: Rate limiting for unauthenticated users (10 prompts/day)
- **Session Management**: Persistent chat sessions with user tracking
- **CORS Protection**: Configurable origin whitelisting

### 💬 Chat Interface

- **Modern UI**: Beautiful, responsive chat interface built with Next.js
- **Chat History**: Persistent conversation history with local storage
- **Source Preview**: Interactive source panel showing retrieved document snippets
- **Markdown Rendering**: Rich text formatting for answers
- **Mobile Responsive**: Optimized for desktop and mobile devices

### ⚡ Performance Optimizations

- **Efficient Storage**: Optimized chat history management
- **Lightweight Rendering**: Smart markdown rendering for large responses
- **Streaming Architecture**: Non-blocking response streaming
- **Memory Management**: Capped message history to prevent performance degradation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Frontend Layer                               │
│  Next.js 14 + React 18 + Tailwind CSS + TypeScript                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Chat Interface │ Authentication │ Chat History │ Sources     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP/REST API (FastAPI)
┌──────────────────────────────▼──────────────────────────────────────┐
│                         Backend Layer                                │
│  FastAPI + Python 3.10+                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Query Processing Pipeline                                    │   │
│  │  1. Domain Gate (Keyword Filter)                             │   │
│  │  2. Query Normalization (Acronym Expansion)                  │   │
│  │  3. Intent Classification (5 Types)                         │   │
│  │  4. Embedding Generation (Local Model)                      │   │
│  │  5. Vector Search (Supabase pgvector)                        │   │
│  │  6. Second-Pass Retrieval (if needed)                       │   │
│  │  7. Reranking (Cross-Encoder + Keyword Boost)              │   │
│  │  8. Context Assembly                                         │   │
│  │  9. LLM Generation (Qwen 1.8B Local) OR Extractive Builder  │   │
│  │  10. Post-Generation Validation (Grounding, Citations)       │   │
│  │  11. Streaming Response (SSE)                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────┬──────────────────┬──────────────────┬────────────────────┘
           │                  │                  │
    ┌──────▼──────┐  ┌───────▼───────┐  ┌──────▼──────┐
    │  Supabase   │  │ Local Models   │  │  Optional   │
    │ PostgreSQL  │  │ ┌───────────┐ │  │  Azure/     │
    │ + pgvector  │  │ │Sentence   │ │  │  OpenAI     │
    │             │  │ │Transformer │ │  │  (for      │
    │ - Chunks    │  │ │(Embeddings)│ │  │  translation│
    │ - Vectors   │  │ └───────────┘ │  │  only)      │
    │ - Sessions  │  │ ┌───────────┐ │  └────────────┘
    │ - Feedback  │  │ │Qwen 1.8B   │ │
    └─────────────┘  │ │(Generation)│ │
                     │ └───────────┘ │
                     └───────────────┘
```

### System Components

1. **Document Ingestion**: PDF processing, OCR, chunking pipeline (500 tokens, 120 overlap)
2. **Embedding Generation**: 
   - **Local Model (Default)**: SentenceTransformer with multilingual-e5-small (384-dim) - runs locally
   - **Azure OpenAI (Optional)**: text-embedding-3-small (1536-dim) or text-embedding-3-large (3072-dim) - requires API key
3. **Vector Storage**: Supabase PostgreSQL with pgvector extension
4. **Query Processing Pipeline**:
   - Domain gate (keyword filtering)
   - Query normalization (acronym expansion, legal terms)
   - Intent classification (5 types)
   - Vector search (with optional dual retrieval for Arabic)
   - Second-pass retrieval (if similarity borderline)
   - Reranking (cross-encoder + keyword boosting)
   - Context assembly
   - LLM generation (Qwen 1.8B 4-bit) OR extractive builder
   - Post-generation validation (grounding, confabulation, citations)
   - Answer language validation
5. **Response Streaming**: Incremental text delivery to frontend via SSE
6. **Session Management**: User and conversation tracking with Supabase
7. **Feedback System**: Star ratings (1-5) with optional comments

---

## 🎯 Use Cases

| Scenario                        | Application                                   | Core Value                                               |
| ------------------------------- | --------------------------------------------- | -------------------------------------------------------- |
| **Regulatory Compliance** | SAMA/NORA rulebook Q&A, policy interpretation | Accurate, cited answers for compliance teams             |
| **Legal Research**        | Regulatory document search, clause retrieval  | Fast access to relevant regulations with page references |
| **Training & Onboarding** | Staff training on regulatory requirements     | Interactive learning with source-backed explanations     |
| **Audit Support**         | Document verification, citation checking      | Traceable answers for audit documentation                |
| **Multilingual Support**  | Arabic/English regulatory queries             | Seamless language switching for diverse teams            |

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.10+
- **PostgreSQL** (via Supabase with pgvector extension)
- **GPU** (required for local Qwen inference, ~6GB VRAM for 4-bit quantized model)
- **CUDA** (for GPU acceleration)
- **HuggingFace Token** (optional, for gated models like Qwen)
- **Azure OpenAI API key** (optional, only if using Azure embeddings instead of local model)
- **OpenAI API key** (optional, only if using Arabic translation feature)

### Installation

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-org/Iotav2.git
cd Iotav2
```

#### 2️⃣ Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials:
# - SUPABASE_URL (required)
# - SUPABASE_SERVICE_ROLE_KEY (required)
# - CORS_ORIGINS (required)
# - USE_MULTILINGUAL_EMBEDDING=true (default, uses local SentenceTransformer)
# - AZURE_OPENAI_API_KEY (optional, only if USE_MULTILINGUAL_EMBEDDING=false)
# - AZURE_OPENAI_ENDPOINT (optional, only if using Azure embeddings)
# - OPENAI_API_KEY (optional, only if using Arabic translation)
```

#### 3️⃣ Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env.local
# Edit .env.local:
# NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### 4️⃣ Start Services

**Backend** (Terminal 1):

```bash
cd backend
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend** (Terminal 2):

```bash
cd frontend
npm run dev
```

#### 5️⃣ Access the Application

- **Web UI**: [http://localhost:3000](http://localhost:3000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📘 API Reference

### Endpoints

#### `POST /api/user`

Create a new user session.

**Response:**

```json
{
  "user_id": "uuid-string"
}
```

#### `POST /api/query-stream`

Streaming query endpoint for real-time responses.

**Request:**

```json
{
  "query": "What are the capital requirements for banks?",
  "user_id": "optional-uuid",
  "session_id": "optional-uuid"
}
```

**Response:** Server-Sent Events (SSE) stream with JSON lines:

```json
{"type": "chunk", "text": "According to SAMA regulations..."}
{"type": "chunk", "text": " the minimum capital requirement..."}
{"type": "meta", "meta": {
  "sources": [
    {
      "document_name": "SAMA Banking Regulations",
      "page_start": 45,
      "page_end": 47,
      "snippet": "...",
      "article_id": "Article 12"
    }
  ],
  "session_id": "uuid",
  "message_id": "uuid",
  "user_id": "uuid"
}}
{"type": "done"}
```

#### `POST /api/query`

Non-streaming query endpoint (returns complete response).

**Request:** Same as `/api/query-stream`

**Response:**

```json
{
  "answer": "According to SAMA regulations...",
  "sources": [...],
  "message_id": "uuid",
  "user_id": "uuid",
  "session_id": "uuid"
}
```

#### `POST /api/feedback`

Submit feedback for a response (star rating 1-5 with optional comments).

**Request:**

```json
{
  "session_id": "uuid",
  "user_id": "uuid",
  "message_id": "uuid",
  "feedback": 5,
  "comments": "Very helpful answer"
}
```

#### `POST /api/session`

Create a new chat session for a user.

**Request:**

```json
{
  "user_id": "uuid"
}
```

**Response:**

```json
{
  "session_id": "uuid"
}
```

#### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

### Authentication

The API supports optional user authentication via WebAuthn. Unauthenticated users are limited to 10 prompts per day.

---

## 🔧 Configuration

### Backend Environment Variables

```env
# Embeddings: Local model (default) - runs locally, no API key needed
USE_MULTILINGUAL_EMBEDDING=true  # Default: true (uses local SentenceTransformer)
MULTILINGUAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
MULTILINGUAL_EMBEDDING_DIMENSION=384

# OR use Azure OpenAI embeddings (set USE_MULTILINGUAL_EMBEDDING=false)
# USE_MULTILINGUAL_EMBEDDING=false
# AZURE_OPENAI_API_KEY=your-key
# AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
# AZURE_EMBEDDING_MODEL=text-embedding-3-small  # or text-embedding-3-large
# AZURE_EMBEDDING_DIMENSION=1536  # or 3072 for large

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-key

# CORS
CORS_ORIGINS=http://localhost:3000,https://your-domain.com

# Optional: OpenAI API (for Arabic translation)
OPENAI_API_KEY=your-openai-key

# Qwen Model (default: Qwen/Qwen1.5-1.8B-Chat)
QWEN_MODEL=Qwen/Qwen1.5-1.8B-Chat

# HuggingFace Token (for gated models)
HF_TOKEN=your-hf-token
```

### Frontend Environment Variables

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🎨 Interface Showcase

### Chat Interface

- **Streaming Responses**: Real-time answer generation with typing indicators
- **Source Citations**: Clickable source references with page numbers
- **Chat History**: Persistent conversation history in sidebar
- **Markdown Support**: Rich text formatting for answers

### Authentication

- **WebAuthn Login**: Fingerprint-based secure authentication
- **Prompt Counter**: Visual indicator of remaining prompts
- **Session Management**: Automatic session creation and tracking

### Source Panel

- **Document Preview**: View retrieved document snippets
- **Page References**: Direct links to source pages
- **Highlighting**: Highlighted relevant text in sources

---

## 🧩 Feature Matrix

| Feature                       | Status | Description                              |
| ----------------------------- | ------ | ---------------------------------------- |
| **Streaming Responses** | ✅     | Real-time incremental text delivery      |
| **Source Citations**    | ✅     | Document name and page number references |
| **Chat History**        | ✅     | Persistent conversation storage          |
| **WebAuthn Auth**       | ✅     | Fingerprint-based authentication         |
| **Prompt Limits**       | ✅     | Rate limiting for unauthenticated users  |
| **Multilingual**        | ✅     | Arabic and English support with dual retrieval |
| **Domain Gate**         | ✅     | Automatic query filtering                |
| **Intent Classification** | ✅   | 5 intent types (fact_definition, metadata, procedural, synthesis, other) |
| **Vector Search**       | ✅     | Semantic similarity retrieval            |
| **Reranking**          | ✅     | Cross-encoder reranking with keyword boosting |
| **Second-Pass Retrieval** | ✅  | Automatic re-fetch with larger k when similarity borderline |
| **RRF Merging**        | ✅     | Reciprocal Rank Fusion for multi-query results |
| **Query Normalization** | ✅    | Acronym expansion, legal term mapping |
| **Extractive Builder** | ✅     | Direct extraction for fact_definition/metadata |
| **Semantic Grounding**  | ✅     | Post-generation similarity validation |
| **Confabulation Detection** | ✅  | Blocklist-based ungrounded term detection |
| **Citation Validation** | ✅    | Verifies cited pages exist in chunks |
| **Answer Language Validation** | ✅ | Ensures Arabic queries get Arabic answers |
| **PDF Processing**      | ✅     | PyMuPDF-based extraction                 |
| **OCR Support**         | ✅     | PaddleOCR image processing (en + ar)    |
| **Markdown Rendering**  | ✅     | Rich text formatting (limited for large responses) |
| **Mobile Responsive**   | ✅     | Optimized mobile interface               |
| **Conversation History** | ✅    | Configurable max messages/chars per session |
| **Feedback System**    | ✅     | Star ratings (1-5) with optional comments |

---

## 🔒 Security & Compliance

### Containment Strategy

The system enforces strict domain containment at multiple levels:

1. **Domain Gate**: Rule-based keyword filtering (SAMA, NORA, regulatory terms in EN + AR)
2. **Intent Classification**: Query type detection for specialized processing
3. **Similarity Thresholds**: Intent-specific thresholds (synthesis, procedural, fact_definition, metadata)
4. **Strict Quality Gates**: Additional validation for fact_definition/metadata queries
5. **Generation Prompt**: Strict context-only answer generation with mandatory citations
6. **Semantic Grounding**: Post-generation similarity checks
7. **Confabulation Detection**: Blocklist terms that appear in answer but not context
8. **Entity Containment**: Ensures fact_definition/metadata answers contain query entities
9. **Citation Validation**: Verifies cited pages exist in retrieved chunks
10. **Definition Guard**: Prevents hallucination for "what is X?" queries

### Security Features

- ✅ **CORS Protection**: Configurable origin whitelisting
- ✅ **API Authentication**: Optional WebAuthn-based user authentication
- ✅ **Rate Limiting**: Prompt limits for unauthenticated access
- ✅ **Input Validation**: Strict query validation and sanitization
- ✅ **Error Handling**: Safe error messages without internal details

---

## ⚡ Performance Optimizations

### Frontend Optimizations

- **Efficient Storage**: Optimized chat history management (max 50 chats, max 50 rendered messages)
- **Lightweight Rendering**: Smart markdown rendering (disabled for responses >1000 chars)
- **Memoization**: React.useMemo for sorted chat lists, React.memo for expensive components
- **Fixed Height Input**: Textarea with fixed height to prevent layout shifts
- **Narrowed Transitions**: Only color/shadow transitions, not layout-affecting properties
- **Disabled Highlighting**: Snippet highlighting disabled for performance (O(n²) algorithm)
- **Message Capping**: Only renders latest 50 messages to limit DOM work

### Backend Optimizations

- **Streaming Architecture**: Non-blocking response streaming via threading and queue
- **Connection Pooling**: Efficient database connection management via Supabase client
- **Model Caching**: Qwen model loaded once and reused (persistent in memory)
- **Quantization**: 4-bit NF4 quantization reduces VRAM usage (~6GB for 1.8B model)
- **Intent-Aware Processing**: Different retrieval strategies per query type
- **Early Rejection**: Domain gate prevents unnecessary API calls
- **Batch Processing**: Efficient chunk insertion to Supabase

---

## 🛠️ Development

### Project Structure

```
Iotav2/
├── backend/                    # FastAPI backend
│   ├── server.py               # Main API server (FastAPI routes)
│   ├── simple_rag.py          # Core RAG pipeline (retrieval, generation, validation)
│   ├── qwen_model.py          # Qwen 1.8B LLM inference (4-bit quantized)
│   ├── embeddings.py          # Embedding generation (Azure OpenAI or multilingual)
│   ├── rerank.py              # Cross-encoder reranking with keyword boosting
│   ├── query_multilingual.py  # Arabic detection, dual retrieval, RRF merging
│   ├── query_normalize.py     # Query normalization (acronym expansion, legal terms)
│   ├── extractive_builder.py  # Direct extraction for fact_definition/metadata
│   ├── grounding.py           # Semantic grounding validation
│   ├── ontology.py            # Ontology-based document selection
│   ├── translate.py           # Arabic translation support
│   ├── users_sessions.py      # User, session, message, feedback management
│   ├── supabase_client.py     # Supabase client wrapper
│   ├── config.py              # Configuration (thresholds, flags, paths)
│   └── requirements.txt
├── frontend/                   # Next.js frontend
│   ├── app/                   # Next.js app directory
│   │   ├── page.tsx           # Main chat page
│   │   └── api/               # API routes (if any)
│   ├── components/            # React components
│   │   ├── chat/              # Chat interface components
│   │   ├── sidebar/           # Chat history sidebar
│   │   └── ui/                # shadcn/ui components
│   ├── lib/                   # Utilities and types
│   │   ├── types.ts           # TypeScript interfaces
│   │   ├── storage.ts         # localStorage utilities
│   │   └── utils.ts           # Helper functions
│   ├── hooks/                 # Custom React hooks
│   │   ├── useFingerprintAuth.ts  # WebAuthn authentication
│   │   └── usePromptLimit.ts      # Prompt limit tracking
│   └── package.json
├── Core/                      # Architecture documentation
│   └── 01_overall_architecture.md
└── README.md
```

### Development Mode

**Backend** (with hot reload):

```bash
cd backend
source venv/bin/activate
uvicorn server:app --reload
```

**Frontend** (with hot reload):

```bash
cd frontend
npm run dev
```

---

## 📊 Technology Stack

### Frontend

- **Framework**: Next.js 14 (App Router)
- **UI Library**: React 18
- **Styling**: Tailwind CSS
- **Components**: Radix UI, shadcn/ui
- **Animations**: Framer Motion
- **Markdown**: react-markdown + remark-gfm
- **Authentication**: @simplewebauthn/browser

### Backend

- **Framework**: FastAPI
- **Language**: Python 3.10+
- **LLM**: Qwen 1.8B Instruct (4-bit quantized with BitsAndBytes)
- **Embeddings**: 
  - **Local Model (Default)**: SentenceTransformer with multilingual-e5-small (384-dim) - runs locally
  - **Azure OpenAI (Optional)**: text-embedding-3-small (1536-dim) or text-embedding-3-large (3072-dim) - requires API key
- **Database**: Supabase (PostgreSQL + pgvector)
- **PDF Processing**: PyMuPDF (fitz)
- **OCR**: PaddleOCR (en + ar, 200 DPI)
- **Vector Search**: pgvector (cosine similarity)
- **Reranking**: Cross-encoder reranking with keyword boosting

---

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **SAMA** (Saudi Arabian Monetary Authority) for regulatory framework
- **Supabase** for database infrastructure
- **Qwen** team for the open-source LLM

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/Iotav2/issues)

---

<div align="center">

**Built with ❤️ by IOTA Technologies**

[⭐ Star us on GitHub](https://github.com/your-org/Iotav2) | [🐛 Report Bug](https://github.com/your-org/Iotav2/issues)

</div>
