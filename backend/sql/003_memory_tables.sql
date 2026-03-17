-- Memory-related tables: user_profile, session_summary, memory_item, memory_item_embedding.
-- Run in Supabase SQL Editor after 001_user_session_tables.sql and 002_feedback_stars_and_messages.sql.

-- Ensure pgvector extension is available for memory embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. user_profile: long-lived, structured user preferences
CREATE TABLE IF NOT EXISTS user_profile (
  user_id UUID PRIMARY KEY REFERENCES "user"(user_id) ON DELETE CASCADE,
  preferred_language TEXT, -- e.g. 'en' or 'ar'
  strictness_level SMALLINT CHECK (strictness_level BETWEEN 1 AND 5),
  topics TEXT[],           -- e.g. {'licensing','capital'}
  flags JSONB,             -- extensible bag for boolean/other preferences
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. session_summary: compressed view of a session's conversation
CREATE TABLE IF NOT EXISTS session_summary (
  session_id UUID PRIMARY KEY REFERENCES session(session_id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES "user"(user_id) ON DELETE CASCADE,
  summary_text TEXT,
  summary_json JSONB,
  message_count INTEGER DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_session_summary_user_id ON session_summary(user_id);

-- 3. memory_item: selective episodic memory entries
CREATE TABLE IF NOT EXISTS memory_item (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES "user"(user_id) ON DELETE CASCADE,
  session_id UUID REFERENCES session(session_id) ON DELETE CASCADE,
  type TEXT NOT NULL,              -- e.g. 'preference','decision','entity','clarification'
  text TEXT NOT NULL,              -- short description of the memory
  metadata JSONB,                  -- structured details (preference_key, entity name, etc.)
  source_message_id UUID REFERENCES session_messages(message_id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_memory_item_user_id ON memory_item(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_item_session_id ON memory_item(session_id);

-- 4. memory_item_embedding: vector representation for semantic search over memory items
-- NOTE: the embedding vector dimension must match EMBEDDING_DIMENSION in backend/config.py.
CREATE TABLE IF NOT EXISTS memory_item_embedding (
  memory_item_id UUID PRIMARY KEY REFERENCES memory_item(id) ON DELETE CASCADE,
  embedding vector(384), -- default to 384 (multilingual-e5-small); update if EMBEDDING_DIMENSION changes
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_memory_item_embedding_vector
  ON memory_item_embedding
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

