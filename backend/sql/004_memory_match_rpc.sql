-- RPC: match_memory_items(query_embedding, match_count, user_id)
-- Returns the top-N episodic memory items for a given user ordered by cosine similarity.
-- Assumes pgvector extension and memory_item / memory_item_embedding tables from 003_memory_tables.sql.

CREATE OR REPLACE FUNCTION match_memory_items(
  query_embedding vector(384),
  match_count INT,
  match_user_id UUID
)
RETURNS TABLE (
  memory_item_id UUID,
  user_id UUID,
  session_id UUID,
  type TEXT,
  text TEXT,
  metadata JSONB,
  similarity FLOAT
) AS $$
  SELECT
    mi.id AS memory_item_id,
    mi.user_id,
    mi.session_id,
    mi.type,
    mi.text,
    mi.metadata,
    1 - (mie.embedding <=> query_embedding) AS similarity
  FROM memory_item_embedding mie
  JOIN memory_item mi ON mi.id = mie.memory_item_id
  WHERE mi.user_id = match_user_id
  ORDER BY mie.embedding <=> query_embedding
  LIMIT match_count;
$$ LANGUAGE sql STABLE;

