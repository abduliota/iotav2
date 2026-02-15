-- User, session, session_messages, session_feedback tables for Iotav2 chat.
-- Run in Supabase SQL Editor or via migrations.

-- 1. user
CREATE TABLE IF NOT EXISTS "user" (
  user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. session
CREATE TABLE IF NOT EXISTS session (
  session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES "user"(user_id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_session_user_id ON session(user_id);

-- 3. session_messages
CREATE TABLE IF NOT EXISTS session_messages (
  message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES session(session_id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES "user"(user_id) ON DELETE CASCADE,
  user_message TEXT NOT NULL,
  assistant_message TEXT NOT NULL,
  "timestamp" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_session_messages_session_id ON session_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_session_messages_user_id ON session_messages(user_id);

-- 4. session_feedback
CREATE TABLE IF NOT EXISTS session_feedback (
  session_id UUID NOT NULL REFERENCES session(session_id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES "user"(user_id) ON DELETE CASCADE,
  message_id UUID NOT NULL REFERENCES session_messages(message_id) ON DELETE CASCADE,
  feedback SMALLINT NOT NULL CHECK (feedback IN (0, 1)),
  comments TEXT,
  "timestamp" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (session_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_session_feedback_session_id ON session_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_session_feedback_user_id ON session_feedback(user_id);
