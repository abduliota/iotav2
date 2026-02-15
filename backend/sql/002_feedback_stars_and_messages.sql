-- Feedback: 1-5 stars; add user_message and assistant_message for finetuning export.
-- Run in Supabase SQL Editor after 001_user_session_tables.sql.

ALTER TABLE session_feedback DROP CONSTRAINT IF EXISTS session_feedback_feedback_check;
ALTER TABLE session_feedback ADD CONSTRAINT session_feedback_rating_check CHECK (feedback >= 1 AND feedback <= 5);
ALTER TABLE session_feedback ADD COLUMN IF NOT EXISTS user_message TEXT;
ALTER TABLE session_feedback ADD COLUMN IF NOT EXISTS assistant_message TEXT;
