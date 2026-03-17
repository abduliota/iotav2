import { useEffect, useMemo, useState } from 'react';
import type { Chat, Message } from '../lib/types';

/**
 * Minimal chat hook for the IOTAV3 frontend.
 *
 * This keeps chat state and the derived latest references in one place,
 * while letting the existing ChatInterface component handle streaming.
 */
export function useIotav3Chat() {
  const [userId, setUserId] = useState<string | null>(null);
  const [chat, setChat] = useState<Chat | null>(null);

  useEffect(() => {
    const existing = window.localStorage.getItem('iotav3_user_id');
    if (existing) {
      setUserId(existing);
      return;
    }
    const API_URL =
      process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    fetch(`${API_URL}/api/user`, { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        const id = data?.user_id;
        if (id) {
          window.localStorage.setItem('iotav3_user_id', id);
          setUserId(id);
        }
      })
      .catch(() => {
        // frontend can still work without a user_id; backend will
        // create one lazily on first query.
      });
  }, []);

  const latestAssistantWithRefs: Message | null = useMemo(() => {
    if (!chat || !chat.messages.length) return null;
    const assistant = [...chat.messages]
      .reverse()
      .find(m => m.role === 'assistant' && m.references?.length);
    return assistant ?? null;
  }, [chat]);

  return {
    userId,
    chat,
    setChat,
    latestAssistantWithRefs,
  };
}

