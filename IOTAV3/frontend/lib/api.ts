import { Chat } from './types';

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function createUser(): Promise<string | null> {
  try {
    const res = await fetch(`${API_URL}/api/user`, { method: 'POST' });
    if (!res.ok) return null;
    const data = await res.json();
    return data?.user_id ?? null;
  } catch {
    return null;
  }
}

export interface QueryStreamMeta {
  user_id?: string;
  session_id?: string;
  message_id?: string;
  sources?: any[];
  session_summary?: string;
  memory_used?: { type: string; text: string }[];
}

export type QueryStreamEvent =
  | { type: 'chunk'; text: string }
  | { type: 'meta'; meta: QueryStreamMeta }
  | { type: 'done' }
  | { type: 'error'; detail: string };

export function parseJSONLine(line: string): QueryStreamEvent | null {
  try {
    return JSON.parse(line) as QueryStreamEvent;
  } catch {
    return null;
  }
}

