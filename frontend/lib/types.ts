export interface Reference {
  id: string;
  source: string;
  page: number;
  snippet: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  references?: Reference[];
  timestamp: Date;
  /** Backend message_id for feedback (session_messages.message_id). */
  messageId?: string;
  /** Optional session summary text returned with this assistant message. */
  sessionSummary?: string;
  /** Optional list of memory items used when generating this answer (debug/inspection only). */
  memoryUsed?: { type: string; text: string }[];
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  /** Backend session_id; set after first query response. */
  serverSessionId?: string;
}
