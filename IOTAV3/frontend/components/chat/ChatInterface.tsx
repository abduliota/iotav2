'use client';

import React, { useEffect, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Chat, Message, Reference } from '../../lib/types';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { ThemeToggle } from '../ui/ThemeToggle';
import { AnimatedTypingIndicator } from './AnimatedTypingIndicator';

type ConversationState = {
  active_topic?: string;
  last_query?: string;
};

function parseJSONLine(line: string): any | null {
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

interface ChatInterfaceProps {
  initialChat?: Chat | null;
  onChatUpdate?: (chat: Chat) => void;
  userId?: string | null;
  reduceAnimations?: boolean;
}

export function ChatInterface(props: ChatInterfaceProps) {
  const {
    initialChat = null,
    onChatUpdate,
    userId,
    reduceAnimations = false
  } = props;
  const [chat, setChat] = useState<Chat | null>(initialChat);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationState] = useState<ConversationState>({});
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (chat && onChatUpdate) {
      onChatUpdate(chat);
    }
  }, [chat, onChatUpdate]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chat]);

  const handleSend = async () => {
    const content = input.trim();
    if (!content || isLoading) return;

    const now = new Date();
    const userMessage: Message = {
      id: uuidv4(),
      role: 'user',
      content,
      timestamp: now
    };

    let nextChat: Chat;
    if (!chat) {
      nextChat = {
        id: uuidv4(),
        title: content.slice(0, 48) || 'New chat',
        messages: [userMessage],
        createdAt: now,
        updatedAt: now
      };
    } else {
      nextChat = {
        ...chat,
        messages: [...chat.messages, userMessage],
        updatedAt: now
      };
    }

    setChat(nextChat);
    setInput('');

    setIsLoading(true);

    try {
      const API_URL =
        process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

      const response = await fetch(`${API_URL}/api/query-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: content,
          ...(userId && { user_id: userId }),
          ...(nextChat.serverSessionId && {
            session_id: nextChat.serverSessionId
          })
        })
      });

      if (!response.ok || !response.body) {
        throw new Error('Streaming API request failed');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      const assistantId = uuidv4();
      let accumulated = '';
      let meta:
        | {
            user_id?: string;
            session_id?: string;
            message_id?: string;
            sources?: any[];
            session_summary?: string;
            memory_used?: { type: string; text: string }[];
          }
        | null = null;

      // Insert placeholder assistant message
      setChat(prev => {
        if (!prev) return prev;
        const msg: Message = {
          id: assistantId,
          role: 'assistant',
          content: '',
          timestamp: new Date()
        };
        const updated: Chat = {
          ...prev,
          messages: [...prev.messages, msg],
          updatedAt: new Date()
        };
        return updated;
      });

      let done = false;
      while (!done) {
        const { value, done: streamDone } = await reader.read();
        if (streamDone) break;
        const chunkText = decoder.decode(value, { stream: true });
        const lines = chunkText.split('\n').filter(Boolean);

        for (const line of lines) {
          const evt = parseJSONLine(line);
          if (!evt || typeof evt !== 'object') continue;

          if (evt.type === 'meta') {
            meta = evt.meta || {};
            continue;
          }

          if (evt.type === 'chunk') {
            const text: string = evt.text ?? '';
            accumulated += text;
            setChat(prev => {
              if (!prev) return prev;
              const updatedMessages = prev.messages.map(m =>
                m.id === assistantId ? { ...m, content: accumulated } : m
              );
              const updatedChat: Chat = {
                ...prev,
                messages: updatedMessages,
                updatedAt: new Date()
              };
              return updatedChat;
            });
          }

          if (evt.type === 'done') {
            done = true;
          }

          if (evt.type === 'error') {
            throw new Error(evt.detail || 'Streaming error');
          }
        }
      }

      const rawSources: Array<{
        document_name?: string;
        page_start?: number;
        page_end?: number;
        snippet?: string;
      }> = (meta && Array.isArray(meta.sources) ? (meta.sources as any[]) : []);

      const references: Reference[] = rawSources.map((s, i) => ({
        id: `${s.document_name ?? ''}-${s.page_start ?? 0}-${s.page_end ?? 0}-${i}`,
        source: s.document_name ?? 'Source',
        page: typeof s.page_start === 'number' ? s.page_start : 0,
        snippet: s.snippet ?? ''
      }));

      const sessionSummary = meta?.session_summary;
      const memoryUsed = meta?.memory_used;

      setChat(prev => {
        if (!prev) return prev;
        const updatedMessages = prev.messages.map(m =>
          m.id === assistantId
            ? {
                ...m,
                references,
                ...(meta && meta.message_id && { messageId: meta.message_id }),
                ...(sessionSummary && { sessionSummary }),
                ...(memoryUsed && { memoryUsed })
              }
            : m
        );
        const updatedChat: Chat = {
          ...prev,
          messages: updatedMessages,
          updatedAt: new Date(),
          ...(meta && meta.session_id && {
            serverSessionId: meta.session_id
          })
        };
        return updatedChat;
      });
    } catch (error) {
      // Basic error surface: append an assistant error message.
      const nowError = new Date();
      const errorMessage: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: 'Sorry, the request failed. Please try again.',
        timestamp: nowError
      };
      setChat(prev => {
        if (!prev) return prev;
        const updated: Chat = {
          ...prev,
          messages: [...prev.messages, errorMessage],
          updatedAt: nowError
        };
        return updated;
      });
      // eslint-disable-next-line no-console
      console.error('IOTAV3 chat error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const allMessages = chat?.messages ?? [];

  return (
    <div className="flex h-full flex-col bg-card border border-border rounded-xl shadow-sm">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30">
        <div className="flex flex-col">
          <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Conversation
          </span>
          <h2 className="text-lg font-bold text-foreground">
            IOTAV3 Assistant
          </h2>
        </div>
        <ThemeToggle />
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3 transition-all duration-150">
        {allMessages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            Ask a question about SAMA/NORA regulations to get started.
          </div>
        ) : (
          <>
            {allMessages.map(m => (
              <div
                key={m.id}
                className={`flex ${
                  m.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[90%] text-sm leading-relaxed border px-3 py-2 rounded-2xl shadow-sm transition-transform duration-150 ${
                    m.role === 'user'
                      ? 'bg-accent text-white border-transparent rounded-tr-sm'
                      : 'bg-card text-foreground border-border rounded-tl-sm'
                  }`}
                >
                  {m.role === 'assistant' && !m.content ? (
                    <div
                      className="flex items-center gap-1 py-1"
                      role="status"
                      aria-busy="true"
                      aria-live="polite"
                    >
                      <span className="sr-only">Loading…</span>
                      <span className="loader-dot h-2 w-2 rounded-full bg-muted-foreground/80" />
                      <span className="loader-dot loader-dot-2 h-2 w-2 rounded-full bg-muted-foreground/80" />
                      <span className="loader-dot loader-dot-3 h-2 w-2 rounded-full bg-muted-foreground/80" />
                    </div>
                  ) : (
                    <div className="whitespace-pre-wrap break-words">
                      {m.content}
                    </div>
                  )}
                  {m.references && m.references.length > 0 && (
                    <div className="mt-2 text-xs text-muted-foreground space-y-1">
                      <div className="font-semibold">Sources</div>
                      <ul className="list-disc pl-4 space-y-0.5">
                        {m.references.map(ref => (
                          <li key={ref.id}>
                            <span className="font-mono text-[11px]">
                              {ref.source} (Page {ref.page})
                            </span>
                            {ref.snippet && (
                              <span className="ml-1 text-[11px]">
                                – {ref.snippet}
                              </span>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <AnimatedTypingIndicator reduceAnimations={reduceAnimations} />
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="border-t border-border px-4 py-3">
        <form
          onSubmit={e => {
            e.preventDefault();
            handleSend();
          }}
          className="flex items-center gap-2"
        >
          <Input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask a question about SAMA or NORA..."
            disabled={isLoading}
          />
          <Button type="submit" disabled={isLoading || !input.trim()}>
            Send
          </Button>
        </form>
      </div>
    </div>
  );
}

