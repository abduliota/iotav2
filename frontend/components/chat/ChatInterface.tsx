'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Message, Reference } from '@/lib/types';
import { AnimatedMessage } from './AnimatedMessage';
import { AnimatedInput } from './AnimatedInput';
import { AnimatedTypingIndicator } from './AnimatedTypingIndicator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface ChatInterfaceProps {
  messages: Message[];
  onNewMessage: (message: Message) => void;
  canSend?: boolean;
  onLimitReached?: () => void;
  /** Backend session_id (serverSessionId on Chat). */
  sessionId?: string | null;
  /** Backend user_id for query and feedback. */
  userId?: string | null;
  /** Called when backend returns a new session_id (e.g. first message in a new chat). */
  onSessionId?: (sessionId: string) => void;
}

export type ConversationState = {
  active_topic?: string;
  active_regulator?: string;
  active_domain?: string;
  language?: string;
  last_intent?: string;
  last_query?: string;
};

function parseJSONLine(line: string): any | null {
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

export function ChatInterface({ messages, onNewMessage, canSend = true, onLimitReached, sessionId, userId, onSessionId }: ChatInterfaceProps) {
  const [localMessages, setLocalMessages] = useState<Message[]>([]);
  const [conversationState, setConversationState] = useState<ConversationState>({});
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('answer');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const skipSyncRef = useRef(false);
  const streamingBufferRef = useRef<string>(''); // buffer for streaming text
  const streamingTimerRef = useRef<number | null>(null); // throttle timer id

  useEffect(() => {
    if (skipSyncRef.current) {
      skipSyncRef.current = false;
      return;
    }
    setLocalMessages(messages);
  }, [messages]);

  const handleSend = async (content: string) => {
    if (!canSend) {
      onLimitReached?.();
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setLocalMessages(prev => [...prev, userMessage]);
    skipSyncRef.current = true;
    onNewMessage(userMessage);

    setIsLoading(true);

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/api/query-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: content,
          ...(userId && { user_id: userId }),
          ...(sessionId && { session_id: sessionId }),
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error('Streaming API request failed');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      const assistantId = (Date.now() + 1).toString();
      let accumulated = '';
      let meta:
        | {
            user_id?: string;
            session_id?: string;
            message_id?: string;
            sources?: any[];
            user_id_created?: boolean;
            session_id_created?: boolean;
          }
        | null = null;
      let done = false;

      // Insert placeholder assistant message immediately
      setLocalMessages(prev => [
        ...prev,
        {
          id: assistantId,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
        },
      ]);

      while (!done) {
        const { value, done: streamDone } = await reader.read();
        if (streamDone) {
          break;
        }
        const chunkText = decoder.decode(value, { stream: true });
        const lines = chunkText.split('\n').filter(Boolean);

        for (const line of lines) {
          const evt = parseJSONLine(line);
          if (!evt || typeof evt !== 'object') continue;

          if (evt.type === 'meta') {
            meta = evt.meta || {};
            if (meta.session_id && !sessionId) {
              onSessionId?.(meta.session_id);
            }
            continue;
          }

          if (evt.type === 'chunk') {
            const text: string = evt.text ?? '';
            accumulated += text;
            streamingBufferRef.current = accumulated;

            // Throttle UI updates to avoid re-rendering on every tiny chunk
            if (streamingTimerRef.current === null) {
              streamingTimerRef.current = window.setTimeout(() => {
                const latest = streamingBufferRef.current;
                setLocalMessages(prev =>
                  prev.map(m =>
                    m.id === assistantId ? { ...m, content: latest } : m
                  )
                );
                streamingTimerRef.current = null;
              }, 50);
            }
          }

          if (evt.type === 'done') {
            done = true;
          }

          if (evt.type === 'error') {
            throw new Error(evt.detail || 'Streaming error');
          }
        }
      }

      // Flush any pending throttled update at the end
      if (streamingTimerRef.current !== null) {
        window.clearTimeout(streamingTimerRef.current);
        streamingTimerRef.current = null;
        const latest = streamingBufferRef.current || accumulated;
        setLocalMessages(prev =>
          prev.map(m =>
            m.id === assistantId ? { ...m, content: latest } : m
          )
        );
      }

      const rawSources: Array<{
        document_name?: string;
        page_start?: number;
        page_end?: number;
        snippet?: string;
        similarity?: number;
      }> = (meta && Array.isArray(meta.sources) ? (meta.sources as any[]) : []);

      const references: Reference[] = rawSources.map((s, i) => ({
        id: `${s.document_name ?? ''}-${s.page_start ?? 0}-${s.page_end ?? 0}-${i}`,
        source: s.document_name ?? 'Source',
        page: typeof s.page_start === 'number' ? s.page_start : 0,
        snippet: s.snippet ?? '',
      }));

      setLocalMessages(prev =>
        prev.map(m =>
          m.id === assistantId
            ? {
                ...m,
                references,
                ...(meta && meta.message_id && { messageId: meta.message_id }),
              }
            : m
        )
      );

      skipSyncRef.current = true;
      onNewMessage({
        id: assistantId,
        role: 'assistant',
        content: accumulated,
        references,
        timestamp: new Date(),
        ...(meta && meta.message_id && { messageId: meta.message_id }),
      });

      setIsLoading(false);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, the request failed. Please try again.',
        timestamp: new Date(),
      };
      setLocalMessages(prev => [...prev, errorMessage]);
      skipSyncRef.current = true;
      onNewMessage(errorMessage);
      setIsLoading(false);
    }
  };

  const allMessages = [...localMessages];

  return (
    <div className="flex h-full">
      <div className="flex flex-col flex-1 bg-card text-foreground transition-colors duration-200 border border-border overflow-hidden cyber-chamfer-sm">
        {/* Chat header - cyberpunk style */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-border">
          <div className="flex flex-col gap-0.5 leading-tight">
            <span className="text-label font-heading text-muted-foreground uppercase tracking-wider">
              Conversation
            </span>
            <h2 className="font-heading-h2 text-foreground">
              Assistant
            </h2>
            <p className="text-sm text-muted-foreground font-mono">
              Responses cite the retrieved passages.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="rounded-sm border border-border bg-muted px-3 py-1 text-xs font-mono uppercase tracking-wider text-muted-foreground">
              {allMessages.length} messages
            </span>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="hidden">
              <TabsList className="bg-muted/60 h-8 px-1 rounded-full">
                <TabsTrigger value="answer" className="rounded-full px-3 py-1 text-xs">
                  Answer
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </div>

        {/* Messages area */}
        <div className="flex-1 min-h-0">
          <ScrollArea className="h-full px-3 sm:px-4 lg:px-6 py-4">
            {allMessages.length === 0 ? (
              <div className="flex min-h-[50vh] flex-1 items-center justify-center p-6 md:min-h-0">
                <p className="text-center text-sm text-muted-foreground">
                  No messages yet. Ask your first question.
                </p>
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                {allMessages.map((msg, index) => (
                  <AnimatedMessage
                    key={msg.id}
                    message={msg}
                    index={index}
                    userId={userId ?? undefined}
                    sessionId={sessionId ?? undefined}
                  />
                ))}
                {isLoading && <AnimatedTypingIndicator />}
                <div ref={messagesEndRef} />
              </div>
            )}
          </ScrollArea>
        </div>

        {/* Input bar - Regulation AI style */}
        <div className="border-t border-border bg-card px-4 py-4">
          <AnimatedInput
            onSend={handleSend}
            disabled={isLoading}
            canSend={canSend}
            onLimitReached={onLimitReached}
          />
        </div>
      </div>
    </div>
  );
}
