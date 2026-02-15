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
  /** Optional session/conversation id for conversational RAG memory (Phase 4). */
  sessionId?: string | null;
}

export type ConversationState = {
  active_topic?: string;
  active_regulator?: string;
  active_domain?: string;
  language?: string;
  last_intent?: string;
  last_query?: string;
};

export function ChatInterface({ messages, onNewMessage, canSend = true, onLimitReached, sessionId }: ChatInterfaceProps) {
  const [localMessages, setLocalMessages] = useState<Message[]>([]);
  const [conversationState, setConversationState] = useState<ConversationState>({});
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('answer');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const skipSyncRef = useRef(false);

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

    let references: Reference[] = [];

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: content, ...(sessionId && { session_id: sessionId }) }),
      });

      if (!response.ok) {
        throw new Error('API request failed');
      }

      const data = await response.json();
      const answer: string = data.answer ?? '';
      const rawSources: Array<{ document_name?: string; page_start?: number; page_end?: number; snippet?: string; similarity?: number }> = data.sources ?? [];

      references = rawSources.map((s, i) => ({
        id: `${s.document_name ?? ''}-${s.page_start ?? 0}-${s.page_end ?? 0}-${i}`,
        source: s.document_name ?? 'Source',
        page: typeof s.page_start === 'number' ? s.page_start : 0,
        snippet: s.snippet ?? '',
      }));

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: answer,
        references,
        timestamp: new Date(),
      };
      setLocalMessages(prev => [...prev, assistantMessage]);
      skipSyncRef.current = true;
      onNewMessage(assistantMessage);
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
      <div className="flex flex-col flex-1 bg-background text-foreground transition-colors duration-200 rounded-none sm:rounded-2xl border border-border/60 overflow-hidden">
        {/* Chat header - Regulation AI style */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-border">
          <div className="flex flex-col gap-0.5">
            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Conversation
            </span>
            <h2 className="text-xl font-semibold text-foreground md:text-2xl">
              Assistant
            </h2>
            <p className="text-sm text-muted-foreground">
              Responses cite the retrieved passages.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-muted px-3 py-1 text-xs font-medium text-muted-foreground">
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
              <div className="flex flex-1 items-center justify-center p-6">
                <p className="text-center text-sm text-muted-foreground">
                  No messages yet. Ask your first question.
                </p>
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                {allMessages.map((msg, index) => (
                  <AnimatedMessage key={msg.id} message={msg} index={index} />
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
