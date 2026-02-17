'use client';

import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Message } from '@/lib/types';
import { normalizeMarkdownLists } from '@/lib/utils';
import { References } from './References';
import { Button } from '@/components/ui/button';
import { Download, Copy, Star } from 'lucide-react';

interface MessageBubbleProps {
  message: Message;
  userId?: string;
  sessionId?: string;
}

function MessageBubbleComponent({ message, userId, sessionId }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const [showSources, setShowSources] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState<number | null>(null);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const canSendFeedback = !isUser && Boolean(message.messageId && userId && sessionId);
  const isStreamingAssistant =
    !isUser && (!message.references || message.references.length === 0);
  const [enableMarkdown, setEnableMarkdown] = useState(false);

  // Defer heavy markdown rendering slightly so the UI can show the full
  // plain-text answer immediately without feeling frozen on large responses.
  useEffect(() => {
    // While streaming, always use the lightweight view.
    if (isStreamingAssistant) {
      setEnableMarkdown(false);
      return;
    }

    // When streaming finishes (references appear), first show plain text,
    // then upgrade to markdown after a short delay.
    let timeoutId: number | null = window.setTimeout(() => {
      setEnableMarkdown(true);
    }, 150);

    return () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [isStreamingAssistant, message.content]);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(message.content);
  };

  const downloadMessage = () => {
    const blob = new Blob([message.content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `message-${message.id}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      <div
        className={`max-w-[90%] sm:max-w-[80%] text-sm leading-relaxed transition-all duration-150 border border-border cyber-chamfer-sm px-3 py-2.5 ${
          isUser
            ? 'bg-muted text-foreground border-accent/50'
            : 'bg-card text-foreground'
        }`}
      >
        {isUser ? (
          <div className="whitespace-pre-wrap break-words">
            {message.content}
          </div>
        ) : (
          <>
            {isStreamingAssistant || !enableMarkdown ? (
              // Lightweight view while streaming and immediately after completion
              // (before markdown is enabled) to avoid heavy renders on large answers.
              <div className="whitespace-pre-wrap break-words font-mono">
                {message.content}
              </div>
            ) : (
              <div className="chat-markdown prose prose-sm max-w-none prose-invert prose-pre:bg-transparent prose-pre:p-0 prose-code:text-xs prose-headings:font-semibold prose-headings:tracking-tight prose-p:leading-relaxed prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5 prose-strong:font-semibold prose-ul:list-disc prose-ul:pl-5 prose-ol:list-decimal prose-ol:pl-5">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    ul: ({ node, ...props }) => <ul className="list-disc pl-5 my-2" {...props} />,
                    ol: ({ node, ...props }) => <ol className="list-decimal pl-5 my-2" {...props} />,
                    li: ({ node, ...props }) => <li className="my-0.5" {...props} />,
                  }}
                >
                  {normalizeMarkdownLists(message.content)}
                </ReactMarkdown>
              </div>
            )}
            {message.references && message.references.length > 0 ? (
              <>
                <div className="mt-2 flex items-center gap-2">
                  <button
                    type="button"
                    className={`text-xs font-mono px-2 py-1 rounded-sm border transition-colors duration-150 ${
                      !showSources
                        ? 'bg-muted text-foreground border-accent'
                        : 'bg-card text-muted-foreground border-border hover:border-accent/50'
                    }`}
                    onClick={() => setShowSources(false)}
                  >
                    Answer
                  </button>
                  <button
                    type="button"
                    className={`text-xs font-mono px-2 py-1 rounded-sm border transition-colors duration-150 ${
                      showSources
                        ? 'bg-muted text-foreground border-accent'
                        : 'bg-card text-muted-foreground border-border hover:border-accent/50'
                    }`}
                    onClick={() => setShowSources(true)}
                  >
                    Sources ({message.references.length})
                  </button>
                </div>
                {showSources && (
                  <div className="mt-3">
                    <References references={message.references} answerText={message.content} />
                  </div>
                )}
              </>
            ) : (
              <div className="mt-2 text-xs text-muted-foreground">
                No sources used
              </div>
            )}
            <div className="mt-2 flex flex-wrap gap-1.5 items-center">
              <Button
                variant="ghost"
                size="sm"
                onClick={downloadMessage}
                className="text-[11px] h-7 px-2"
              >
                <Download className="h-3 w-3 mr-1" />
                Download
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={copyToClipboard}
                className="text-[11px] h-7 px-2"
              >
                <Copy className="h-3 w-3 mr-1" />
                Copy
              </Button>
              {canSendFeedback && (
                <span className="flex items-center gap-1 text-muted-foreground" role="group" aria-label="Rate this response 1 to 5 stars">
                  <span className="text-[11px] mr-1">Rate:</span>
                  {[1, 2, 3, 4, 5].map((rating) => (
                    <button
                      key={rating}
                      type="button"
                      disabled={feedbackSubmitting || feedbackSent !== null}
                      onClick={async () => {
                        setFeedbackSubmitting(true);
                        try {
                          const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
                          await fetch(`${API_URL}/api/feedback`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                              session_id: sessionId,
                              user_id: userId,
                              message_id: message.messageId,
                              feedback: rating,
                            }),
                          });
                          setFeedbackSent(rating);
                        } catch (e) {
                          console.error('Feedback failed:', e);
                        } finally {
                          setFeedbackSubmitting(false);
                        }
                      }}
                      className="p-0.5 rounded-sm focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 disabled:opacity-50 disabled:pointer-events-none"
                      aria-label={`${rating} star${rating === 1 ? '' : 's'}`}
                      aria-pressed={feedbackSent !== null && feedbackSent >= rating}
                    >
                      <Star
                        className={`h-4 w-4 ${
                          feedbackSent !== null
                            ? rating <= feedbackSent
                              ? 'fill-accent text-accent'
                              : 'text-muted-foreground'
                            : 'text-muted-foreground hover:text-accent'
                        }`}
                      />
                    </button>
                  ))}
                  {feedbackSent !== null && (
                    <span className="text-[11px] text-muted-foreground ml-0.5">Thanks</span>
                  )}
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export const MessageBubble = React.memo(MessageBubbleComponent);
