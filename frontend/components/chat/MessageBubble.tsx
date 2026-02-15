'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Message } from '@/lib/types';
import { normalizeMarkdownLists } from '@/lib/utils';
import { References } from './References';
import { Button } from '@/components/ui/button';
import { Download, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';

interface MessageBubbleProps {
  message: Message;
  userId?: string;
  sessionId?: string;
}

export function MessageBubble({ message, userId, sessionId }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const [showSources, setShowSources] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState<0 | 1 | null>(null);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const canSendFeedback = !isUser && Boolean(message.messageId && userId && sessionId);

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
        className={`max-w-[90%] sm:max-w-[80%] text-sm leading-relaxed transition-all duration-200 ${
          isUser
            ? 'bg-primary text-primary-foreground rounded-2xl rounded-br-sm px-3 py-2.5'
            : 'bg-muted/70 text-foreground rounded-2xl rounded-bl-sm px-3 py-2.5'
        }`}
      >
        {isUser ? (
          <div className="whitespace-pre-wrap break-words">
            {message.content}
          </div>
        ) : (
          <>
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
            {message.references && message.references.length > 0 ? (
              <>
                <div className="mt-2 flex items-center gap-2">
                  <button
                    type="button"
                    className={`text-xs px-2 py-1 rounded-full border transition-colors ${
                      !showSources
                        ? 'bg-muted text-foreground border-border'
                        : 'bg-background text-foreground border-border'
                    }`}
                    onClick={() => setShowSources(false)}
                  >
                    Answer
                  </button>
                  <button
                    type="button"
                    className={`text-xs px-2 py-1 rounded-full border transition-colors ${
                      showSources
                        ? 'bg-muted text-foreground border-border'
                        : 'bg-background text-foreground border-border'
                    }`}
                    onClick={() => setShowSources(true)}
                  >
                    Sources ({message.references.length})
                  </button>
                </div>
                {showSources && (
                  <div className="mt-3">
                    <References references={message.references} />
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
                <span className="flex items-center gap-1 text-muted-foreground">
                  <span className="text-[11px] mr-1">Helpful?</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className={`text-[11px] h-7 px-2 ${feedbackSent === 1 ? 'text-green-600' : ''}`}
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
                            feedback: 1,
                            comments: undefined,
                          }),
                        });
                        setFeedbackSent(1);
                      } catch (e) {
                        console.error('Feedback failed:', e);
                      } finally {
                        setFeedbackSubmitting(false);
                      }
                    }}
                  >
                    <ThumbsUp className="h-3 w-3 mr-1" />
                    Yes
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className={`text-[11px] h-7 px-2 ${feedbackSent === 0 ? 'text-red-600' : ''}`}
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
                            feedback: 0,
                            comments: undefined,
                          }),
                        });
                        setFeedbackSent(0);
                      } catch (e) {
                        console.error('Feedback failed:', e);
                      } finally {
                        setFeedbackSubmitting(false);
                      }
                    }}
                  >
                    <ThumbsDown className="h-3 w-3 mr-1" />
                    No
                  </Button>
                  {feedbackSent !== null && (
                    <span className="text-[11px] text-muted-foreground">Thanks</span>
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
