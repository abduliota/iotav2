'use client';

import React, { useState, KeyboardEvent } from 'react';
import { Button } from '@/components/ui/button';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, disabled = false }: ChatInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = () => {
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-border p-4 bg-card transition-colors duration-200">
      <div className="flex gap-2 items-center">
        <div className="flex-1 flex items-center gap-0 border border-border bg-[var(--input)] cyber-chamfer-sm focus-within:border-accent focus-within:shadow-neon-sm transition-all duration-150">
          <span className="pl-3 text-accent font-mono select-none" aria-hidden="true">
            &gt;
          </span>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder="Type your message... (Enter to send, Shift+Enter for newline)"
            className="flex-1 min-w-0 pl-2 pr-4 py-2 bg-transparent border-0 resize-none focus:outline-none focus:ring-0 disabled:opacity-50 text-foreground placeholder:text-muted-foreground font-mono text-sm tracking-wide"
            rows={1}
          />
        </div>
        <Button onClick={handleSubmit} disabled={disabled || !input.trim()}>
          Send
        </Button>
      </div>
    </div>
  );
}
