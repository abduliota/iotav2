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
        <div className="flex-1 flex items-center gap-2 border border-input bg-background rounded-full pl-4 pr-1 py-1 focus-within:ring-2 focus-within:ring-ring focus-within:border-ring transition-shadow duration-200 shadow-sm">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder="Type your message... (Enter to send, Shift+Enter for newline)"
            className="flex-1 min-w-0 py-2 bg-transparent border-0 resize-none focus:outline-none focus:ring-0 disabled:opacity-50 text-foreground placeholder:text-muted-foreground text-sm"
            rows={1}
          />
        </div>
        <Button onClick={handleSubmit} disabled={disabled || !input.trim()} className="rounded-full px-6">
          Send
        </Button>
      </div>
    </div>
  );
}
