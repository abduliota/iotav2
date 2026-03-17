'use client';

import React, { useState, KeyboardEvent, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Send } from 'lucide-react';
import { motion } from 'framer-motion';

interface AnimatedInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  canSend?: boolean;
  onLimitReached?: () => void;
}

export function AnimatedInput({ onSend, disabled = false, canSend = true, onLimitReached }: AnimatedInputProps) {
  const [input, setInput] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Keep the input height stable and rely on internal scrolling for long input
  // to avoid layout shifts on every keystroke.
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  const handleSubmit = () => {
    if (!canSend) {
      onLimitReached?.();
      return;
    }

    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="p-0">
      <div className="flex min-h-[44px] items-center gap-2">
        <div
          className={`flex-1 relative flex min-h-[44px] items-center gap-2 border border-input bg-background rounded-2xl pl-4 pr-1 py-1 focus-within:ring-2 focus-within:ring-ring focus-within:border-ring transition-shadow duration-200 shadow-sm ${!canSend ? 'cursor-pointer' : ''}`}
          onClick={!canSend ? () => onLimitReached?.() : undefined}
        >

          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            disabled={disabled || !canSend}
            placeholder={canSend ? "Ask about the uploaded PDF..." : "Sign up for unlimited prompts"}
            className="flex-1 min-w-0 px-2 py-2 bg-transparent border-0 resize-none focus:outline-none focus:ring-0 disabled:opacity-50 text-foreground placeholder:text-muted-foreground text-sm transition-colors duration-200 h-20 overflow-y-auto custom-scroll"
            rows={3}
          />
          <motion.div
            className="flex items-center self-end mb-0.5"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Button
              onClick={handleSubmit}
              disabled={disabled || !input.trim() || !canSend}
              className="rounded-xl h-10 w-10 p-0 flex items-center justify-center transition-all duration-200 shadow-sm hover:shadow-md"
            >
              <Send className="h-4 w-4" />
              <span className="sr-only">Send</span>
            </Button>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
