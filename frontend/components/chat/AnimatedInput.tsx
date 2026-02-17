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
          className={`flex-1 relative flex min-h-[44px] items-center gap-0 border border-border bg-[var(--input)] cyber-chamfer-sm focus-within:border-accent focus-within:shadow-neon-sm transition-colors transition-shadow duration-150 ${!canSend ? 'cursor-pointer' : ''}`}
          onClick={!canSend ? () => onLimitReached?.() : undefined}
        >
          <span className="pl-3 text-accent font-mono select-none flex items-center" aria-hidden="true">
            &gt;
            {isFocused && <span className="cyber-cursor-blink" />}
          </span>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            disabled={disabled || !canSend}
            placeholder={canSend ? "Ask about the uploaded PDF..." : "Sign up for unlimited prompts"}
            className="flex-1 min-w-0 px-3 py-3 bg-transparent border-0 resize-none focus:outline-none focus:ring-0 disabled:opacity-50 text-foreground placeholder:text-muted-foreground font-mono text-sm tracking-wide transition-colors duration-150 h-20 overflow-y-auto custom-scroll"
            rows={3}
          />
        </div>
        <motion.div
          className="flex items-center"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Button
            onClick={handleSubmit}
            disabled={disabled || !input.trim() || !canSend}
            className="cyber-chamfer h-10 min-h-[44px]"
          >
            <Send className="mr-0 h-4 w-4 sm:mr-2" />
            <span className="hidden sm:inline">Send</span>
          </Button>
        </motion.div>
      </div>
    </div>
  );
}
