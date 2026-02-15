'use client';

import React from 'react';

interface PromptCounterProps {
  remaining: number;
  total: number;
  isAuthenticated: boolean;
}

export function PromptCounter({ remaining, total, isAuthenticated }: PromptCounterProps) {
  if (isAuthenticated) {
    return (
      <div className="inline-flex h-10 min-h-[44px] items-center px-3 rounded-sm border border-accent/50 bg-accent/10 text-xs font-mono font-medium text-accent transition-colors duration-200">
        Unlimited
      </div>
    );
  }

  const percentage = (remaining / total) * 100;
  const isWarning = remaining <= 2;

  return (
    <div className="flex h-10 min-h-[44px] items-center justify-center gap-2">
      <div className="text-xs font-mono text-muted-foreground">
        {remaining}/{total} prompts
      </div>
      <div className="w-20 h-1.5 rounded-sm overflow-hidden bg-muted transition-colors duration-200">
        <div
          className={`h-full transition-all duration-300 ${
            isWarning ? 'bg-destructive' : 'bg-accent'
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
