'use client';

import React from 'react';
import { ThemeToggle } from '@/components/ui/ThemeToggle';

interface SummaryCardProps {
  rightSlot?: React.ReactNode;
}

export function SummaryCard({ rightSlot }: SummaryCardProps) {
  return (
    <div className="border border-border bg-card p-4 md:p-6 cyber-chamfer transition-all duration-150 hover:border-accent/50 hover:shadow-neon-sm">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-label uppercase tracking-wider text-muted-foreground font-heading">
            IOTA KSA
          </p>
          <div className="mt-2 flex items-center gap-2">
            <div className="h-8 w-1 rounded-full bg-accent" />
            <h1 className="font-heading-h1 text-foreground cyber-hero-glitch" id="hero-title">
              Regulation AI
            </h1>
          </div>
          <p className="mt-2 text-sm text-muted-foreground font-mono">
            AI answers with citations from SAMA rulebooks and schemes.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {rightSlot != null ? (
            <div className="hidden md:block">{rightSlot}</div>
          ) : null}
          <ThemeToggle />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <span className="rounded-sm border border-accent/50 bg-accent/10 px-3 py-1 text-xs font-mono font-medium text-accent">
          Live · API healthy
        </span>
      </div>
    </div>
  );
}
