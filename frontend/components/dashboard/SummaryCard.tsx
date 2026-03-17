'use client';

import React from 'react';
import { ThemeToggle } from '@/components/ui/ThemeToggle';

interface SummaryCardProps {
  rightSlot?: React.ReactNode;
}

export function SummaryCard({ rightSlot }: SummaryCardProps) {
  return (
    <div className="border border-border bg-card p-4 md:p-6 rounded-xl transition-all duration-200 hover:border-border/80 hover:shadow-md shadow-sm">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            IOTA KSA
          </p>
          <div className="mt-2 flex items-center gap-2">
            <div className="h-8 w-1 rounded-full bg-primary" />
            <h1 className="text-2xl font-bold tracking-tight text-foreground" id="hero-title">
              Regulation AI
            </h1>
          </div>
          <p className="mt-2 text-sm text-muted-foreground">
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
        <span className="rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-medium text-success">
          Live · API healthy
        </span>
      </div>
    </div>
  );
}
