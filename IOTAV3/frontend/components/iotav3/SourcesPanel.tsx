import React from 'react';
import type { Reference } from '../../lib/types';

interface SourcesPanelProps {
  references?: Reference[] | null;
}

export function SourcesPanel({ references }: SourcesPanelProps) {
  if (!references || references.length === 0) {
    return (
      <div className="h-full rounded-xl border border-dashed border-border bg-muted/40 px-3 py-3 text-xs text-muted-foreground">
        <div className="font-semibold mb-1">Sources</div>
        <p>No sources available for this answer yet.</p>
      </div>
    );
  }

  return (
    <div className="h-full rounded-xl border border-border bg-card px-3 py-3 text-xs">
      <div className="font-semibold mb-2 text-foreground">Sources</div>
      <ul className="space-y-1.5">
        {references.map(ref => (
          <li key={ref.id} className="border-b border-border/40 pb-1.5 last:border-b-0">
            <div className="font-mono text-[11px] text-foreground">
              {ref.source} (Page {ref.page})
            </div>
            {ref.snippet && (
              <div className="mt-0.5 text-[11px] text-muted-foreground line-clamp-3">
                {ref.snippet}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

