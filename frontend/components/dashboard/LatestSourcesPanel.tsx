'use client';

import React from 'react';
import { Reference } from '@/lib/types';
import { getSnippetHighlightSegments } from '@/lib/utils';

interface LatestSourcesPanelProps {
  references?: Reference[] | null;
  /** Answer text for the message these sources belong to; used to highlight overlap in snippets */
  answerText?: string;
}

export function LatestSourcesPanel({ references, answerText }: LatestSourcesPanelProps) {
  const hasRefs = references && references.length > 0;

  return (
    <div className="flex flex-col flex-1 min-h-0 border border-border bg-card p-4 rounded-xl shadow-sm transition-colors duration-200 md:p-5">
      <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Latest Sources
      </p>
      <h3 className="mt-1 text-lg font-semibold text-foreground">
        Context spotlight
      </h3>
      {hasRefs ? (
        <div className="mt-3 flex-1 min-h-0 overflow-y-auto custom-scroll">
          <div className="space-y-2">
            {references!.map((ref) => {
              const segments = getSnippetHighlightSegments(ref.snippet, answerText ?? '');
              return (
                <div
                  key={ref.id}
                  className="rounded-lg border border-border bg-muted/50 p-3 text-sm"
                >
                  <p className="font-medium text-foreground">{ref.source}</p>
                  <p className="mt-0.5 text-xs text-muted-foreground">Page {ref.page}</p>
                  <p className="mt-1 text-muted-foreground max-h-32 overflow-y-auto custom-scroll">
                    {segments.map((seg, i) =>
                      seg.type === 'highlight' ? (
                        <mark
                          key={i}
                          className="bg-primary/20 text-primary rounded-sm px-0.5"
                        >
                          {seg.content}
                        </mark>
                      ) : (
                        <React.Fragment key={i}>{seg.content}</React.Fragment>
                      )
                    )}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <p className="mt-3 text-sm text-muted-foreground">
          Ask a question to see cited passages.
        </p>
      )}
    </div>
  );
}
