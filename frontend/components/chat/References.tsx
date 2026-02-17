'use client';

import React, { useState } from 'react';
import { Reference } from '@/lib/types';
import { getSnippetHighlightSegments } from '@/lib/utils';

interface ReferencesProps {
  references: Reference[];
  /** Answer text for this message; used to highlight verbatim overlap in snippets */
  answerText?: string;
}

export function References({ references, answerText }: ReferencesProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  if (!references || references.length === 0) return null;

  const selected = references[selectedIndex];

  const selectedSnippet = selected.snippet ?? '';
  const safeAnswerText = answerText ?? '';

  // Extra guard: avoid running expensive highlighting on very large snippets.
  const shouldHighlight =
    safeAnswerText.length > 0 && selectedSnippet.length <= 1500;

  const segments = shouldHighlight
    ? getSnippetHighlightSegments(selectedSnippet, safeAnswerText)
    : [{ type: 'text', content: selectedSnippet }];

  return (
    <div className="mt-2">
      <div className="flex flex-wrap gap-2 mb-2">
        {references.map((ref, i) => (
          <button
            key={ref.id}
            type="button"
            onClick={() => setSelectedIndex(i)}
            className={`text-xs font-mono px-2 py-1 rounded-sm border transition-colors duration-150 ${
              i === selectedIndex
                ? 'bg-muted text-foreground border-accent'
                : 'bg-card text-muted-foreground border-border hover:text-foreground hover:border-accent/50'
            }`}
          >
            Page {ref.page}
          </button>
        ))}
      </div>
      <div className="p-3 cyber-chamfer-sm border border-border bg-card text-xs font-mono">
        <div className="font-semibold text-foreground">
          {selected.source}
        </div>
        <div className="text-muted-foreground mt-0.5">
          Page {selected.page}
        </div>
        <div className="text-muted-foreground mt-1 whitespace-pre-wrap break-words">
          {segments.map((seg, i) =>
            seg.type === 'highlight' ? (
              <mark
                key={i}
                className="bg-accent/30 text-accent rounded-sm px-0.5"
              >
                {seg.content}
              </mark>
            ) : (
              <React.Fragment key={i}>{seg.content}</React.Fragment>
            )
          )}
        </div>
      </div>
    </div>
  );
}
