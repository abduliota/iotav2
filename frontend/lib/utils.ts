import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Normalize assistant content so inline " * item" list separators become proper
 * markdown list lines. Backend/LLM sometimes outputs "text * Item1 * Item2"
 * on one line; markdown requires each bullet on its own line starting with "* ".
 */
export function normalizeMarkdownLists(text: string): string {
  if (!text || typeof text !== "string") return text
  return text.replace(/\s+\*\s+/g, "\n* ")
}

const SNIPPET_HIGHLIGHT_MIN_LEN = 15
const SNIPPET_HIGHLIGHT_MAX_SNIPPET_LEN = 1500
const SNIPPET_HIGHLIGHT_MAX_ANSWER_LEN = 6000

/**
 * Find ranges in snippet that appear (verbatim, normalized) in answerText.
 * Used to highlight "text we used" in source snippets.
 */
export function getSnippetHighlightRanges(
  snippet: string,
  answerText: string
): { start: number; end: number }[] {
  if (!snippet?.trim() || !answerText?.trim()) return []

  // Protect against very large texts causing long main-thread blocks.
  if (
    snippet.length > SNIPPET_HIGHLIGHT_MAX_SNIPPET_LEN ||
    answerText.length > SNIPPET_HIGHLIGHT_MAX_ANSWER_LEN
  ) {
    return []
  }

  const answerNorm = answerText.replace(/\s+/g, " ").trim().toLowerCase()
  const ranges: { start: number; end: number }[] = []
  for (let start = 0; start < snippet.length; start++) {
    let bestEnd = start
    for (let end = start + SNIPPET_HIGHLIGHT_MIN_LEN; end <= snippet.length; end++) {
      const sub = snippet.slice(start, end).replace(/\s+/g, " ").trim()
      if (sub.length < SNIPPET_HIGHLIGHT_MIN_LEN) continue
      if (answerNorm.includes(sub.toLowerCase())) bestEnd = end
    }
    if (bestEnd > start) ranges.push({ start, end: bestEnd })
  }
  if (ranges.length === 0) return []
  ranges.sort((a, b) => a.start - b.start)
  const merged: { start: number; end: number }[] = [ranges[0]]
  for (let i = 1; i < ranges.length; i++) {
    const last = merged[merged.length - 1]
    if (ranges[i].start <= last.end) {
      last.end = Math.max(last.end, ranges[i].end)
    } else {
      merged.push(ranges[i])
    }
  }
  return merged
}

export type SnippetHighlightSegment = { type: "text" | "highlight"; content: string }

/**
 * Split snippet into text and highlight segments for rendering with <mark>.
 */
export function getSnippetHighlightSegments(
  snippet: string,
  answerText: string
): SnippetHighlightSegment[] {
  const ranges = getSnippetHighlightRanges(snippet, answerText ?? "")
  if (ranges.length === 0) return [{ type: "text", content: snippet }]
  const segments: SnippetHighlightSegment[] = []
  let pos = 0
  for (const r of ranges) {
    if (r.start > pos) segments.push({ type: "text", content: snippet.slice(pos, r.start) })
    segments.push({ type: "highlight", content: snippet.slice(r.start, r.end) })
    pos = r.end
  }
  if (pos < snippet.length) segments.push({ type: "text", content: snippet.slice(pos) })
  return segments
}
