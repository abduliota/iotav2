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
  // Previous implementation performed an O(n^2) search over snippet and
  // answerText, which could easily block the main thread for seconds on
  // large snippets. For responsiveness, highlighting is disabled and the
  // caller falls back to rendering the snippet as plain text.
  if (!snippet?.trim() || !answerText?.trim()) return []
  return []
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
