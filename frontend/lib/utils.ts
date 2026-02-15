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
