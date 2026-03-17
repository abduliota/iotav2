'use client';

import React, { useState, useEffect } from 'react';
import { Sun, Moon } from 'lucide-react';

const THEME_KEY = 'theme';

export function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setIsDark(document.documentElement.classList.contains('dark'));
  }, []);

  const toggle = () => {
    const nextDark = !document.documentElement.classList.contains('dark');
    if (nextDark) {
      document.documentElement.classList.add('dark');
      try {
        localStorage.setItem(THEME_KEY, 'dark');
      } catch {
        /* ignore */
      }
    } else {
      document.documentElement.classList.remove('dark');
      try {
        localStorage.setItem(THEME_KEY, 'light');
      } catch {
        /* ignore */
      }
    }
    setIsDark(nextDark);
  };

  if (!mounted) {
    return (
      <div
        className="flex h-10 w-10 min-h-[44px] min-w-[44px] shrink-0 items-center justify-center rounded-md border border-border"
        aria-hidden
      />
    );
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="flex h-10 w-10 min-h-[44px] min-w-[44px] shrink-0 items-center justify-center rounded-md border border-border text-muted-foreground transition-colors duration-150 hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background"
    >
      {isDark ? (
        <Sun className="h-4 w-4" aria-hidden />
      ) : (
        <Moon className="h-4 w-4" aria-hidden />
      )}
    </button>
  );
}
