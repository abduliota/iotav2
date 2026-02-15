'use client';

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Fingerprint } from 'lucide-react';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  onRegister: (email: string) => Promise<{ success: boolean; error?: string }>;
  onLogin: (email: string) => Promise<{ success: boolean; error?: string }>;
}

export function AuthModal({ isOpen, onClose, onSuccess, onRegister, onLogin }: AuthModalProps) {
  const [mode, setMode] = useState<'signup' | 'login'>('signup');
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) {
      setError('Please enter your email');
      return;
    }

    setIsLoading(true);
    setError('');

    const result = mode === 'signup'
      ? await onRegister(email.trim())
      : await onLogin(email.trim());

    setIsLoading(false);

    if (result.success) {
      onSuccess();
      onClose();
      setEmail('');
    } else {
      setError(result.error || 'Authentication failed');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
      <div className="bg-card border border-border cyber-chamfer p-6 w-full max-w-md shadow-neon-sm">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-heading font-semibold uppercase tracking-wider text-foreground">
            {mode === 'signup' ? 'Sign Up' : 'Login'}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card rounded-sm"
            aria-label="Close modal"
          >
            ✕
          </button>
        </div>

        <p className="text-sm text-muted-foreground mb-6 font-mono">
          {mode === 'signup'
            ? 'Create an account with fingerprint authentication for unlimited prompts'
            : 'Login with your fingerprint to access unlimited prompts'}
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="auth-email" className="block text-label text-muted-foreground mb-2">
              Email
            </label>
            <div className="flex border border-border bg-[var(--input)] cyber-chamfer-sm focus-within:border-accent focus-within:shadow-neon-sm transition-all duration-150">
              <span className="pl-3 text-accent font-mono select-none self-center" aria-hidden="true">
                &gt;
              </span>
              <input
                id="auth-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                className="flex-1 min-w-0 pl-2 pr-4 py-2 bg-transparent border-0 text-foreground placeholder:text-muted-foreground font-mono text-sm tracking-wide focus:outline-none focus:ring-0 disabled:opacity-50"
                disabled={isLoading}
              />
            </div>
          </div>

          {error && (
            <div className="text-sm text-destructive bg-destructive/10 border border-destructive/30 rounded-sm p-3 font-mono">
              {error}
            </div>
          )}

          <Button
            type="submit"
            disabled={isLoading || !email.trim()}
            className="w-full"
          >
            <Fingerprint className="h-4 w-4 mr-2" />
            {isLoading
              ? 'Processing...'
              : mode === 'signup'
              ? 'Sign Up with Fingerprint'
              : 'Login with Fingerprint'}
          </Button>
        </form>

        <div className="mt-4 text-center">
          <button
            type="button"
            onClick={() => {
              setMode(mode === 'signup' ? 'login' : 'signup');
              setError('');
            }}
            className="text-sm text-accent hover:text-accent/90 transition-colors font-mono focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-sm"
          >
            {mode === 'signup'
              ? 'Already have an account? Login'
              : "Don't have an account? Sign Up"}
          </button>
        </div>
      </div>
    </div>
  );
}
