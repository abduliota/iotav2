'use client';

import React, { useEffect, useState } from 'react';
import { ChatInterface } from '../components/chat/ChatInterface';
import { SourcesPanel } from '../components/iotav3/SourcesPanel';
import { useIotav3Chat } from '../hooks/useIotav3Chat';

export default function Page() {
  const { userId, chat, setChat, latestAssistantWithRefs } = useIotav3Chat();

  const [reduceAnimations, setReduceAnimations] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const stored = window.localStorage.getItem('iotav3_reduce_animations');
    if (stored === 'true') {
      setReduceAnimations(true);
    }
  }, []);

  const toggleReduceAnimations = () => {
    setReduceAnimations(prev => {
      const next = !prev;
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(
          'iotav3_reduce_animations',
          next ? 'true' : 'false'
        );
      }
      return next;
    });
  };

  const enabled =
    (process.env.NEXT_PUBLIC_IOTAV3_ENABLE ?? 'true').toLowerCase() !==
    'false';

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <main className="flex flex-1 items-stretch justify-center px-4 py-6 md:px-6">
        <div className="w-full max-w-5xl flex flex-col gap-4">
          <header className="flex items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">
                Saudi GRC Compliance Assistant
              </h1>
              <p className="text-sm text-muted-foreground">
                Ask about SAMA, NORA, Aramco CCC, NCA ECC, PDPL, and ISO 27k.
              </p>
            </div>
            <button
              type="button"
              onClick={toggleReduceAnimations}
              className="text-xs px-3 py-1 rounded-full border border-border text-muted-foreground hover:bg-muted transition-colors"
            >
              {reduceAnimations ? 'Enable animations' : 'Reduce animations'}
            </button>
          </header>

          {!enabled ? (
            <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
              IOTAV3 chat is currently disabled.
            </div>
          ) : (
            <div className="flex flex-1 flex-col gap-4 md:flex-row">
              <div className="md:flex-1 md:min-w-0">
                <ChatInterface
                  initialChat={chat}
                  userId={userId}
                  onChatUpdate={setChat}
                  reduceAnimations={reduceAnimations}
                />
              </div>
              <div className="hidden md:flex md:w-72 lg:w-80 md:flex-col">
                <SourcesPanel references={latestAssistantWithRefs?.references} />
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

