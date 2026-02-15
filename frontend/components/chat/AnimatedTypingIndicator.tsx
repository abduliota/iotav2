'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

export function AnimatedTypingIndicator() {
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduceMotion(mq.matches);
    const handler = () => setReduceMotion(mq.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  return (
    <div className="flex justify-start mb-3">
      <div className="inline-flex items-center gap-1.5 cyber-chamfer-sm border border-border bg-card px-3 py-2">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="w-1.5 h-1.5 bg-accent rounded-full"
            animate={reduceMotion ? undefined : { y: [0, -3, 0], opacity: [0.4, 1, 0.4] }}
            transition={
              reduceMotion
                ? undefined
                : {
                    duration: 0.6,
                    repeat: Infinity,
                    delay: i * 0.15,
                    ease: 'easeInOut',
                  }
            }
          />
        ))}
      </div>
    </div>
  );
}
