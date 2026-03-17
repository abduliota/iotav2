'use client';

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface AnimatedTypingIndicatorProps {
  reduceAnimations?: boolean;
}

export function AnimatedTypingIndicator({
  reduceAnimations = false
}: AnimatedTypingIndicatorProps) {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mq.matches);

    const handleChange = () => {
      setPrefersReducedMotion(mq.matches);
    };

    mq.addEventListener('change', handleChange);
    return () => mq.removeEventListener('change', handleChange);
  }, []);

  const disableAnimation = reduceAnimations || prefersReducedMotion;

  return (
    <div className="flex justify-start mb-3">
      <div className="inline-flex items-center gap-1.5 border border-border bg-card px-3 py-2 rounded-2xl shadow-sm">
        {[0, 1, 2].map(index => (
          <motion.div
            // eslint-disable-next-line react/no-array-index-key
            key={index}
            className="h-1.5 w-1.5 rounded-full bg-accent"
            animate={
              disableAnimation
                ? undefined
                : { y: [0, -3, 0], opacity: [0.4, 1, 0.4] }
            }
            transition={
              disableAnimation
                ? undefined
                : {
                    duration: 0.6,
                    repeat: Infinity,
                    delay: index * 0.15,
                    ease: 'easeInOut'
                  }
            }
          />
        ))}
      </div>
    </div>
  );
}

