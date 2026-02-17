'use client';

import React from 'react';
import { Chat } from '@/lib/types';
import { ChatItem } from './ChatItem';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Plus } from 'lucide-react';

interface ChatHistoryProps {
  chats: Chat[];
  selectedChatId: string | null;
  onSelectChat: (chatId: string | null) => void;
}

export function ChatHistory({ chats, selectedChatId, onSelectChat }: ChatHistoryProps) {
  const sortedChats = React.useMemo(
    () => [...chats].sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime()),
    [chats]
  );

  const handleNewChat = () => {
    onSelectChat(null);
  };

  return (
    <div className="flex flex-1 flex-col min-h-0">
      <div className="p-2 shrink-0">
        <Button
          variant="outline"
          onClick={handleNewChat}
          className="w-full justify-start gap-2 rounded-sm px-3 py-2.5"
          aria-label="New chat"
        >
          <Plus className="h-4 w-4 shrink-0" />
          New chat
        </Button>
      </div>
      <ScrollArea className="flex-1 px-2 pb-2 sidebar-scroll">
        <div className="space-y-0.5">
          {sortedChats.map((chat) => (
            <ChatItem
              key={chat.id}
              chat={chat}
              isSelected={selectedChatId === chat.id}
              onClick={() => onSelectChat(chat.id)}
            />
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
