import { Chat } from './types';

const STORAGE_KEY = 'ksa_regtech_chats';
const USER_ID_KEY = 'ksa_regtech_user_id';

export function getUserId(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(USER_ID_KEY);
}

export function setUserId(userId: string): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(USER_ID_KEY, userId);
}

export function saveChat(chat: Chat): void {
  const chats = getChats();
  const existingIndex = chats.findIndex(c => c.id === chat.id);
  
  if (existingIndex >= 0) {
    chats[existingIndex] = chat;
  } else {
    chats.push(chat);
  }
  
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}

export function getChats(): Chat[] {
  if (typeof window === 'undefined') return [];
  
  const stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) return [];
  
  try {
    const chats = JSON.parse(stored);
    return chats.map((chat: any) => ({
      ...chat,
      serverSessionId: chat.serverSessionId ?? undefined,
      createdAt: new Date(chat.createdAt),
      updatedAt: new Date(chat.updatedAt),
      messages: chat.messages.map((msg: any) => ({
        ...msg,
        messageId: msg.messageId ?? undefined,
        timestamp: new Date(msg.timestamp),
      })),
    }));
  } catch {
    return [];
  }
}

export function getChat(id: string): Chat | null {
  const chats = getChats();
  return chats.find(c => c.id === id) || null;
}

export function deleteChat(id: string): void {
  const chats = getChats().filter(c => c.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}

export function updateChat(id: string, updates: Partial<Chat>): void {
  const chat = getChat(id);
  if (!chat) return;
  
  const updated = { ...chat, ...updates, updatedAt: new Date() };
  saveChat(updated);
}

export function saveChats(chats: Chat[]): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}
