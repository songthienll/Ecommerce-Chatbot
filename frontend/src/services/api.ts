import axios from 'axios';
import type { ChatMessage, ChatResponse } from '../types';

// Uses Vite proxy — /api -> http://localhost:8000
const BASE = '/api';

export const api = {
  async chat(message: string, history: ChatMessage[]): Promise<ChatResponse> {
    const res = await axios.post<ChatResponse>(
      `${BASE}/chat`,
      { message, history },
      { timeout: 30_000 }
    );
    return res.data;
  },

  async products(q: string, limit = 10) {
    const res = await axios.get(`${BASE}/products`, {
      params: { q, limit },
      timeout: 10_000,
    });
    return res.data;
  },

  async health() {
    const res = await axios.get(`${BASE}/health`);
    return res.data;
  },
};
