import { useState, useCallback } from 'react'
import { api } from '../services/api'
import type { ChatMessage, Product } from '../types'

export function useChat() {
  const [history, setHistory] = useState<ChatMessage[]>([])
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim()) return

    const userMsg: ChatMessage = { role: 'user', content: text }
    setHistory(prev => [...prev, userMsg])
    setLoading(true)
    setError(null)
    setProducts([])

    try {
      const res = await api.chat(text, history)
      const botMsg: ChatMessage = { role: 'assistant', content: res.answer }
      setHistory(prev => [...prev, botMsg])
      setProducts(res.sources || [])
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Server error'
      setError(msg)
      setHistory(prev => [
        ...prev,
        { role: 'assistant', content: `Error: ${msg}` },
      ])
    } finally {
      setLoading(false)
    }
  }, [history])

  const clearChat = useCallback(() => {
    setHistory([])
    setProducts([])
    setError(null)
  }, [])

  return { history, products, loading, error, sendMessage, clearChat }
}
