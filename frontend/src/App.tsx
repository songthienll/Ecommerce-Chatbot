import { useState, useRef, useEffect } from 'react'
import { useChat } from './hooks/useChat'
import { ProductCard } from './components/ProductCard'
import { api } from './services/api'

const SUGGESTIONS = [
  'Tai nghe bluetooth duoi 500k',
  'Tu van ban phim wireless',
  'May loc nuoc gia dinh 4 nguoi',
  'Laptop ASUS gia tot',
]

export default function App() {
  const { history, products, loading, sendMessage, clearChat } = useChat()
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history, products])

  const handleSend = () => {
    if (!input.trim()) return
    sendMessage(input)
    setInput('')
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>Ecommerce Chatbot</h1>
          <p>RAG-powered shopping assistant</p>
        </div>
        <button className="btn-clear" onClick={clearChat}>
          Clear chat
        </button>
        <div className="sidebar-suggestions">
          <p className="label">Suggestions</p>
          {SUGGESTIONS.map(s => (
            <button key={s} className="chip" onClick={() => sendMessage(s)}>
              {s}
            </button>
          ))}
        </div>
        <div className="sidebar-health">
          <button className="btn-health" onClick={async () => {
            try {
              const h = await api.health()
              alert(`OK — ${h.chunks?.toLocaleString()} chunks indexed`)
            } catch {
              alert('Backend offline')
            }
          }}>
            Health check
          </button>
        </div>
      </aside>

      <main className="main">
        <div className="chat-area">
          {history.length === 0 && (
            <div className="empty">
              <div className="empty-icon">🛍️</div>
              <p>Ask me anything about products!</p>
            </div>
          )}
          {history.map((msg, i) => (
            <div key={i} className={`bubble-row ${msg.role}`}>
              <div className={`bubble ${msg.role}`}>{msg.content}</div>
            </div>
          ))}
          {loading && (
            <div className="bubble-row assistant">
              <div className="bubble typing">...</div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {products.length > 0 && (
          <div className="product-section">
            <p className="section-title">Products</p>
            <div className="product-grid">
              {products.map((p, i) => (
                <ProductCard key={i} product={p} />
              ))}
            </div>
          </div>
        )}

        <div className="input-area">
          <input
            className="chat-input"
            placeholder="Ask about products..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            disabled={loading}
          />
          <button className="btn-send" onClick={handleSend} disabled={loading}>
            Send
          </button>
        </div>
      </main>
    </div>
  )
}
