# Ecommerce Chatbot

Vietnamese RAG chatbot for Tiki product search — React + FastAPI + Qdrant.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript + Vite |
| Backend | FastAPI + Python |
| Vector DB | Qdrant |
| LLM | Groq (llama-3.3-70b) |
| Embeddings | sentence-transformers (paraphrase-multilingual-mpnet-base-v2) |

## Architecture

```
User Query (Vietnamese)
        │
        ▼
   FastAPI /api/chat
        │
        ▼
  SentenceTransformer
  (embed query)
        │
        ▼
  Qdrant hybrid search
  (vector + keyword scoring)
        │
        ▼
  Groq LLM
  (llama-3.3-70b-versatile)
        │
        ▼
  Vietnamese answer + product cards
```

## Quick Start

### 1. Clone & setup

```bash
git clone https://github.com/songthienll/Ecommerce-Chatbot.git
cd Ecommerce-Chatbot
cp .env.example .env
```

### 2. Fill `.env`

```env
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...        # optional
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.3-70b-versatile
VECTOR_DB=qdrant
QDRANT_URL=http://localhost:6333
```

### 3. Docker Compose (recommended)

```bash
docker compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
# Qdrant:   http://localhost:6333
```

### 4. Local development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn src.main:app --port 8000 --reload

# Frontend
cd frontend
npm install
npm run dev
```

### 5. Seed vector data (133,400 Tiki chunks)

Data is stored in Qdrant. On first deploy, either:
- **Restore** from a Qdrant backup snapshot
- **Re-ingest** from `backend/src/ingest.py`

```bash
cd backend
python -m src.migrate_to_qdrant  # migrate from existing ChromaDB
```

## Project Structure

```
backend/
├── src/
│   ├── main.py          # FastAPI app
│   ├── config.py         # env vars
│   ├── retriever.py      # hybrid search (Qdrant + ChromaDB)
│   ├── pipeline.py        # RAG orchestration
│   ├── generator.py      # Groq / Ollama LLM
│   ├── routers/chat.py   # /api/chat, /api/products, /api/health
│   └── migrate_to_qdrant.py
├── Dockerfile
└── requirements.txt

frontend/
├── src/
│   ├── App.tsx          # main layout
│   ├── components/        # ProductCard, etc.
│   ├── hooks/useChat.ts   # chat state
│   └── services/api.ts    # Axios API client
├── Dockerfile            # Node build + nginx
├── nginx.conf            # /api proxy
└── vite.config.ts       # Vite proxy to backend
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat` | RAG chat — `{message, history}` → `{answer, sources}` |
| GET | `/api/products` | Tiki API proxy — `?q=...&limit=10` |
| GET | `/api/health` | Health check — chunk count |

## Configuration

### Vector DB switch

```env
# Use Qdrant (default)
VECTOR_DB=qdrant

# Use ChromaDB
VECTOR_DB=chroma
```

### LLM switch

```env
LLM_PROVIDER=groq         # Groq (default, fast)
LLM_PROVIDER=ollama       # Ollama local
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

## Deployment

### Docker Compose (production)

```bash
docker compose -f docker-compose.yml up -d
```

### Railway / Render / Fly.io

1. Set env vars in dashboard
2. Point `docker-compose.yml` to your Qdrant snapshot or re-ingest on startup
3. Build: `docker compose up --build`

## CI/CD

GitHub Actions triggers on every push to `master`:
- `backend-test` — ruff lint + pytest
- `frontend-test` — typecheck + build
- `docker-build` — build both images

## Evaluation Metrics (latest)

Baseline on real Tiki index (133,400 chunks):
- Pass rate: **60.0%** (18/30)
- By type: happy 50% | hard_negative 90% | mixed 40%

After retrieval + gating fixes:
- Pass rate: **76.67%** (23/30)
- By type: happy 90% | hard_negative 90% | mixed 50%

Extra quality signals:
- Backend unit tests: **18 passed**
- Active vector collection: `tiki_products` (**133,400 chunks**)

## License

MIT
