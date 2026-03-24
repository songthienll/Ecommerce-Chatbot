"""Configuration for Ecommerce Chatbot backend."""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Paths ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # backend/
DATA_DIR = PROJECT_ROOT / "data"

# Load .env from backend root (before using env vars)
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"))
CHROMA_DIR = DATA_DIR / "chroma_db"

# ─── LLM ─────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# ─── Vector DB ───────────────────────────────────────
VECTOR_DB = os.getenv("VECTOR_DB", "qdrant")  # "qdrant" or "chroma"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "tiki_products"
FAQ_COLLECTION_NAME = "tiki_faqs"

# ─── Embedding ───────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ─── RAG ─────────────────────────────────────────────
DEFAULT_TOP_K = 5
SEARCH_MULTIPLIER = 8
KEYWORD_WEIGHT = 0.40

# ─── Cohere re-ranking (optional) ───────────────────
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL: str = "rerank-multilingual-v2.0"
RERANK_TOP_N: int = 5

# ─── Tiki API ───────────────────────────────────────
TIKI_BASE_URL = os.getenv("TIKI_BASE_URL", "https://tiki.vn")
TIKI_API_V2 = f"{TIKI_BASE_URL}/api/v2"
TIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://tiki.vn/",
    "x-guest-token": "8jRvE44nhMhEPqCwpOJ1GSKgyhAfAhJG",
}
