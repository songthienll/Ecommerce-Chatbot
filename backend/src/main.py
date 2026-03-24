"""
Ecommerce Chatbot FastAPI backend entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers import chat
from src.logger import setup_logger

logger = setup_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ChromaDB + embedder lazily at startup."""
    logger.info("Ecommerce Chatbot backend starting...")
    logger.info("ChromaDB + embedder will load on first /api/chat request (lazy)")
    yield
    logger.info("Ecommerce Chatbot backend shutting down...")


app = FastAPI(
    title="Ecommerce Chatbot API",
    description="RAG e-commerce chatbot for Tiki",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)


@app.get("/")
async def root():
    return {
        "name": "Ecommerce Chatbot",
        "version": "1.0.0",
        "docs": "/docs",
    }
