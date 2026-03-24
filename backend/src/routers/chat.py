"""
Chat router — /api/chat and /api/products endpoints.
"""

from typing import Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import TIKI_API_V2, TIKI_HEADERS, TIKI_BASE_URL
from ..pipeline import answer as rag_answer
from ..retriever import count_chunks

router = APIRouter(prefix="/api", tags=["chat"])


# ─── Request / Response models ──────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatSource(BaseModel):
    product_name: str
    category: str
    price: float
    original_price: float
    url: str
    thumbnail_url: str
    rating: float
    review_count: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]


class ProductQuery(BaseModel):
    q: str
    limit: int = 10


class ProductItem(BaseModel):
    id: int
    name: str
    price: float
    original_price: Optional[float]
    url: str
    thumbnail_url: str
    rating_average: float
    review_count: int


class ProductResponse(BaseModel):
    products: list


# ─── Endpoints ───────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    RAG-powered chat endpoint.
    Receives {message, history} → returns {answer, sources: [product]}.
    """
    result = rag_answer(req.message, top_k=5)

    sources = [
        ChatSource(
            product_name=s["product_name"],
            category=s["category"],
            price=s["price"],
            original_price=s.get("original_price") or 0,
            url=s["url"],
            thumbnail_url=s.get("thumbnail_url", ""),
            rating=s.get("rating") or 0,
            review_count=s.get("review_count") or 0,
        )
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=sources,
    )


@router.get("/products")
async def get_products(q: str = "", limit: int = 10):
    """
    Proxy Tiki API for product search.
    GET /api/products?q=tai+nghe&limit=10
    """
    if not q:
        return {"products": []}

    try:
        resp = requests.get(
            f"{TIKI_API_V2}/products",
            params={"q": q, "limit": limit},
            headers=TIKI_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        products = []
        for item in data.get("data", []):
            products.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "price": item.get("price"),
                "original_price": item.get("original_price"),
                "url": f"{TIKI_BASE_URL}/{item.get('url_path', '')}",
                "thumbnail_url": item.get("thumbnail_url", ""),
                "rating_average": item.get("rating_average", 0),
                "review_count": item.get("review_count", 0),
            })

        return {"products": products}

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Tiki API error: {e}")


@router.get("/health")
async def health():
    """Health check — returns status and chunk count."""
    try:
        n = count_chunks()
    except Exception:
        n = 0
    return {"status": "ok", "chunks": n}
