"""
Retriever — hybrid search (vector + keyword overlap) with weighted score fusion.
Supports both ChromaDB and Qdrant vector stores.

Usage:
    from src.retriever import retrieve
    results = retrieve("tai nghe khong day duoi 500k", top_k=5)
"""

import re
from typing import Optional, Tuple

import requests
import chromadb
import qdrant_client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import (
    CHROMA_DIR,
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    COLLECTION_NAME,
    FAQ_COLLECTION_NAME,
    QDRANT_URL,
    RERANK_TOP_N,
    SEARCH_MULTIPLIER,
    KEYWORD_WEIGHT,
    EMBEDDING_MODEL,
    VECTOR_DB,
)
from .logger import setup_logger

logger = setup_logger("retriever")

# Lazy-load globals
_chroma_client = None
_qdrant_client = None
_model = None


# ──────────────────────────────────────────────
# ChromaDB helpers
# ──────────────────────────────────────────────

def _get_chroma_collection():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client.get_collection(COLLECTION_NAME)


# ──────────────────────────────────────────────
# Qdrant helpers
# ──────────────────────────────────────────────

def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = qdrant_client.QdrantClient(
            url=QDRANT_URL,
            check_compatibility=False,
        )
    return _qdrant_client


def _qdrant_search(query_embedding: list, top_k: int) -> list:
    client = _get_qdrant_client()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_vectors=False,
        with_payload=True,
    )
    chunks = []
    for scored in results:
        payload = scored.payload or {}
        chunks.append({
            "document": payload.get("document", ""),
            "distance": float(scored.score),  # Qdrant uses score (higher = better)
            "product_id": payload.get("product_id", ""),
            "product_name": payload.get("product_name", ""),
            "category": payload.get("category", ""),
            "price": float(payload.get("price", 0)),
            "original_price": float(payload.get("original_price", 0)),
            "url": payload.get("url", ""),
            "thumbnail_url": payload.get("thumbnail_url", ""),
            "rating": float(payload.get("rating", 0)),
            "review_count": int(payload.get("review_count", 0)),
        })
    return chunks


def _qdrant_count() -> int:
    client = _get_qdrant_client()
    info = client.get_collection(collection_name=COLLECTION_NAME)
    return info.points_count


# ──────────────────────────────────────────────
# Embedding model
# ──────────────────────────────────────────────

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model ready")
    return _model


# ──────────────────────────────────────────────
# Keyword extraction
# ──────────────────────────────────────────────

VIETNESE_STOPWORDS = {
    "voi", "va", "cua", "cho", "hay", "co", "khong", "nao",
    "duoi", "tren", "tot", "nhat", "re", "mot", "cac",
    "nen", "mua", "tim", "gi", "o", "dau", "la", "thi",
    "nhu", "hon", "tu", "ve", "ra", "vao", "bang", "de",
    "toi", "ban", "minh", "ho", "no", "nay", "kia", "do",
    "cai", "viec", "dieu", "thu", "may", "bao", "nhieu",
    "trieu", "nghin", "k", "dong", "vnd",
}

PRICE_WORDS = {
    "nghin": 1e3, "k": 1e3, "trieu": 1e6, "tr": 1e6,
    "dong": 1, "vnd": 1,
}

BRAND_TERMS = {
    "sony", "samsung", "apple", "xiaomi", "oppo", "vivo", "huawei", "realme",
    "nike", "adidas",
}

ELECTRONICS_TERMS = {
    "tai", "nghe", "dien", "thoai", "laptop", "may", "tinh", "smartphone",
}

NOISE_TERMS = {
    "gau", "bong", "do", "choi", "sach", "wreath", "noel", "qua", "be",
}


def extract_price_range(query: str) -> Optional[Tuple[float, float]]:
    text = query.lower()
    text_clean = re.sub(r"[^\w\s]", " ", text)

    PRICE_UNIT_WORDS = {"k", "nghin", "trieu", "tr", "dong", "vnd"}
    PRICE_KEYWORDS = {"duoi", "tren", "tam", "khoang", "gia", "bao", "may", "price"}

    number_pattern = r"(\d+(?:[.,]\d+)?)\s*(k|nghin|trieu|tr|dong|vnd)?"
    raw_matches = list(re.finditer(number_pattern, text_clean))

    values = []
    has_price_phrase = "gia bao nhieu" in text_clean
    for m in raw_matches:
        num_str, unit = m.group(1), m.group(2)
        if unit and unit in PRICE_UNIT_WORDS:
            pass
        elif has_price_phrase:
            continue
        else:
            start, end = m.start(), m.end()
            before = text_clean[max(0, start - 5):start]
            after = text_clean[end:end + 5]
            nearby = (before + " " + after).split()
            if not any(pk in nearby for pk in PRICE_KEYWORDS):
                continue
        try:
            val = float(num_str.replace(",", "."))
        except ValueError:
            continue
        mult = PRICE_WORDS.get(unit, 1) if unit else 1
        values.append(val * mult)

    if not values:
        return None

    price = values[0]
    if "duoi" in text_clean:
        return (0, price)
    elif "tren" in text_clean:
        return (price, price * 10)
    elif "tam" in text_clean or "khoang" in text_clean:
        delta = price * 0.3
        return (max(0, price - delta), price + delta)
    else:
        return (0, price)


def _word_boundary_match(keyword: str, text: str) -> bool:
    escaped = re.escape(keyword)
    pattern = r"(?<!\w)" + escaped + r"(?!\w)"
    return bool(re.search(pattern, text, re.IGNORECASE))


def extract_keywords(query: str) -> Tuple[set, Optional[Tuple[float, float]]]:
    text = query.lower()
    text_clean = re.sub(r"[^\w\s]", " ", text)
    words = text_clean.split()
    keywords = {
        w for w in words
        if w not in VIETNESE_STOPWORDS and len(w) >= 2
    }
    price_range = extract_price_range(query)
    return keywords, price_range


# ──────────────────────────────────────────────
# Hybrid scoring
# ──────────────────────────────────────────────

def compute_keyword_score(c: dict, keywords: set, price_range: Optional[Tuple[float, float]]) -> float:
    score = 0.0
    name_lower = c["product_name"].lower()
    doc_lower = c["document"].lower()

    for kw in keywords:
        if _word_boundary_match(kw, name_lower):
            score += 20

    if keywords:
        matched = sum(
            1 for kw in keywords
            if _word_boundary_match(kw, doc_lower)
        )
        score += (matched / len(keywords)) * 5

    if price_range is not None:
        pmin, pmax = price_range
        p = c.get("price", 0)
        if pmin <= p <= pmax:
            score += 30
        elif pmax > 0 and p <= pmax * 2:
            score += 10

    brand_hits = sum(1 for b in BRAND_TERMS if _word_boundary_match(b, name_lower))
    if brand_hits:
        score += brand_hits * 18

    electronics_hits = sum(1 for t in ELECTRONICS_TERMS if _word_boundary_match(t, name_lower))
    if electronics_hits:
        score += min(electronics_hits, 3) * 6

    noise_hits = sum(1 for t in NOISE_TERMS if _word_boundary_match(t, name_lower))
    if noise_hits:
        score -= noise_hits * 12

    return score


# ──────────────────────────────────────────────
# Cohere re-ranking
# ──────────────────────────────────────────────

def _cohere_rerank(query: str, chunks: list, top_n: int) -> list:
    if not COHERE_API_KEY:
        logger.debug("COHERE_API_KEY not set -- skipping re-ranking")
        return chunks

    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": COHERE_RERANK_MODEL,
        "query": query,
        "documents": [c["document"] for c in chunks],
        "top_n": top_n,
    }

    try:
        resp = requests.post(
            "https://api.cohere.com/v1/rerank",
            headers=headers,
            json=body,
            timeout=15,
        )
        if resp.status_code in (429, 401):
            logger.warning(f"Cohere error {resp.status_code} -- skipping re-ranking")
            return chunks
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Cohere request failed ({e}) -- skipping re-ranking")
        return chunks

    try:
        data = resp.json()
    except Exception:
        return chunks

    score_map: dict = {}
    for item in data.get("results", []):
        score_map[item["index"]] = float(item["relevance_score"])

    reranked = []
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = score_map.get(i, 0.0)
        reranked.append(chunk)

    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return reranked


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def retrieve(query: str, top_k: int = 5) -> list:
    """
    Hybrid retrieval: vector similarity + keyword/price scoring, weighted fusion.

    Returns list of dicts:
        {
            "document": str,
            "distance": float,
            "product_id": str,
            "product_name": str,
            "category": str,
            "price": float,
            "url": str,
            "thumbnail_url": str,
            "rating": float,
            "review_count": int,
        }
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    model = _get_model()
    search_n = top_k * SEARCH_MULTIPLIER
    query_embedding = model.encode(query, show_progress_bar=False).tolist()

    # Vector search based on configured DB
    if VECTOR_DB == "qdrant":
        chunks = _qdrant_search(query_embedding, top_k=search_n)
    else:
        # ChromaDB
        collection = _get_chroma_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=search_n,
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        chunks = []
        for doc, meta, dist in zip(docs, metas, dists):
            if doc is None:
                continue
            chunks.append({
                "document": doc,
                "distance": float(dist),
                "product_id": meta.get("product_id", ""),
                "product_name": meta.get("product_name", ""),
                "category": meta.get("category", ""),
                "price": float(meta.get("price", 0)),
                "original_price": float(meta.get("original_price", 0)),
                "url": meta.get("url", ""),
                "thumbnail_url": meta.get("thumbnail_url", ""),
                "rating": float(meta.get("rating", 0)),
                "review_count": int(meta.get("review_count", 0)),
            })

    keywords, price_range = extract_keywords(query)

    if keywords or price_range is not None:
        kw_scores = {c["product_id"]: compute_keyword_score(c, keywords, price_range)
                     for c in chunks}
        max_kw = max(kw_scores.values()) if kw_scores else 1.0
        if max_kw == 0:
            max_kw = 1.0
    else:
        kw_scores = {}

    max_dist = max((c["distance"] for c in chunks), default=1.0)
    if max_dist == 0:
        max_dist = 1.0

    fusion_scores = []
    for c in chunks:
        pid = c["product_id"]
        if VECTOR_DB == "qdrant":
            # Qdrant score: higher is better.
            vec_score = c["distance"] / max_dist
        else:
            # Chroma distance: lower is better.
            vec_score = 1.0 - (c["distance"] / max_dist)

        kw = kw_scores.get(pid, 0.0) / max_kw if max_kw > 0 else 0.0
        combined = (1.0 - KEYWORD_WEIGHT) * vec_score + KEYWORD_WEIGHT * kw
        fusion_scores.append((combined, c))

    fusion_scores.sort(key=lambda x: x[0], reverse=True)

    seen_products = set()
    final = []
    for score, chunk in fusion_scores:
        pid = chunk["product_id"]
        if pid and pid not in seen_products:
            seen_products.add(pid)
            final.append(chunk)
            if len(final) >= top_k:
                break

    if len(final) < top_k:
        for c in chunks:
            pid = c["product_id"]
            if pid not in seen_products:
                seen_products.add(pid)
                final.append(c)
                if len(final) >= top_k:
                    break

    if len(final) >= 2:
        final = _cohere_rerank(query, final, top_n=RERANK_TOP_N)

    logger.info(
        f"[{VECTOR_DB}] Retrieved {len(final)} chunks (k={top_k}) | query: {query[:50]} | "
        f"keywords={keywords} | price_range={price_range}"
    )
    return final


def count_chunks() -> int:
    """Return total number of chunks in the active vector store."""
    if VECTOR_DB == "qdrant":
        return _qdrant_count()
    else:
        collection = _get_chroma_collection()
        return collection.count()


def retrieve_faqs(query: str, top_k: int = 3) -> list:
    """Retrieve FAQ chunks by semantic similarity (ChromaDB only for now)."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # FAQ still uses ChromaDB
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    try:
        faq_col = _chroma_client.get_collection(FAQ_COLLECTION_NAME)
    except Exception:
        return []

    if faq_col.count() == 0:
        return []

    model = _get_model()
    query_embedding = model.encode(query, show_progress_bar=False).tolist()

    results = faq_col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    faqs = []
    for doc, meta, dist in zip(docs, metas, dists):
        if doc is None:
            continue
        faqs.append({
            "document": doc,
            "distance": float(dist),
            "question": meta.get("question", ""),
            "answer": meta.get("answer", ""),
            "source": meta.get("source", "policy"),
        })

    logger.info(f"Retrieved {len(faqs)} FAQs for query: {query[:50]}")
    return faqs
