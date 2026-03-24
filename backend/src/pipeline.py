"""
RAG Pipeline — retriever + generator for Vietnamese e-commerce chat.

Usage:
    from src.pipeline import answer
    result = answer("tai nghe Sony duoi 3 trieu")
"""

from .retriever import retrieve, retrieve_faqs, count_chunks
from .generator import generate
from .logger import setup_logger

logger = setup_logger("pipeline")
DEFAULT_TOP_K = 5

# Off-topic: greeting-only queries — skip retrieval for these
OFF_TOPIC = {
    "xin chao", "hello", "helo", "halo", "hi there", "hi you",
    "chao ban", "chao anh", "chao chi", "how are you", "what's up",
    "howdy", "greetings", "good morning", "good afternoon",
    "good evening",
}

# Product intent: multi-word or distinctive patterns (avoid "ban", "mua")
PRODUCT_INTENT = {
    "tai nghe", "dien thoai", "may tinh", "laptop", "smartphone",
    "apple", "samsung", "sony", "xiaomi", "oppo", "vivo", "huawei",
    "realme", "nike", "adidas", "giay", "ao quan", "may anh",
    "duoi", "tren", "tam", "khoang", "tot nhat", "tot",
    "tim", "can", "ban", "gia", "nao", "loai", "hang",
    "dien thoai", "may lanh", "tu lanh", "quat", "bep", "am nhac",
}


def answer(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    if not question or not question.strip():
        return {
            "answer": "Cau hoi trong. Ban vui long hoi lai.",
            "sources": [], "chunks": [],
            "product_context_used": "", "faq_context_used": None, "faqs": [],
        }

    total_chunks = count_chunks()
    q = question.lower().strip()
    logger.info(f"[Pipeline] Q={question[:50]} | top_k={top_k} | chunks={total_chunks}")

    # ── Intent detection ──────────────────────────────────────────────
    is_off_topic = any(g in q for g in OFF_TOPIC)
    has_product_intent = any(kw in q for kw in PRODUCT_INTENT)

    # ── Off-topic: skip retrieval entirely ─────────────────────────────
    if is_off_topic and not has_product_intent:
        logger.info(f"Off-topic suppressed: {question[:40]}")
        answer_text = generate(question, context="", faq_context=None)
        return {
            "answer": answer_text,
            "sources": [], "chunks": [],
            "product_context_used": "", "faq_context_used": None, "faqs": [],
        }

    # ── Retrieve ────────────────────────────────────────────────────
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        logger.warning("No chunks retrieved — vector DB empty or query failed")
        answer_text = generate(question, context="", faq_context=None)
        return {
            "answer": answer_text,
            "sources": [], "chunks": [],
            "product_context_used": "", "faq_context_used": None, "faqs": [],
        }

    # ── Relevance gate ────────────────────────────────────────────────
    first_score = chunks[0]["distance"]
    is_qdrant = first_score > 1.0
    retrieval_relevant = first_score >= 0.65 if is_qdrant else first_score < 1.5

    if retrieval_relevant:
        seen = set()
        unique_sources = []
        for c in chunks:
            key = c["product_name"]
            if key and key not in seen:
                seen.add(key)
                unique_sources.append({
                    "product_name": c["product_name"],
                    "category": c["category"],
                    "price": c["price"],
                    "original_price": c.get("original_price", 0),
                    "url": c["url"],
                    "thumbnail_url": c.get("thumbnail_url", ""),
                    "rating": c.get("rating", 0),
                    "review_count": c.get("review_count", 0),
                })
        product_context = "\n\n".join(c["document"] for c in chunks)
    else:
        unique_sources = []
        product_context = ""
        logger.info(f"Low relevance (score={first_score:.4f}), suppressing sources")

    # ── FAQs + generate ─────────────────────────────────────────────────
    faqs = retrieve_faqs(question, top_k=3)
    faq_context = "\n\n".join(f"{f['question']}\n{f['answer']}" for f in faqs) if faqs else None
    answer_text = generate(question, product_context, faq_context=faq_context)

    logger.info(
        f"[Pipeline] done | answer={len(answer_text)} chars | "
        f"sources={len(unique_sources)} | score={first_score:.4f}"
    )
    return {
        "answer": answer_text,
        "sources": unique_sources,
        "chunks": chunks if retrieval_relevant else [],
        "product_context_used": product_context,
        "faq_context_used": faq_context,
        "faqs": faqs,
    }
