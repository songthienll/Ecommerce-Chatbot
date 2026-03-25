"""
RAG Pipeline — retriever + generator for Vietnamese e-commerce chat.

Usage:
    from src.pipeline import answer
    result = answer("tai nghe Sony duoi 3 trieu")
"""

import re

from .retriever import retrieve, retrieve_faqs, count_chunks
from .generator import generate
from .logger import setup_logger
from .config import VECTOR_DB

logger = setup_logger("pipeline")
DEFAULT_TOP_K = 5

# Off-topic: greeting-only queries — skip retrieval for these
OFF_TOPIC = {
    "xin chao", "hello", "helo", "halo", "hi there", "hi you",
    "chao ban", "chao anh", "chao chi", "how are you", "what's up",
    "howdy", "greetings", "good morning", "good afternoon",
    "good evening",
}

# Product intent: multi-word or distinctive patterns
PRODUCT_INTENT = {
    "tai nghe", "dien thoai", "may tinh", "laptop", "smartphone",
    "apple", "samsung", "sony", "xiaomi", "oppo", "vivo", "huawei",
    "realme", "nike", "adidas", "giay", "ao quan", "may anh",
    "duoi", "tren", "tam", "khoang", "tot nhat", "tot",
    "tim", "can", "gia", "nao", "loai", "hang",
    "may lanh", "tu lanh", "quat", "bep", "am nhac",
}

NEGATIVE_HINTS = {
    "sach", "loc nuoc", "nuoc", "ve sinh", "giat", "noel", "qua",
    "do choi", "duong the", "trang", "book", "wreath",
}

BRANDS = {"sony", "samsung", "apple", "xiaomi", "oppo", "vivo", "huawei", "realme", "nike", "adidas"}
CATEGORY_TERMS = {"dien thoai", "tai nghe", "laptop", "may tinh", "smartphone", "giay", "ao quan", "may anh"}
LAPTOP_TERMS = {"laptop", "notebook", "gaming", "rtx", "4050", "4060", "4070"}
LAPTOP_DEVICE_TERMS = {"notebook", "macbook", "ideapad", "thinkpad", "vivobook", "legion", "rog", "nitro", "rtx", "4050", "4060", "4070"}
PHONE_TERMS = {"dien", "thoai", "smartphone", "ram", "mah", "pin"}


def _strip_greetings(query: str) -> str:
    q = _normalize_text(query)
    for g in OFF_TOPIC:
        q = q.replace(g, " ")
    return _normalize_text(q)


def _is_laptop_query(query: str) -> bool:
    q = _normalize_text(query)
    return any(t in q for t in LAPTOP_TERMS)


def _is_phone_query(query: str) -> bool:
    q = _normalize_text(query)
    return any(t in q for t in PHONE_TERMS)


def _has_category_conflict(question: str, sources: list[dict]) -> bool:
    q_laptop = _is_laptop_query(question)
    q_phone = _is_phone_query(question)
    if not (q_laptop or q_phone):
        return False

    source_text = " ".join(_normalize_text(s.get("product_name", "")) for s in sources)
    laptop_in_sources = any(t in source_text for t in LAPTOP_TERMS)
    phone_in_sources = any(t in source_text for t in PHONE_TERMS)

    if q_laptop and not laptop_in_sources:
        return True
    if q_phone and not phone_in_sources:
        return True
    return False


def _has_hard_constraints(query: str) -> bool:
    q = _normalize_text(query)
    return any(t in q for t in {"rtx", "ram", "mah", "gb", "4050", "4060", "4070"})


def _extract_digits(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def _hard_constraint_digits_match(question: str, sources: list[dict]) -> bool:
    q_digits = _extract_digits(_normalize_text(question))
    if not q_digits:
        return True
    source_text = " ".join(_normalize_text(s.get("product_name", "")) for s in sources)
    s_digits = _extract_digits(source_text)
    return len(q_digits.intersection(s_digits)) > 0


def _is_domain_supported(question: str, chunks: list[dict]) -> bool:
    q = _normalize_text(question)
    if "laptop" not in q and "rtx" not in q and "gaming" not in q:
        return True
    names = " ".join(_normalize_text(c.get("product_name", "")) for c in chunks[:10])
    return any(t in names for t in LAPTOP_DEVICE_TERMS)


def _fallback_out_of_scope(question: str) -> str:
    q = _normalize_text(question)
    if "laptop" in q or "rtx" in q:
        return "Xin lỗi, hiện dữ liệu chưa có sản phẩm laptop gaming phù hợp (ví dụ RTX 4050). Bạn thử mở rộng mức giá hoặc đổi sang danh mục khác nhé."
    if "dien thoai" in q or "smartphone" in q:
        return "Xin lỗi, hiện dữ liệu chưa đủ sản phẩm điện thoại đúng cấu hình bạn yêu cầu (RAM/PIN/GIÁ). Bạn có thể nới điều kiện để mình tìm tốt hơn."
    return "Xin lỗi, mình không tìm thấy thông tin phù hợp để trả lời câu hỏi này."


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _has_product_intent(query: str) -> bool:
    return any(kw in query for kw in PRODUCT_INTENT)


def _query_has_brand_or_category_terms(query: str) -> bool:
    q = _normalize_text(query)
    has_brand = any(b in q for b in BRANDS)
    has_category = any(c in q for c in CATEGORY_TERMS)
    return has_brand or has_category


def _source_keyword_overlap_ratio(query: str, sources: list[dict]) -> float:
    query_words = {w for w in _normalize_text(query).split() if len(w) >= 3}
    query_words -= {"tot", "nhat", "cho", "sinh", "vien", "duoi", "tren", "tam", "khoang"}
    if not query_words or not sources:
        return 0.0

    source_text = " ".join(_normalize_text(s.get("product_name", "")) for s in sources)
    matched = sum(1 for w in query_words if w in source_text)
    return matched / max(len(query_words), 1)


def _has_negative_sources(sources: list[dict]) -> bool:
    source_text = " ".join(_normalize_text(s.get("product_name", "")) for s in sources)
    return any(h in source_text for h in NEGATIVE_HINTS)


def _is_source_set_relevant(
    question: str,
    sources: list[dict],
    first_score: float,
    is_qdrant: bool,
    has_product_intent: bool,
) -> bool:
    overlap = _source_keyword_overlap_ratio(question, sources)
    negative_sources = _has_negative_sources(sources)
    has_brand_or_category = _query_has_brand_or_category_terms(question)
    category_conflict = _has_category_conflict(question, sources)
    hard_constraints = _has_hard_constraints(question)
    digit_match = _hard_constraint_digits_match(question, sources)

    if has_product_intent:
        overlap_gate = overlap >= 0.15
        if category_conflict:
            return False
        if hard_constraints and not digit_match:
            return False

        # If query has explicit product signal but retrieved set looks dirty, suppress.
        if has_brand_or_category and negative_sources and overlap < 0.40:
            return False

        return overlap_gate

    score_gate = first_score >= 0.72 if is_qdrant else first_score < 1.3
    overlap_gate = overlap >= 0.35
    negative_gate = not negative_sources
    return score_gate and overlap_gate and negative_gate


def answer(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    if not question or not question.strip():
        return {
            "answer": "Cau hoi trong. Ban vui long hoi lai.",
            "sources": [], "chunks": [],
            "product_context_used": "", "faq_context_used": None, "faqs": [],
        }

    total_chunks = count_chunks()
    q = question.lower().strip()
    q_no_greeting = _strip_greetings(q)
    logger.info(f"[Pipeline] Q={question[:50]} | top_k={top_k} | chunks={total_chunks}")

    # ── Intent detection ──────────────────────────────────────────────
    is_off_topic = any(g in q for g in OFF_TOPIC)
    has_product_intent = _has_product_intent(q_no_greeting)

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
        answer_text = _fallback_out_of_scope(question)
        return {
            "answer": answer_text,
            "sources": [], "chunks": [],
            "product_context_used": "", "faq_context_used": None, "faqs": [],
        }

    # ── Domain support gate ───────────────────────────────────────────
    if not _is_domain_supported(question, chunks):
        logger.info("Domain not supported in current dataset, suppressing sources")
        return {
            "answer": _fallback_out_of_scope(question),
            "sources": [], "chunks": [],
            "product_context_used": "", "faq_context_used": None, "faqs": [],
        }

    # ── Relevance gate ────────────────────────────────────────────────
    first_score = chunks[0]["distance"]
    is_qdrant = VECTOR_DB == "qdrant"

    seen = set()
    candidate_sources = []
    for c in chunks:
        key = c["product_name"]
        if key and key not in seen:
            seen.add(key)
            candidate_sources.append({
                "product_name": c["product_name"],
                "category": c["category"],
                "price": c["price"],
                "original_price": c.get("original_price", 0),
                "url": c["url"],
                "thumbnail_url": c.get("thumbnail_url", ""),
                "rating": c.get("rating", 0),
                "review_count": c.get("review_count", 0),
            })

    retrieval_relevant = _is_source_set_relevant(
        question=question,
        sources=candidate_sources,
        first_score=first_score,
        is_qdrant=is_qdrant,
        has_product_intent=has_product_intent,
    )

    if retrieval_relevant:
        unique_sources = candidate_sources
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
