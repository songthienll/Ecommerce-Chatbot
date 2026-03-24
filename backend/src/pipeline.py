"""
RAG Pipeline — combine retriever + generator for end-to-end question answering.

Usage:
    from src.pipeline import answer
    result = answer("tai nghe không dây dưới 500k")
"""

from .retriever import retrieve, retrieve_faqs, count_chunks
from .generator import generate
from .logger import setup_logger

logger = setup_logger("pipeline")

DEFAULT_TOP_K = 5


def answer(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Answer a user question using RAG.

    1. Retrieve top-k relevant product chunks from ChromaDB
    2. Generate answer using LLM with retrieved context
    3. Return answer + sources for display
    """
    if not question or not question.strip():
        return {
            "answer": "Xin lỗi, câu hỏi trống. Bạn vui lòng hỏi lại nhé.",
            "sources": [],
            "chunks": [],
            "product_context_used": "",
            "faq_context_used": None,
            "faqs": [],
        }

    total_chunks = count_chunks()
    logger.info(f"[Pipeline] Question: {question[:60]} | top_k={top_k} | total_chunks={total_chunks}")

    # Step 1: retrieve products + FAQs
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        logger.warning("No chunks retrieved — ChromaDB may be empty. Run: python -m src.ingest")
        return {
            "answer": "Xin lỗi, mình không tìm thấy sản phẩm phù hợp. Có thể cần ingest dữ liệu trước.",
            "sources": [],
            "chunks": [],
            "product_context_used": "",
            "faq_context_used": None,
            "faqs": [],
        }

    # Step 2: relevance gate
    # Qdrant uses cosine similarity (0-1, higher=better); ChromaDB uses distance (lower=better)
    if chunks:
        first_dist = chunks[0]["distance"]
        # Qdrant score: if >1 it's similarity; ChromaDB distance: typically 0-3
        is_qdrant = first_dist > 1.0
        if is_qdrant:
            # Cosine similarity: relevant if top score >= 0.4
            retrieval_relevant = first_dist >= 0.40
        else:
            # ChromaDB distance: relevant if top dist < 1.6
            retrieval_relevant = first_dist < 1.6

    if retrieval_relevant:
        sources = [
            {
                "product_name": c["product_name"],
                "category": c["category"],
                "price": c["price"],
                "original_price": c.get("original_price", 0),
                "url": c["url"],
                "thumbnail_url": c["thumbnail_url"],
                "rating": c["rating"],
                "review_count": c["review_count"],
                "rerank_score": c.get("rerank_score"),
            }
            for c in chunks
        ]
        seen = set()
        unique_sources = []
        for s in sources:
            key = s["product_name"]
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)
        product_context = "\n\n".join(c["document"] for c in chunks)
    else:
        unique_sources = []
        product_context = ""
        logger.info(f"Retrieval diffuse: first_dist={first_dist:.4f}")

    # Also retrieve relevant FAQs
    faqs = retrieve_faqs(question, top_k=3)
    faq_context = "\n\n".join(f"{f['question']}\n{f['answer']}" for f in faqs) if faqs else None

    # Step 3: generate
    answer_text = generate(question, product_context, faq_context=faq_context)

    result = {
        "answer": answer_text,
        "sources": unique_sources,
        "chunks": chunks if retrieval_relevant else [],
        "product_context_used": product_context,
        "faq_context_used": faq_context,
        "faqs": faqs,
    }

    logger.info(
        f"[Pipeline] Answer generated ({len(answer_text)} chars), "
        f"{len(unique_sources)} sources, {len(faqs)} FAQs"
    )
    return result
