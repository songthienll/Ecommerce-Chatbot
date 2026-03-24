"""
Generator — call LLM (Groq or Ollama) with context + question.

Usage:
    from src.generator import generate
    answer = generate(
        question="Tai nghe nào tốt dưới 500k?",
        context="[chunk1 text]\n\n[chunk2 text]..."
    )
"""

import html
import re
import time
import unicodedata
from typing import Optional

import requests
from dotenv import load_dotenv

from .config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from .logger import setup_logger

logger = setup_logger("generator")

load_dotenv()


def clean_context(text: str) -> str:
    """Clean garbled Vietnamese text from Tiki API."""
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    for pattern, repl in [
        ("ning)", "nước sạch, "),
        ("ng vi", "người vi"),
    ]:
        text = text.replace(pattern, repl)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


PROMPT_TEMPLATE = """Bạn là trợ lý mua sắm Tiki. Hãy trả lời dựa trên context bên dưới.

QUY TẮC BẮT BUỘC:
1. Nếu context CÓ sản phẩm liên quan (bất kể chất lượng match): LUÔN đưa ra ÍT NHẤT 1 sản phẩm với TÊN + GIÁ + MÔ TẢ.
2. TUYỆT ĐỐI KHÔNG được nói "không tìm thấy", "không có sản phẩm", "không có trong dữ liệu" khi context có sản phẩm — đây là HALLUCINATION và bị cấm.
3. Nếu context KHÔNG có bất kỳ sản phẩm nào: nói rõ "Tôi không tìm thấy sản phẩm phù hợp trong dữ liệu hiện tại."
4. Không bịa đặt tên, giá, hay thông tin không có trong context.
5. Tối đa 3 sản phẩm gợi ý.

--- Context sản phẩm ---
{product_context}

--- FAQ Tiki (tham khảo) ---
{faq_context}

--- Câu hỏi người dùng ---
{question}

--- Trả lời --- """


def _call_ollama(prompt: str) -> str:
    """Call Ollama local LLM."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 384,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _call_groq(prompt: str) -> str:
    """Call Groq cloud API with retry on rate limit."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Set it in .env or use Ollama (LLM_PROVIDER=ollama).")

    for attempt in range(4):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 384,
                },
                timeout=60,
            )
            if response.status_code == 429:
                wait = (attempt + 1) * 5
                logger.warning(f"Groq rate limit, retrying in {wait}s (attempt {attempt+1}/4)")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429 and attempt < 3:
                wait = (attempt + 1) * 5
                logger.warning(f"Groq rate limit, retrying in {wait}s (attempt {attempt+1}/4)")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Groq rate limit: all 4 retries exhausted")


def generate(
    question: str,
    context: str,
    provider: Optional[str] = None,
    faq_context: Optional[str] = None,
) -> str:
    """
    Generate answer using LLM with retrieved context.

    Args:
        question: User's question (Vietnamese)
        context: Retrieved product chunks joined by newlines
        provider: "ollama" or "groq". Defaults to LLM_PROVIDER from env.
        faq_context: Retrieved FAQ chunks joined by newlines (optional)

    Returns:
        LLM-generated answer in Vietnamese.
    """
    if faq_context is None:
        faq_context = "(Không có thông tin chính sách liên quan)"
    elif not faq_context.strip():
        faq_context = "(Không có thông tin chính sách liên quan)"

    clean_ctx = clean_context(context) if context else ""
    no_faq_marker = "(Không có thông tin chính sách liên quan)"
    clean_faq = (
        clean_context(faq_context)
        if faq_context != no_faq_marker
        else faq_context
    )

    if not clean_ctx or not clean_ctx.strip():
        if clean_faq != no_faq_marker:
            prompt = PROMPT_TEMPLATE.format(
                product_context="(Không có thông tin sản phẩm liên quan)",
                faq_context=clean_faq,
                question=question,
            )
        else:
            return "Xin lỗi, mình không tìm thấy thông tin phù hợp để trả lời câu hỏi này."
    else:
        prompt = PROMPT_TEMPLATE.format(
            product_context=clean_ctx,
            faq_context=clean_faq,
            question=question,
        )

    provider = provider or LLM_PROVIDER
    model_name = OLLAMA_MODEL if provider == "ollama" else GROQ_MODEL
    logger.info(f"Generating answer via {provider} (model={model_name})")

    try:
        if provider == "ollama":
            return _call_ollama(prompt)
        elif provider == "groq":
            return _call_groq(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    except Exception as exc:
        logger.error(f"LLM call failed ({provider}): {exc}")
        return f"Xin lỗi, đã có lỗi khi tạo câu trả lời: {exc}"
