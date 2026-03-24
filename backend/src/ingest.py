"""
RAG ingestion — chunk products, embed with sentence-transformers, load into ChromaDB.

Usage:
    python -m src.ingest           # full ingest
    python -m src.ingest --reset   # wipe ChromaDB first
"""

import json
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    FAQ_COLLECTION_NAME,
)
from .logger import setup_logger

logger = setup_logger("ingest")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
FAQ_CLEANED = Path("data/cleaned/faqs.jsonl")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks of roughly `size` chars."""
    if not text or len(text) < 50:
        return [] if not text else [text]

    size = int(size)
    overlap = int(overlap)
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        if end >= len(text):
            chunks.append(text[start:])
            break
        cut = text.rfind(".", start, end)
        if cut > start + 50:
            chunks.append(text[start:cut + 1])
            start = max(start, cut + 1 - overlap)
        else:
            cut = text.rfind(" ", end - 20, end)
            if cut > start + 50:
                chunks.append(text[start:cut])
                start = max(start, cut - overlap)
            else:
                chunks.append(text[start:end])
                start = max(start, end - overlap)

    return chunks


def get_chroma_client() -> chromadb.PersistentClient:
    """Get ChromaDB persistent client (local storage)."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient):
    """Get or create the tiki_products collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Tiki product chunks for RAG chatbot"},
    )


def get_or_create_faq_collection(client: chromadb.PersistentClient):
    """Get or create the tiki_faqs collection."""
    return client.get_or_create_collection(
        name=FAQ_COLLECTION_NAME,
        metadata={"description": "Tiki policy FAQ Q&A for RAG chatbot"},
    )


def build_chunk_id(product_id: str, chunk_idx: int) -> str:
    raw = f"{product_id}_{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def build_faq_chunk_id(source: str, q_idx: int) -> str:
    raw = f"faq_{source}_{q_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_products(jsonl_path: Path) -> list:
    products = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))
    return products


def load_faqs(jsonl_path: Path) -> list:
    faqs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                faqs.append(json.loads(line))
    return faqs


def ingest_products(
    jsonl_path: Optional[Path] = None,
    batch_size: int = 100,
    reset: bool = False,
):
    """
    Chunk all products, embed, and load into ChromaDB.
    """
    if jsonl_path is None:
        jsonl_path = Path("data/cleaned/products.jsonl")

    if not jsonl_path.exists():
        logger.error(f"Cleaned products not found: {jsonl_path}")
        logger.info("Run: python -m src.ingest (with data in data/cleaned/products.jsonl)")
        return

    logger.info(f"Loading products from {jsonl_path}")
    products = load_products(jsonl_path)
    logger.info(f"Loaded {len(products)} products")

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model ready")

    client = get_chroma_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = get_or_create_collection(client)

    all_chunks = []
    all_ids = []
    all_embeddings = []
    all_metadata = []

    for product in products:
        text = product.get("combined_text", "")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        product_name = product.get("name", "")
        for idx, chunk in enumerate(chunks):
            chunk_id = build_chunk_id(str(product["product_id"]), idx)
            labeled_chunk = f"{product_name}. {chunk}" if idx == 0 else chunk
            all_chunks.append(labeled_chunk)
            all_ids.append(chunk_id)
            all_metadata.append({
                "product_id": str(product["product_id"]),
                "product_name": product["name"],
                "category": product.get("category_name", ""),
                "price": float(product.get("price", 0)),
                "original_price": float(product.get("original_price", 0)),
                "url": product.get("url", ""),
                "thumbnail_url": product.get("thumbnail_url", ""),
                "rating": float(product.get("rating_average", 0)),
                "review_count": int(product.get("review_count", 0)),
                "chunk_index": idx,
                "total_chunks": len(chunks),
            })

    logger.info(f"Created {len(all_chunks)} chunks from {len(products)} products")

    logger.info(f"Embedding chunks (batch_size={batch_size})...")
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        batch_meta = all_metadata[i:i + batch_size]

        embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=[e.tolist() if hasattr(e, "tolist") else e for e in embeddings],
            metadatas=batch_meta,
        )
        logger.info(f"  Indexed {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks")

    logger.info(f"Ingestion complete. Collection '{COLLECTION_NAME}' has {collection.count()} chunks")
    return collection.count()


def ingest_faqs(
    jsonl_path: Optional[Path] = None,
    batch_size: int = 100,
    reset: bool = False,
):
    """Chunk all FAQs, embed, and load into ChromaDB FAQ collection."""
    if jsonl_path is None:
        jsonl_path = FAQ_CLEANED

    if not jsonl_path.exists():
        logger.warning(f"Cleaned FAQs not found: {jsonl_path}")
        return 0

    logger.info(f"Loading FAQs from {jsonl_path}")
    faqs = load_faqs(jsonl_path)
    logger.info(f"Loaded {len(faqs)} FAQs")

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model ready")

    client = get_chroma_client()
    if reset:
        try:
            client.delete_collection(FAQ_COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {FAQ_COLLECTION_NAME}")
        except Exception:
            pass

    collection = get_or_create_faq_collection(client)

    texts = []
    ids = []
    metadatas = []

    for idx, faq in enumerate(faqs):
        question = faq.get("question", "")
        answer_text = faq.get("answer", "")
        source = faq.get("source", "policy")
        chunk_text_str = f"Q: {question}\nA: {answer_text}"
        chunk_id = build_faq_chunk_id(source, idx)
        texts.append(chunk_text_str)
        ids.append(chunk_id)
        metadatas.append({
            "question": question,
            "answer": answer_text,
            "source": source,
        })

    logger.info(f"Embedding {len(texts)} FAQ chunks...")
    embeddings = model.encode(texts, show_progress_bar=False)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    logger.info(f"FAQ ingestion complete. Collection '{FAQ_COLLECTION_NAME}' has {collection.count()} chunks")
    return collection.count()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Wipe collections before ingesting")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding")
    parser.add_argument("--faq", action="store_true", help="Also ingest FAQ collection")
    args = parser.parse_args()

    ingest_products(reset=args.reset, batch_size=args.batch_size)
    if args.faq:
        ingest_faqs(reset=args.reset)
