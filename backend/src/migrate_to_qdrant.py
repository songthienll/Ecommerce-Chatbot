"""
Migrate ChromaDB -> Qdrant for tiki_products collection.
Run: python -m src.migrate_to_qdrant
"""

import uuid

import chromadb
import qdrant_client
from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff

from .config import CHROMA_DIR, COLLECTION_NAME

CHROMA_COLLECTION = COLLECTION_NAME
QDRANT_COLLECTION = COLLECTION_NAME
VECTOR_SIZE = 768
BATCH_SIZE = 500


def migrate():
    print("=== ChromaDB -> Qdrant Migration ===")

    # 1. ChromaDB
    print("Loading ChromaDB...")
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    chroma_col = chroma_client.get_collection(CHROMA_COLLECTION)
    total = chroma_col.count()
    print(f"  Collection: {CHROMA_COLLECTION}")
    print(f"  Total chunks: {total:,}")

    # 2. Qdrant
    print("Connecting to Qdrant...")
    qdrant = qdrant_client.QdrantClient(url="http://localhost:6333", check_compatibility=False)

    collections = qdrant.get_collections().collections
    collection_names = [c.name for c in collections]
    if QDRANT_COLLECTION in collection_names:
        print(f"  Deleting existing collection: {QDRANT_COLLECTION}")
        qdrant.delete_collection(collection_name=QDRANT_COLLECTION)

    print(f"  Creating Qdrant collection: {QDRANT_COLLECTION}")
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=50000,
            memmap_threshold=50000,
        ),
    )
    print("  Collection created")

    # 3. Migrate
    offset = 0
    migrated = 0

    while offset < total:
        results = chroma_col.get(
            include=["documents", "metadatas", "embeddings"],
            limit=BATCH_SIZE,
            offset=offset,
        )

        if not results["documents"]:
            break

        ids = []
        vectors = []
        payloads = []

        all_embs = results.get("embeddings") or []
        docs = results["documents"]
        metas = results["metadatas"]

        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            embedding = all_embs[i] if all_embs and i < len(all_embs) else None

            if embedding is None:
                print(f"  WARNING: Missing embedding at offset={offset} i={i} -- SKIP")
                continue

            ids.append(str(uuid.uuid4()))
            vectors.append(embedding)
            payloads.append({
                "product_id": meta.get("product_id", ""),
                "product_name": meta.get("product_name", ""),
                "category": meta.get("category", ""),
                "price": float(meta.get("price", 0)),
                "original_price": float(meta.get("original_price", 0)),
                "url": meta.get("url", ""),
                "thumbnail_url": meta.get("thumbnail_url", ""),
                "rating": float(meta.get("rating", 0)),
                "review_count": int(meta.get("review_count", 0)),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1),
                "document": doc,
            })

        if not ids:
            print(f"  No valid at offset={offset} -- breaking")
            break

        qdrant.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                qdrant_client.models.PointStruct(
                    id=ids[j],
                    vector=vectors[j],
                    payload=payloads[j],
                )
                for j in range(len(ids))
            ],
        )

        migrated += len(ids)
        offset += BATCH_SIZE
        pct = (migrated / total) * 100
        print(f"  Migrated {migrated:,}/{total:,} ({pct:.1f}%)")

    print(f"\n[migration complete] {migrated:,} chunks -> Qdrant '{QDRANT_COLLECTION}'")
    print(f"   Qdrant dashboard: http://localhost:6333/dashboard")


if __name__ == "__main__":
    migrate()
