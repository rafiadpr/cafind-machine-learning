"""
Cafind — Offline ML Runner: ABSA Processing & SBERT Semantic Indexing
======================================================================
CLI script to process all reviews and index cafes in a single run:
  1. Executes ABSA Pipeline on all unprocessed reviews in PostgreSQL.
  2. Saves aspect sentiments (ambience, facility, price, food_beverage) to `aspect_sentiments`.
  3. Builds rich document representations for all cafes.
  4. Generates SBERT embeddings using `paraphrase-multilingual-MiniLM-L12-v2`.
  5. Saves persistent index matrix `cafe_embeddings.npy` and `cafe_id_map.json`.
  6. Executes verification test queries.

Usage:
    python machine_learning/scripts/run_absa_and_sbert.py
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Ensure backend and machine_learning are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
ML_DIR = PROJECT_ROOT / "machine_learning"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ML_DIR))

# Load backend environment variables
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cafind.runner")

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import AspectEnum, AspectSentiment, Cafe, Review, SentimentEnum
from machine_learning.src.absa_pipeline import ABSAPipeline, process_reviews_and_save_to_db
from machine_learning.src.sbert_embedder import SemanticVibeEngine


def get_async_engine():
    """Initializes SQLAlchemy async engine using DATABASE_URL."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:andre123@localhost:5434/cafind"
    )
    if not db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
    return create_async_engine(db_url, echo=False)


async def run_pipeline(batch_size: int = 100, test_query: str = "tempat nongkrong estetik kopi enak colokan banyak"):
    logger.info("================================================================")
    logger.info("Starting Cafind Offline ABSA and SBERT Indexing Pipeline")
    logger.info("================================================================")

    engine = get_async_engine()
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # ── STEP 1: Run ABSA on Unprocessed Reviews ──────────────────────────────
    logger.info("\n--- STEP 1: Aspect-Based Sentiment Analysis (ABSA) ---")
    async with session_factory() as session:
        absa_stats = await process_reviews_and_save_to_db(session, batch_size=batch_size)
        logger.info(f"ABSA Stats: {absa_stats}")

        # Total aspect sentiments in DB
        total_as = await session.scalar(select(func.count(AspectSentiment.id)))
        logger.info(f"Total AspectSentiment rows currently in DB: {total_as}")

    # ── STEP 2: Aggregate Cafe Corpora for SBERT ──────────────────────────────
    logger.info("\n--- STEP 2: Aggregating Cafe Documents & Reviews ---")
    async with session_factory() as session:
        # Load all cafes
        cafes_stmt = select(Cafe).order_by(Cafe.name.asc())
        cafes = (await session.execute(cafes_stmt)).scalars().all()
        logger.info(f"Found {len(cafes)} cafes in database.")

        cafe_documents: List[Dict[str, Any]] = []

        for cafe in cafes:
            # Load cafe reviews
            rev_stmt = (
                select(Review.text, Review.rating, Review.likes)
                .where(Review.cafe_id == cafe.id)
                .order_by(Review.rating.desc(), Review.likes.desc())
                .limit(25)
            )
            rev_rows = (await session.execute(rev_stmt)).all()
            reviews_list = [{"text": r.text, "rating": r.rating, "likes": r.likes} for r in rev_rows]

            # Load dominant ABSA summary per aspect
            as_stmt = (
                select(
                    AspectSentiment.aspect,
                    AspectSentiment.sentiment,
                    func.count().label("cnt")
                )
                .where(AspectSentiment.cafe_id == cafe.id)
                .group_by(AspectSentiment.aspect, AspectSentiment.sentiment)
                .order_by(AspectSentiment.aspect, func.count().desc())
            )
            as_rows = (await session.execute(as_stmt)).all()

            absa_profile: Dict[str, str] = {}
            seen_aspects = set()
            for row in as_rows:
                aspect_name = row.aspect.value if hasattr(row.aspect, "value") else str(row.aspect)
                sentiment_name = row.sentiment.value if hasattr(row.sentiment, "value") else str(row.sentiment)
                if aspect_name not in seen_aspects:
                    seen_aspects.add(aspect_name)
                    absa_profile[aspect_name] = sentiment_name

            # Build aggregated document text
            doc_text = SemanticVibeEngine.construct_cafe_document(
                cafe={
                    "name": cafe.name,
                    "address": cafe.address,
                    "categories": cafe.categories,
                },
                reviews=reviews_list,
                absa_summary=absa_profile
            )

            cafe_documents.append({
                "cafe_id": str(cafe.id),
                "name": cafe.name,
                "address": cafe.address,
                "rating": float(cafe.rating),
                "review_count": int(cafe.review_count),
                "categories": list(cafe.categories),
                "absa_profile": absa_profile,
                "sample_reviews": [r["text"] for r in reviews_list[:5]],
                "document_text": doc_text
            })

    # ── STEP 3: Generate SBERT Embeddings & Save Matrix Index ────────────────
    logger.info("\n--- STEP 3: Encoding with SBERT & Building Vector Index ---")
    vibe_engine = SemanticVibeEngine()
    vibe_engine.build_and_save_index(cafe_documents)

    # ── STEP 4: Verification Search Queries ──────────────────────────────────
    logger.info("\n--- STEP 4: Verification Search Queries ---")
    logger.info(f"Query 1: '{test_query}' (Unfiltered)")
    results1 = vibe_engine.search_vibe(test_query, limit=5)

    print("\n" + "=" * 80)
    print(f"SEARCH RESULTS FOR: \"{test_query}\"")
    print("=" * 80)
    for i, r in enumerate(results1, 1):
        print(f"{i}. {r['name']} (Similarity: {r['similarity_score']:.4f}, Rating: {r['rating']} stars)")
        print(f"   Address: {r['address']}")
        print(f"   ABSA Profile: {r['absa_profile']}")
        if r['highlights']:
            print(f"   Highlight: \"{r['highlights'][0]}\"")
        print("-" * 80)

    # Filtered query
    test_query_2 = "kafe murah meriah nugas sampai malam"
    filters_2 = {"facility": "positive"}
    logger.info(f"\nQuery 2: '{test_query_2}' with aspect_filter={filters_2}")
    results2 = vibe_engine.search_vibe(test_query_2, aspect_filters=filters_2, limit=5)

    print("\n" + "=" * 80)
    print(f"FILTERED SEARCH RESULTS FOR: \"{test_query_2}\" [facility=positive]")
    print("=" * 80)
    for i, r in enumerate(results2, 1):
        print(f"{i}. {r['name']} (Final Score: {r['final_score']:.4f}, Sim: {r['similarity_score']:.4f})")
        print(f"   ABSA Profile: {r['absa_profile']}")
        if r['highlights']:
            print(f"   Highlight: \"{r['highlights'][0]}\"")
        print("-" * 80)

    await engine.dispose()
    logger.info("\nPipeline execution finished successfully!")


def main():
    parser = argparse.ArgumentParser(description="Cafind ABSA & SBERT Offline Pipeline")
    parser.add_argument("--batch-size", type=int, default=100, help="DB commit batch size")
    parser.add_argument("--query", type=str, default="tempat nongkrong estetik kopi enak colokan banyak", help="Test query")
    args = parser.parse_args()

    asyncio.run(run_pipeline(batch_size=args.batch_size, test_query=args.query))


if __name__ == "__main__":
    main()
