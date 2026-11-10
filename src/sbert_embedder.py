"""
Cafind — SBERT Semantic Vibe Search & Embedding Engine
=======================================================
Implements semantic vector search using multilingual Sentence-BERT:
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Features:
  - Memory-efficient Singleton model loader (prevents redundant model reloads).
  - Multi-source cafe text aggregation (metadata, reviews, ABSA vibe tags).
  - Persistent index generation: `cafe_embeddings.npy` & `cafe_id_map.json`.
  - Semantic inference engine with cosine similarity and ABSA aspect filters.
"""

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("cafind.sbert")

# ─── File Paths ───────────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
BASE_ML_DIR = PROJECT_ROOT / "machine_learning"
PROCESSED_DATA_DIR = BASE_ML_DIR / "data" / "processed"

EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "cafe_embeddings.npy"
ID_MAP_FILE = PROCESSED_DATA_DIR / "cafe_id_map.json"

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ─── Singleton SBERT Model Manager ───────────────────────────────────────────
class SBERTModelManager:
    """Thread-safe Singleton manager for loading the SentenceTransformer model."""

    _instance: Optional["SBERTModelManager"] = None
    _lock = threading.Lock()
    _model: Any = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SBERTModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_MODEL_NAME) -> Any:
        """Lazily instantiates and returns the shared SentenceTransformer model."""
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    logger.info(f"Loading SBERT model '{model_name}' into memory...")
                    from sentence_transformers import SentenceTransformer
                    cls._model = SentenceTransformer(model_name)
                    logger.info("SBERT model successfully initialized.")
        return cls._model


# ─── Semantic Vibe Engine ─────────────────────────────────────────────────────
class SemanticVibeEngine:
    """Manages index creation, embedding calculation, and semantic search queries."""

    def __init__(
        self,
        embeddings_path: Path = EMBEDDINGS_FILE,
        id_map_path: Path = ID_MAP_FILE,
        model_name: str = DEFAULT_MODEL_NAME
    ):
        self.embeddings_path = embeddings_path
        self.id_map_path = id_map_path
        self.model_name = model_name

        self.embeddings: Optional[np.ndarray] = None
        self.cafe_records: List[Dict[str, Any]] = []
        self._load_index_if_exists()

    def _load_index_if_exists(self) -> bool:
        """Loads precomputed embeddings and ID map if available on disk."""
        if self.embeddings_path.exists() and self.id_map_path.exists():
            try:
                self.embeddings = np.load(self.embeddings_path)
                with open(self.id_map_path, "r", encoding="utf-8") as f:
                    self.cafe_records = json.load(f)
                logger.info(
                    f"Loaded precomputed SBERT index: {len(self.cafe_records)} cafes, "
                    f"embedding shape {self.embeddings.shape}"
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}. Index will need rebuild.")
        return False

    @staticmethod
    def construct_cafe_document(
        cafe: Dict[str, Any],
        reviews: List[Dict[str, Any]],
        absa_summary: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Creates a dense textual representation combining cafe name, categories,
        address, positive ABSA vibes, and representative review snippets.
        """
        name = cafe.get("name", "")
        address = cafe.get("address", "")
        categories = cafe.get("categories", [])
        if isinstance(categories, list):
            cat_str = ", ".join(categories)
        else:
            cat_str = str(categories)

        # ABSA vibe descriptors
        vibe_phrases = []
        if absa_summary:
            if absa_summary.get("ambience") == "positive":
                vibe_phrases.append("suasana nyaman, estetik, tempat santai, adem")
            if absa_summary.get("facility") == "positive":
                vibe_phrases.append("fasilitas lengkap, wifi kencang, colokan banyak, cocok nugas wfc")
            if absa_summary.get("price") == "positive":
                vibe_phrases.append("harga murah, terjangkau, ramah kantong mahasiswa")
            if absa_summary.get("food_beverage") == "positive":
                vibe_phrases.append("kopi enak, menu makanan lezat, minuman berkualitas")

        # Select top informative reviews (prioritize higher rating and substantive length)
        selected_reviews = []
        for r in reviews:
            text = r.get("text", "")
            if len(text.strip()) > 15:
                selected_reviews.append(text.strip())
            if len(selected_reviews) >= 8:
                break

        doc_parts = [
            f"Kafe {name}.",
            f"Kategori: {cat_str}." if cat_str else "",
            f"Alamat di {address}." if address else "",
            f"Karakteristik suasana: {', '.join(vibe_phrases)}." if vibe_phrases else "",
            "Ulasan pelanggan: " + " ".join(selected_reviews) if selected_reviews else ""
        ]
        return " ".join([p for p in doc_parts if p]).strip()

    def build_and_save_index(
        self,
        cafe_documents: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Computes normalized SBERT embeddings for all cafe documents and persists
        them to disk as cafe_embeddings.npy and cafe_id_map.json.
        """
        if not cafe_documents:
            raise ValueError("No cafe documents provided for indexing.")

        texts = [doc["document_text"] for doc in cafe_documents]
        model = SBERTModelManager.get_model(self.model_name)

        logger.info(f"Generating embeddings for {len(texts)} cafes via SBERT...")
        # L2-normalized embeddings enable cosine similarity via matrix dot product
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=32,
            convert_to_numpy=True
        )

        self.embeddings = embeddings
        self.cafe_records = [
            {
                "index": i,
                "cafe_id": str(doc["cafe_id"]),
                "name": doc.get("name", ""),
                "address": doc.get("address", ""),
                "rating": doc.get("rating", 0.0),
                "review_count": doc.get("review_count", 0),
                "categories": doc.get("categories", []),
                "absa_profile": doc.get("absa_profile", {}),
                "sample_reviews": doc.get("sample_reviews", [])
            }
            for i, doc in enumerate(cafe_documents)
        ]

        # Save to disk
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        with open(self.id_map_path, "w", encoding="utf-8") as f:
            json.dump(self.cafe_records, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Successfully saved SBERT index to {self.embeddings_path} and {self.id_map_path}"
        )
        return self.embeddings, self.cafe_records

    def search_vibe(
        self,
        query: str,
        aspect_filters: Optional[Dict[str, str]] = None,
        limit: int = 10,
        min_similarity: float = 0.15,
        strict_aspects: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic vibe search:
          1. Encodes user query into 384-d normalized vector.
          2. Calculates cosine similarity via dot product against all cafes.
          3. Evaluates ABSA aspect filters / intent boosts.
          4. Returns top cafes sorted by combined score with relevant highlights.
        """
        if self.embeddings is None or not self.cafe_records:
            loaded = self._load_index_if_exists()
            if not loaded or self.embeddings is None:
                raise RuntimeError(
                    "SBERT index not loaded. Please run index generation first."
                )

        if not query or not query.strip():
            return []

        model = SBERTModelManager.get_model(self.model_name)
        query_embedding = model.encode(
            [query.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        # Cosine similarity via dot product (both query and corpus vectors are L2-normalized)
        similarities = np.dot(self.embeddings, query_embedding)

        query_keywords = re.findall(r"\b\w+\b", query.lower())
        results = []

        for idx, sim in enumerate(similarities):
            record = self.cafe_records[idx]
            absa_prof = record.get("absa_profile", {})

            # Aspect filtering / boosting
            filter_match = True
            boost = 0.0

            if aspect_filters:
                for req_aspect, req_sentiment in aspect_filters.items():
                    actual_sentiment = absa_prof.get(req_aspect)
                    if actual_sentiment is not None:
                        actual_lower = actual_sentiment.lower()
                        req_lower = req_sentiment.lower()
                        if actual_lower == req_lower:
                            boost += 0.12  # Reward exact positive match
                        elif actual_lower == "neutral":
                            boost += 0.02  # Neutral is acceptable (not eliminated)
                        else:
                            # Explicitly negative when positive requested (or vice-versa)
                            filter_match = False
                    else:
                        # Aspect not yet tagged: neutral behavior, keep in results
                        pass

            # Direct cafe name match boost (Hybrid Dense + Lexical search)
            record_name_lower = record.get("name", "").lower()
            query_clean = query.strip().lower()
            if len(query_clean) >= 3 and query_clean in record_name_lower:
                boost += 0.35
            elif any(len(kw) >= 4 and kw in record_name_lower for kw in query_keywords):
                boost += 0.15

            if not filter_match:
                continue

            final_score = float(sim) + boost
            if final_score < min_similarity:
                continue

            # Extract relevant positive highlights matching query intent
            highlights = self._extract_relevant_highlights(
                record.get("sample_reviews", []),
                query_keywords,
                absa_prof
            )

            results.append({
                "cafe_id": record["cafe_id"],
                "name": record["name"],
                "address": record["address"],
                "rating": record["rating"],
                "review_count": record["review_count"],
                "categories": record["categories"],
                "similarity_score": round(float(sim), 4),
                "final_score": round(final_score, 4),
                "absa_profile": absa_prof,
                "highlights": highlights
            })

        # Sort by final_score descending
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:limit]

    def _extract_relevant_highlights(
        self,
        reviews: List[str],
        query_keywords: List[str],
        absa_profile: Dict[str, str]
    ) -> List[str]:
        """Extracts 1-3 review sentences that best match query keywords and positive vibes."""
        scored_sentences = []
        for rev in reviews:
            # Split review into sentences
            sentences = re.split(r"[.!?\n]+", rev)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 15 or len(s_clean) > 200:
                    continue

                s_lower = s_clean.lower()
                # Keyword overlap
                kw_hits = sum(1 for kw in query_keywords if kw in s_lower and len(kw) > 2)
                # Positive vibe term hit
                pos_hits = sum(1 for p in ["enak", "nyaman", "kencang", "estetik", "murah", "adem", "lengkap"] if p in s_lower)

                score = kw_hits * 2.0 + pos_hits * 1.0
                if score > 0:
                    scored_sentences.append((score, s_clean))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        # Deduplicate and return top 2
        seen = set()
        highlights = []
        for _, s in scored_sentences:
            if s not in seen:
                seen.add(s)
                highlights.append(s)
            if len(highlights) >= 2:
                break

        return highlights
