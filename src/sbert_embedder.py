"""Early Cafind ML subset. Advanced search, ranking and database sync are pending."""
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
logger = logging.getLogger('cafind.sbert')
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
BASE_ML_DIR = PROJECT_ROOT
PROCESSED_DATA_DIR = BASE_ML_DIR / 'data' / 'processed'
EMBEDDINGS_FILE = PROCESSED_DATA_DIR / 'cafe_embeddings.npy'
ID_MAP_FILE = PROCESSED_DATA_DIR / 'cafe_id_map.json'
DEFAULT_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

class SBERTModelManager:
    """Thread-safe Singleton manager for loading the SentenceTransformer model."""
    _instance: Optional['SBERTModelManager'] = None
    _lock = threading.Lock()
    _model: Any = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SBERTModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_model(cls, model_name: str=DEFAULT_MODEL_NAME) -> Any:
        """Lazily instantiates and returns the shared SentenceTransformer model."""
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    logger.info(f"Loading SBERT model '{model_name}' into memory...")
                    from sentence_transformers import SentenceTransformer
                    cls._model = SentenceTransformer(model_name)
                    logger.info('SBERT model successfully initialized.')
        return cls._model

class SemanticVibeEngine:

    def __init__(self, embeddings_path: Path=EMBEDDINGS_FILE, id_map_path: Path=ID_MAP_FILE, model_name: str=DEFAULT_MODEL_NAME):
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
                with open(self.id_map_path, 'r', encoding='utf-8') as f:
                    self.cafe_records = json.load(f)
                logger.info(f'Loaded precomputed SBERT index: {len(self.cafe_records)} cafes, embedding shape {self.embeddings.shape}')
                return True
            except Exception as e:
                logger.warning(f'Failed to load cached index: {e}. Index will need rebuild.')
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