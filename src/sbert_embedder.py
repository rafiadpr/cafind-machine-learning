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
    def construct_cafe_document(cafe, reviews, absa_summary=None):
        parts = [str(cafe.get('name', '')), str(cafe.get('address', ''))]
        parts.extend((str(review.get('text', '')) for review in reviews[:3]))
        return ' '.join(parts).strip()

    def build_and_save_index(self, cafe_documents):
        if not cafe_documents:
            raise ValueError('No cafe documents provided for indexing')
        model = SBERTModelManager.get_model(self.model_name)
        self.embeddings = model.encode([doc['document_text'] for doc in cafe_documents], normalize_embeddings=True, convert_to_numpy=True)
        self.cafe_records = [dict(doc, cafe_id=str(doc['cafe_id'])) for doc in cafe_documents]
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        self.id_map_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        self.id_map_path.write_text(json.dumps(self.cafe_records, ensure_ascii=False), encoding='utf-8')
        return (self.embeddings, self.cafe_records)
