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
