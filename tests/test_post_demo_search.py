from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.sbert_embedder import SemanticVibeEngine, SBERTModelManager

class FixedEncoder:
    # Deterministic vectors test filtering/index behavior, not model quality.
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] for text in texts])

class FilterTests(unittest.TestCase):
    def test_reload_and_single_filter(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(SBERTModelManager, 'get_model', return_value=FixedEncoder()):
            embeddings = Path(directory) / 'vectors.npy'
            mapping = Path(directory) / 'ids.json'
            engine = SemanticVibeEngine(embeddings, mapping)
            engine.build_and_save_index([
                dict(cafe_id='a', document_text='Cafe A', absa_profile={'facility': 'positive'}),
                dict(cafe_id='b', document_text='Cafe B', absa_profile={'facility': 'negative'}),
            ])
            loaded = SemanticVibeEngine(embeddings, mapping)
            results = loaded.search_vibe('wifi', aspect_filters={'facility': 'positive'})
            self.assertEqual([r['cafe_id'] for r in results], ['a'])
            self.assertEqual(loaded.search_vibe(''), [])
            with self.assertRaises(NotImplementedError):
                loaded.search_vibe('wifi', aspect_filters={'facility': 'positive', 'price': 'positive'})
            with self.assertRaises(ValueError):
                loaded.search_vibe('wifi', limit=0)
