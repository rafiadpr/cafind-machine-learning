import unittest
from src.absa_pipeline import ABSAPipeline

class AspectTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = ABSAPipeline()

    def test_empty_text(self):
        self.assertEqual(self.pipeline.extract_aspects(''), [])

    def test_wifi_positive_and_negated(self):
        for text, expected in [('wifi cepat', 'positive'), ('wifi tidak cepat', 'negative')]:
            results = {r.aspect: r.sentiment for r in self.pipeline.extract_aspects(text)}
            self.assertEqual(results['facility'], expected)

    def test_contrasting_aspects(self):
        results = {r.aspect: r.sentiment for r in self.pipeline.extract_aspects('wifi cepat tapi harga mahal')}
        self.assertEqual(results['facility'], 'positive')
        self.assertEqual(results['price'], 'negative')
