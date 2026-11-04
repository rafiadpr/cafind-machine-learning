import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from src.review_size_comparison import nested_samples,compare_review_sizes,validate_sizes


class Encoder:
    def encode(self,texts,**kwargs):
        return np.array([[1.0,0.0] for _ in texts],dtype=np.float32)


class Pipeline:
    def extract_aspects(self,text):
        return [SimpleNamespace(aspect='facility',sentiment='positive')] if 'wifi' in text else []


class ReviewComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'review_id':str(i),'cafe_id':str(i%3),'text':'wifi cepat dan nyaman untuk bekerja' if i%2 else 'ulasan tentang tempat ini','rating':5,'likes':0} for i in range(40)]
        self.cafes={str(i):{'name':'Cafe '+str(i)} for i in range(3)}

    def test_nested_sample_prefixes_are_reproducible(self):
        a=nested_samples(self.rows,[10,20,30],42)
        b=nested_samples(self.rows,[30,10,20],42)
        self.assertEqual(a,b)
        self.assertEqual(a[0][1],a[1][1][:10])
        self.assertEqual(a[1][1],a[2][1][:20])
        self.assertNotEqual(a,nested_samples(self.rows,[10,20,30],43))
        with self.assertRaises(ValueError):nested_samples(self.rows,[100],42)
        for sizes in [[],[0],[10001]]:
            with self.assertRaises(ValueError):validate_sizes(sizes)

    def test_report_uses_exact_counts_isolated_indexes_and_no_input_mutation(self):
        original=copy.deepcopy(self.rows)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            production=root/'production-index.npy';production.write_bytes(b'unchanged')
            report=compare_review_sizes(self.rows,self.cafes,[10,20,30],'wifi',root/'run',
                Encoder(),Pipeline(),lambda cafe,reviews,profile:cafe['name']+' '+' '.join(r['text'] for r in reviews),
                seed=42,repeats=2)
            self.assertEqual(len(report['trials']),6)
            self.assertEqual([r['reviews'] for r in report['summary']],[10,20,30])
            self.assertEqual(production.read_bytes(),b'unchanged')
            self.assertEqual(json.loads((root/'run/result.json').read_text())['sizes'],[10,20,30])
            self.assertTrue((root/'run/trial-01/n-10/cafe_embeddings.npy').is_file())
            for row in report['trials']:
                self.assertLessEqual(row['aspect_coverage'],1)
                self.assertLessEqual(row['quoted_reviews'],row['reviews'])
                if row['reviews']==30:self.assertEqual(row['top5_jaccard'],1)
            self.assertNotIn('accuracy',report['summary'][0])
            with self.assertRaises(ValueError):
                compare_review_sizes(self.rows,self.cafes,[10],'wifi',root/'run',Encoder(),Pipeline(),lambda *args:'doc')
        self.assertEqual(self.rows,original)


if __name__=='__main__':unittest.main()