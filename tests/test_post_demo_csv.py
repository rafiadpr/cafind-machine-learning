from pathlib import Path
import tempfile
import unittest
from src.csv_validation import validate_cafe_csv

class CsvTests(unittest.TestCase):
    def check(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cafes.csv'
            path.write_text(text, encoding='utf-8')
            return validate_cafe_csv(path)

    def test_valid_csv(self):
        self.assertEqual(self.check('latitude,longitude,rating,review_count\n-7.9,112.6,4.5,10\n'), 1)

    def test_missing_columns_and_empty_rows(self):
        for data in ['name\nCafe\n', 'latitude,longitude,rating,review_count\n']:
            with self.assertRaises(ValueError):
                self.check(data)

    def test_invalid_values(self):
        for row in ['nan,112.6,4,10', '-7.9,181,4,10', '-7.9,112.6,6,10',
                    '-7.9,112.6,4,-1', '-7.9,112.6,4,1.5', ',112.6,4,10']:
            with self.assertRaises(ValueError):
                self.check('latitude,longitude,rating,review_count\n' + row + '\n')
