import csv
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from machine_learning.scripts.export_database_csv import prepare_rows, write_snapshot


class DatabaseCsvExportTests(unittest.TestCase):
    def test_export_preserves_ids_unicode_multiline_prices_and_backs_up_previous_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reviews.csv").write_bytes(b"previous CSV\r\n")
            cafe_id, review_id = uuid.uuid4(), uuid.uuid4()
            date = datetime(2026, 10, 17, tzinfo=timezone.utc)
            report = write_snapshot(root, [{"cafe_id": cafe_id, "name": "Kopi", "categories": ["Cafe", "Coffee shop"], "price_min": 25000.0, "opening_hours": "Senin: 07–21"}], [{"review_id": review_id, "cafe_id": cafe_id, "text": "Enak, 👍\nMurah", "scraped_at": date}])
            with (root / "reviews.csv").open(newline="", encoding="utf-8") as file:
                review = next(csv.DictReader(file))
            self.assertEqual(review["text"], "Enak, 👍\nMurah")
            self.assertEqual(review["review_id"], str(review_id))
            self.assertEqual(review["cafe_id"], str(cafe_id))
            self.assertEqual(review["scraped_at"], date.isoformat())
            self.assertEqual((Path(report["backup"]) / "reviews.csv").read_bytes(), b"previous CSV\r\n")
            self.assertEqual(report["reviews"], 1)
            with (root / "cafes.csv").open(newline="", encoding="utf-8") as file:
                cafe = next(csv.DictReader(file))
            self.assertEqual(cafe["categories"], "Cafe, Coffee shop")
            self.assertEqual(cafe["price_min"], "25000.0")
            self.assertEqual(cafe["source_query"], "")

    def test_missing_parent_is_rejected_before_overwriting_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "reviews.csv"
            path.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "missing"):
                write_snapshot(root, [], [{"review_id": "r1", "cafe_id": "missing"}])
            self.assertEqual(path.read_bytes(), b"original")

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate cafe"):
            prepare_rows([{"cafe_id": "c1"}, {"cafe_id": "c1"}], [])
        with self.assertRaisesRegex(ValueError, "Duplicate review"):
            prepare_rows([{"cafe_id": "c1"}], [{"review_id": "r1", "cafe_id": "c1"}] * 2)
