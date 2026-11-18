import csv
import tempfile
import unittest
from pathlib import Path

from machine_learning.scripts import scraper_storage as storage


class CsvStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "cafes.csv"
        self.legacy = [f for f in storage.CAFE_FIELDS if f not in storage.OPTIONAL_CAFE_FIELDS]

    def write(self, header, rows):
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)

    def test_mixed_legacy_header_preserves_every_value_and_original_backup(self):
        old = [f"old-{f}" for f in self.legacy]
        new = [f"new-{f}" for f in storage.CAFE_FIELDS]
        new[storage.CAFE_FIELDS.index("opening_hours")] = "Senin: 07.00–23.00,\nSelasa: Tutup"
        self.write(self.legacy, [old, new])
        original = self.path.read_bytes()
        storage.ensure_cafe_csv_schema(self.path, storage.CAFE_FIELDS)
        with self.path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, storage.CAFE_FIELDS)
        self.assertEqual(rows[0], {f: dict(zip(self.legacy, old)).get(f, "") for f in storage.CAFE_FIELDS})
        self.assertEqual(rows[1], dict(zip(storage.CAFE_FIELDS, new)))
        self.assertEqual(next((self.root / ".csv_backups").glob("*.csv")).read_bytes(), original)
        storage.ensure_cafe_csv_schema(self.path, storage.CAFE_FIELDS)
        self.assertEqual(len(list((self.root / ".csv_backups").glob("*.csv"))), 1)

    def test_current_header_with_legacy_row_does_not_shift_timestamp_into_price(self):
        self.write(storage.CAFE_FIELDS, [[f"old-{f}" for f in self.legacy]])
        storage.ensure_cafe_csv_schema(self.path, storage.CAFE_FIELDS)
        with self.path.open(newline="", encoding="utf-8") as file:
            row = next(csv.DictReader(file))
        self.assertEqual(row["scraped_at"], "old-scraped_at")
        self.assertEqual(row["price_range"], "")

    def test_unknown_row_width_preserves_original(self):
        self.write(self.legacy, [["x"] * 13])
        original = self.path.read_bytes()
        with self.assertRaises(storage.CsvSchemaError):
            storage.upsert_cafe_csv(self.path, {"cafe_id": "c1"}, storage.CAFE_FIELDS)
        self.assertEqual(self.path.read_bytes(), original)

    def test_review_write_failure_does_not_truncate_existing_csv(self):
        path = self.root / "reviews.csv"
        storage.append_review_rows(path, [{"review_id": "r1", "text": "Kopi, enak\nsekali"}], storage.REVIEW_FIELDS)
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            storage.append_review_rows(path, [{"review_id": "r2", "unexpected": "value"}], storage.REVIEW_FIELDS)
        self.assertEqual(path.read_bytes(), original)

    def test_recovery_replays_partial_write_without_duplicate_reviews(self):
        cafe = {"cafe_id": "c1", "name": "Cafe"}
        reviews = [{"review_id": "r1", "cafe_id": "c1", "text": "Enak\nsekali"}]
        recovery = self.root / "csv_recovery"
        storage.save_csv_recovery(recovery, cafe, reviews, RuntimeError("locked"))
        storage.append_review_rows(self.root / "reviews.csv", reviews, storage.REVIEW_FIELDS)
        self.assertEqual(storage.replay_csv_recovery(recovery, self.root), (1, 0))
        self.assertEqual(storage.replay_csv_recovery(recovery, self.root), (0, 0))
        self.assertEqual(len(list((recovery / "processed").glob("*.json"))), 1)
        with (self.root / "reviews.csv").open(newline="", encoding="utf-8") as file:
            self.assertEqual(len(list(csv.DictReader(file))), 1)
