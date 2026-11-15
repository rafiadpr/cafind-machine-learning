"""Export a consistent, read-only database snapshot in the scraper CSV format."""

import argparse
import asyncio
import csv
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .scraper_storage import CAFE_FIELDS, REVIEW_FIELDS, _write_rows
else:
    from scraper_storage import CAFE_FIELDS, REVIEW_FIELDS, _write_rows

ROOT = Path(__file__).resolve().parents[2]


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, list):
        return ", ".join(value)
    return value


def prepare_rows(cafes, reviews):
    cafe_rows = [{field: csv_value(row.get(field)) for field in CAFE_FIELDS} for row in cafes]
    review_rows = [{field: csv_value(row.get(field)) for field in REVIEW_FIELDS} for row in reviews]
    cafe_ids = {row["cafe_id"] for row in cafe_rows}
    if len(cafe_ids) != len(cafe_rows):
        raise ValueError("Duplicate cafe IDs in export")
    if len({row["review_id"] for row in review_rows}) != len(review_rows):
        raise ValueError("Duplicate review IDs in export")
    if any(row["cafe_id"] not in cafe_ids for row in review_rows):
        raise ValueError("Review refers to a cafe missing from the export")
    return cafe_rows, review_rows


def write_snapshot(directory, cafes, reviews):
    """Validate before writing; retain a backup when replacing existing CSV files."""
    cafes, reviews = prepare_rows(cafes, reviews)
    directory.mkdir(parents=True, exist_ok=True)
    existing = [directory / name for name in ("cafes.csv", "reviews.csv") if (directory / name).exists()]
    backup = None
    if existing:
        backup = directory / ".csv_backups" / f"before_db_export_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
        backup.mkdir(parents=True)
        for path in existing:
            shutil.copy2(path, backup / path.name)
    for name, fields, rows in (("cafes.csv", CAFE_FIELDS, cafes), ("reviews.csv", REVIEW_FIELDS, reviews)):
        _write_rows(directory / name, fields, rows)
        with (directory / name).open(newline="", encoding="utf-8") as file:
            saved = list(csv.DictReader(file))
        expected = [{key: str(value) for key, value in row.items()} for row in rows]
        if saved != expected:
            raise RuntimeError(f"CSV verification failed: {directory / name}")
    report = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "cafes": len(cafes), "reviews": len(reviews),
        "backup": str(backup.resolve()) if backup else None,
        "notes": [
            "CSV IDs use database IDs, which differ from original scraper IDs.",
            "Review text uses raw_text when present, otherwise text.",
            "scraped_at uses database created_at; source_query is unavailable in the database.",
            "published_at uses the stored database date, which may be an ingestion-date fallback.",
            "Only data already stored in the database can be recovered.",
        ],
    }
    (directory / "database_export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


async def export_database(directory):
    import asyncpg
    from dotenv import load_dotenv

    load_dotenv(ROOT / "backend" / ".env")
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL tidak tersedia; periksa backend/.env")
    url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    connection = await asyncpg.connect(url, timeout=15)
    try:
        async with connection.transaction(isolation="repeatable_read", readonly=True):
            cafes = await connection.fetch("""
                SELECT id AS cafe_id, name, address, latitude, longitude, rating,
                       review_count, categories, phone, website, price_range, price_min,
                       price_max, opening_hours, created_at AS scraped_at
                FROM cafes ORDER BY id
            """)
            reviews = await connection.fetch("""
                SELECT id AS review_id, cafe_id, author_name, rating,
                       COALESCE(raw_text, text) AS text, published_at, likes,
                       created_at AS scraped_at
                FROM reviews ORDER BY cafe_id, id
            """)
    finally:
        await connection.close()
    return write_snapshot(directory, cafes, reviews)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "machine_learning" / "data" / "db_export" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    args = parser.parse_args()
    report = asyncio.run(export_database(args.output_dir))
    print(f"Export verified: {report['cafes']} cafes, {report['reviews']} reviews")
    print(f"Output: {args.output_dir.resolve()}")
    if report["backup"]:
        print(f"Previous CSV backup: {report['backup']}")


if __name__ == "__main__":
    main()
