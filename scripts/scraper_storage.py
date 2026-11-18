"""Validate, migrate and save scraper CSV data with recoverable writes."""

import csv
import json
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("cafind.scraper")
CAFE_FIELDS = [
    "cafe_id", "name", "address", "latitude", "longitude", "rating", "review_count",
    "categories", "phone", "website", "price_range", "price_min", "price_max",
    "opening_hours", "scraped_at", "source_query",
]
REVIEW_FIELDS = ["review_id", "cafe_id", "author_name", "rating", "text", "published_at", "likes", "scraped_at"]
OPTIONAL_CAFE_FIELDS = {"price_range", "price_min", "price_max", "opening_hours"}


class CsvSchemaError(RuntimeError):
    """A CSV cannot be interpreted without risking loss of existing values."""


def _read_rows(path: Path, fields: list[str], *, legacy_cafes: bool = False) -> tuple[list[dict], bool]:
    if not path.exists():
        return [], False
    legacy = [field for field in fields if field not in OPTIONAL_CAFE_FIELDS]
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file, strict=True)
        header = next(reader, [])
        if not header or len(set(header)) != len(header) or set(header) - set(fields):
            raise CsvSchemaError(f"Invalid CSV header in {path}; original file was preserved.")
        migrated = header != fields
        rows = []
        for values in reader:
            if not values:
                continue
            columns = header
            if len(values) != len(header):
                if legacy_cafes and header == legacy and len(values) == len(fields):
                    columns = fields
                elif legacy_cafes and header == fields and len(values) == len(legacy):
                    columns = legacy
                else:
                    raise CsvSchemaError(f"CSV column mismatch in {path} near line {reader.line_num}: expected {len(header)}, got {len(values)}. Original file was preserved.")
                migrated = True
            row = dict(zip(columns, values))
            rows.append({field: row.get(field, "") for field in fields})
    return rows, migrated


def _write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def ensure_cafe_csv_schema(path: Path, fields: list[str]) -> None:
    rows, migrated = _read_rows(path, fields, legacy_cafes=True)
    if path.exists() and migrated:
        backup_dir = path.parent / ".csv_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup = backup_dir / f"{path.stem}_{stamp}_{uuid.uuid4().hex[:8]}.csv"
        shutil.copy2(path, backup)
        _write_rows(path, fields, rows)
        log.warning("Migrated CSV schema: %s (%s rows). Original backup: %s", path, len(rows), backup)
    elif not path.exists():
        _write_rows(path, fields, [])


def append_review_rows(path: Path, reviews: list[dict], fields: list[str]) -> int:
    """Write a complete replacement atomically; recovery replay is idempotent by review ID."""
    if not reviews:
        return 0
    rows, _ = _read_rows(path, fields)
    ids = {row["review_id"] for row in rows}
    added = 0
    for review in reviews:
        if review["review_id"] not in ids:
            rows.append(review)
            ids.add(review["review_id"])
            added += 1
    if added:
        _write_rows(path, fields, rows)
    return added


def save_csv_recovery(directory: Path, cafe: dict, reviews: list[dict], error: Exception) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps({"cafe": cafe, "reviews": reviews, "error": str(error)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def replay_csv_recovery(directory: Path, output_dir: Path) -> tuple[int, int]:
    """Restore failed local writes without sending anything to the backend."""
    ensure_cafe_csv_schema(output_dir / "cafes.csv", CAFE_FIELDS)
    recovered, reviews_added = 0, 0
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        upsert_cafe_csv(output_dir / "cafes.csv", payload["cafe"], CAFE_FIELDS)
        reviews_added += append_review_rows(output_dir / "reviews.csv", payload["reviews"], REVIEW_FIELDS)
        archive = directory / "processed"
        archive.mkdir(exist_ok=True)
        path.rename(archive / path.name)
        recovered += 1
    return recovered, reviews_added


def cafe_key(cafe: dict) -> tuple[str, str]:
    return (cafe.get("name", "").strip().casefold(), cafe.get("address", "").strip().casefold())


@dataclass(frozen=True)
class ExistingCafe:
    cafe_id: str
    has_reviews: bool


def upsert_cafe_csv(path: Path, cafe: dict, fields: list[str]) -> None:
    """Refresh an existing cafe by ID without duplicating it or erasing known fields with nulls."""
    ensure_cafe_csv_schema(path, fields)
    existing, _ = _read_rows(path, fields, legacy_cafes=True)
    rows = []
    found = False
    for row in existing:
        if row.get("cafe_id") == cafe["cafe_id"]:
            if found:
                continue
            row.update({k: v for k, v in cafe.items() if v not in (None, "", [])})
            found = True
        rows.append(row)
    if not found:
        rows.append(cafe)
    _write_rows(path, fields, rows)


def load_resume_state(cafes_path: Path, reviews_path: Path) -> dict[tuple[str, str], ExistingCafe]:
    """Prefer a reviewed identity if historical runs produced duplicate cafe rows."""
    reviewed_ids = set()
    if reviews_path.exists():
        with reviews_path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                if row.get("text", "").strip():
                    reviewed_ids.add(row.get("cafe_id", ""))

    existing = {}
    if cafes_path.exists():
        with cafes_path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                cafe_id = row.get("cafe_id", "").strip()
                if not cafe_id or not row.get("name", "").strip():
                    continue
                key = cafe_key(row)
                state = ExistingCafe(cafe_id, cafe_id in reviewed_ids)
                if key not in existing or state.has_reviews:
                    existing[key] = state
    return existing
