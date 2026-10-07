"""
Cafind — Kaggle Dataset Loader
================================
Normalizes a Kaggle Google Maps reviews dataset into the same
cafes.csv / reviews.csv schema used by the scraper.

Place your Kaggle CSV file(s) in:
    machine_learning/data/raw/<your_dataset>.csv

Then run:
    python dataset_loader.py --input data/raw/your_dataset.csv

The script will:
  1. Auto-detect the column schema (handles common Kaggle formats)
  2. Filter for Malang cafes/coffee shops
  3. Deduplicate by cafe name
  4. Output into the canonical cafes.csv / reviews.csv format
  5. Print a summary of what was loaded

Why a separate loader instead of just using the CSV directly?
  Using a canonical schema means the preprocessing pipeline (preprocess.py),
  ABSA trainer (train_absa.py), and the backend API all share the same
  data contract regardless of the data source.
"""

import argparse
import csv
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent   # machine_learning/
DATA_RAW = BASE_DIR / "data" / "raw"
CAFES_CSV = DATA_RAW / "cafes.csv"
REVIEWS_CSV = DATA_RAW / "reviews.csv"

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("cafind.loader")

# ─── Malang filter ────────────────────────────────────────────────────────

Malang_KEYWORDS = [
    "Malang", "Kota Malang", "Kabupaten Malang", "Malang Raya", "Batu", "Kota Batu"
]

CAFE_KEYWORDS = [
    "cafe", "kafe", "coffee", "kopi", "kedai kopi",
    "warung kopi", "coffeehouse", "espresso",
]

# ─── Schema detection ─────────────────────────────────────────────────────────

# Maps from possible Kaggle column names → our canonical names
COLUMN_ALIASES = {
    "name":         ["name", "nama", "place_name", "cafe_name", "restaurant_name", "title"],
    "address":      ["address", "alamat", "location", "lokasi", "formatted_address"],
    "latitude":     ["latitude", "lat", "gps_lat", "lat_deg"],
    "longitude":    ["longitude", "lng", "lon", "gps_lng", "lon_deg"],
    "rating":       ["rating", "avg_rating", "average_rating", "bintang", "stars"],
    "review_count": ["review_count", "total_reviews", "num_reviews", "jumlah_ulasan", "reviews_count"],
    "categories":   ["categories", "category", "tipe", "type", "jenis", "subcategory"],
    "phone":        ["phone", "telepon", "no_hp", "contact"],
    "website":      ["website", "url", "web"],
    # Review fields
    "author_name":  ["author_name", "reviewer", "user_name", "username", "penulis", "author"],
    "review_text":  ["text", "review_text", "komentar", "ulasan", "comment", "content", "review"],
    "review_rating":["review_rating", "rating", "star", "bintang"],
    "published_at": ["published_at", "date", "review_date", "tanggal", "time", "timestamp"],
}

def detect_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Find which column in [df] matches the canonical field name."""
    aliases = COLUMN_ALIASES.get(canonical, [canonical])
    df_cols_lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in df_cols_lower:
            return df_cols_lower[alias.lower()]
    return None


def safe_float(val) -> Optional[float]:
    try:
        return float(str(val).replace(",", ".").strip())
    except (ValueError, TypeError):
        return None


def safe_int(val) -> int:
    try:
        return int(str(val).replace(".", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return 0


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_as_places_and_reviews(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Load a dataset where each row is a (place, review) pair — most common Kaggle format."""
    cafes_out: list[dict] = []
    reviews_out: list[dict] = []

    # Detect columns
    col = {
        field: detect_column(df, field)
        for field in [
            "name", "address", "latitude", "longitude", "rating",
            "review_count", "categories", "phone", "website",
            "author_name", "review_text", "review_rating", "published_at",
        ]
    }
    log.info("Detected column mapping:")
    for k, v in col.items():
        log.info(f"  {k:15s} -> {v or '(not found)'}")

    if col["name"] is None:
        raise ValueError("Could not detect a 'name' column. Use --col-map to specify manually.")

    # ── Deduplicate places ──
    place_groups = df.groupby(col["name"])
    seen_cafe_ids: dict[str, str] = {}   # name → cafe_id

    for cafe_name, group in place_groups:
        # ── Filter: must be in Malang ──
        address_val = str(group[col["address"]].iloc[0]).lower() if col["address"] else ""
        name_lower = str(cafe_name).lower()
        in_Malang = any(k in address_val or k in name_lower for k in Malang_KEYWORDS)

        # ── Filter: must be a cafe/coffee shop ──
        category_val = str(group[col["categories"]].iloc[0]).lower() if col["categories"] else ""
        is_cafe = any(k in name_lower or k in category_val for k in CAFE_KEYWORDS)

        if not in_Malang or not is_cafe:
            continue

        cafe_id = str(uuid.uuid4())
        seen_cafe_ids[cafe_name] = cafe_id

        # Use first row's place data
        row = group.iloc[0]

        cafes_out.append({
            "cafe_id": cafe_id,
            "name": str(cafe_name).strip(),
            "address": str(row[col["address"]]).strip() if col["address"] else "",
            "latitude": safe_float(row[col["latitude"]]) if col["latitude"] else None,
            "longitude": safe_float(row[col["longitude"]]) if col["longitude"] else None,
            "rating": safe_float(row[col["rating"]]) if col["rating"] else None,
            "review_count": safe_int(row[col["review_count"]]) if col["review_count"] else len(group),
            "categories": str(row[col["categories"]]).strip() if col["categories"] else "",
            "phone": str(row[col["phone"]]).strip() if col["phone"] else "",
            "website": str(row[col["website"]]).strip() if col["website"] else "",
            "scraped_at": datetime.utcnow().isoformat(),
            "source_query": "kaggle_dataset",
        })

        # ── Extract reviews from this group ──
        for _, rev_row in group.iterrows():
            text = str(rev_row[col["review_text"]]).strip() if col["review_text"] else ""
            if not text or text.lower() in ("nan", "none", ""):
                continue

            reviews_out.append({
                "review_id": str(uuid.uuid4()),
                "cafe_id": cafe_id,
                "author_name": str(rev_row[col["author_name"]]).strip() if col["author_name"] else "Anonymous",
                "rating": safe_float(rev_row[col["review_rating"]]) if col["review_rating"] else None,
                "text": text,
                "published_at": str(rev_row[col["published_at"]]).strip() if col["published_at"] else "",
                "likes": 0,
                "scraped_at": datetime.utcnow().isoformat(),
            })

    return cafes_out, reviews_out


def load_as_places_only(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Load a dataset where each row is a place (no per-row reviews)."""
    cafes_out: list[dict] = []
    col = {field: detect_column(df, field) for field in ["name", "address", "latitude", "longitude", "rating", "review_count", "categories", "phone", "website"]}

    for _, row in df.iterrows():
        if col["name"] is None:
            continue
        name = str(row[col["name"]]).strip()
        address_val = str(row[col["address"]]).lower() if col["address"] else ""
        in_Malang = any(k in address_val or k in name.lower() for k in Malang_KEYWORDS)
        is_cafe = any(k in name.lower() for k in CAFE_KEYWORDS)
        if not in_Malang or not is_cafe:
            continue

        cafes_out.append({
            "cafe_id": str(uuid.uuid4()),
            "name": name,
            "address": str(row[col["address"]]).strip() if col["address"] else "",
            "latitude": safe_float(row[col["latitude"]]) if col["latitude"] else None,
            "longitude": safe_float(row[col["longitude"]]) if col["longitude"] else None,
            "rating": safe_float(row[col["rating"]]) if col["rating"] else None,
            "review_count": safe_int(row[col["review_count"]]) if col["review_count"] else 0,
            "categories": str(row[col["categories"]]).strip() if col["categories"] else "",
            "phone": str(row[col["phone"]]).strip() if col["phone"] else "",
            "website": str(row[col["website"]]).strip() if col["website"] else "",
            "scraped_at": datetime.utcnow().isoformat(),
            "source_query": "kaggle_dataset",
        })

    return cafes_out, []


def write_output(cafes: list[dict], reviews: list[dict], append: bool) -> None:
    """Write (or append) cafes and reviews to the canonical CSV files."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    with open(CAFES_CSV, mode, newline="", encoding="utf-8") as f:
        # pyrefly: ignore [missing-import]
        from machine_learning_constants import CAFE_FIELDS
        writer = csv.DictWriter(f, fieldnames=CAFE_FIELDS)
        if not append:
            writer.writeheader()
        writer.writerows(cafes)

    with open(REVIEWS_CSV, mode, newline="", encoding="utf-8") as f:
        # pyrefly: ignore [missing-import]
        from machine_learning_constants import REVIEW_FIELDS
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        if not append:
            writer.writeheader()
        writer.writerows(reviews)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a Kaggle dataset into Cafind's canonical CSV schema"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the Kaggle CSV file (relative to machine_learning/ or absolute)"
    )
    parser.add_argument(
        "--mode",
        choices=["places_and_reviews", "places_only"],
        default="places_and_reviews",
        help="Dataset structure: 'places_and_reviews' (one row = one review) or 'places_only'"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing cafes.csv / reviews.csv instead of overwriting"
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV file encoding (default: utf-8). Try 'latin-1' if you get UnicodeDecodeError"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path

    if not input_path.exists():
        log.error(f"File not found: {input_path}")
        sys.exit(1)

    log.info(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path, encoding=args.encoding, low_memory=False)
    log.info(f"  Rows: {len(df):,} | Columns: {list(df.columns)}")

    if args.mode == "places_and_reviews":
        cafes, reviews = load_as_places_and_reviews(df)
    else:
        cafes, reviews = load_as_places_only(df)

    # Write using simple CSV (avoid circular import from constants)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    CAFE_FIELDS = ["cafe_id","name","address","latitude","longitude","rating","review_count","categories","phone","website","scraped_at","source_query"]
    REVIEW_FIELDS = ["review_id","cafe_id","author_name","rating","text","published_at","likes","scraped_at"]

    with open(CAFES_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAFE_FIELDS)
        if not args.append:
            writer.writeheader()
        writer.writerows(cafes)

    with open(REVIEWS_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        if not args.append:
            writer.writeheader()
        writer.writerows(reviews)

    log.info("\nDataset loaded successfully!")
    log.info(f"   Cafes   : {len(cafes):,}  -> {CAFES_CSV}")
    log.info(f"   Reviews : {len(reviews):,}  -> {REVIEWS_CSV}")

    if cafes:
        log.info("\nSample cafes loaded:")
        for c in cafes[:5]:
            log.info(f"  * {c['name']} | {c['address'][:50]} | Rating:{c['rating']}")


if __name__ == "__main__":
    main()
