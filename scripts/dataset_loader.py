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
