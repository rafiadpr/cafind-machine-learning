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

