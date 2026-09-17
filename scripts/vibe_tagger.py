"""
Cafind — Cafe Vibe & Atmosphere Tag Extractor
================================================
Extracts experiential vibe tags (e.g., WFC-Friendly, Aesthetic, Outdoor, Affordable)
from Google Maps review texts using Indonesian domain-specific keyword lexicons
and rule-based frequency thresholding.

Supported Vibe Categories:
  1. WFC (Work From Cafe / Nugas)
  2. Aesthetic (Instagrammable / Fotogenik)
  3. Outdoor (Open-air / Garden / Rooftop)
  4. Affordable (Budget-friendly / Ramah Mahasiswa)

Usage:
    python scripts/vibe_tagger.py
    python scripts/vibe_tagger.py --input data/raw/reviews.csv --output data/processed/cafes_with_vibes.csv
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cafind.vibe_tagger")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

DEFAULT_REVIEWS_FILE = DATA_RAW / "reviews.csv"
DEFAULT_OUTPUT_FILE = DATA_PROCESSED / "cafes_with_vibes.csv"


# ─── Vibe Keyword Lexicons (Indonesian Domain Terms) ───────────────────────────
VIBE_LEXICONS: Dict[str, List[str]] = {
    "WFC": [
        "wfc", "nugas", "colokan", "stopkontak", "stop kontak", "wifi kencang", "wifi cepat",
        "laptop", "kerja", "meja luas", "kursi nyaman", "tenang", "hening", "coworking",
        "charger", "fokus", "buka laptop"
    ],
    "Aesthetic": [
        "estetik", "aesthetic", "instagramable", "instagenic", "foto", "cantik", "cakep",
        "kece", "interior", "dekorasi", "vintage", "industrial", "minimalis", "ootd",
        "spot foto", "bagus banget", "konsep unik"
    ],
    "Outdoor": [
        "outdoor", "luar ruangan", "taman", "halaman", "rooftop", "asri", "sejuk",
        "smoking area", "angin sepoi", "pepohonan", "pemandangan", "sunset", "alam",
        "breeze", "teras"
    ],
    "Affordable": [
        "murah", "terjangkau", "ramah kantong", "kantong mahasiswa", "hemat", "pas di kantong",
        "ramah dompet", "harga bersahabat", "worth it", "porsi banyak", "murmer", "merakyat"
    ]
}


# ─── Synthetic Data Generator (Fallback) ──────────────────────────────────────
