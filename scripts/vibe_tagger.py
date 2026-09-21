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
def generate_sample_reviews() -> pd.DataFrame:
    """Generates synthetic review records with characteristic atmosphere phrases."""
    sample_data = [
        ("c_001", "Kopi Titik Koma", "Tempatnya asik banget buat WFC, colokan banyak dan wifi kenceng pol!"),
        ("c_001", "Kopi Titik Koma", "Bisa nugas berjam-jam dengan tenang, kopinya enak."),
        ("c_002", "Karsa Garden Cafe", "Outdoor luas banget dengan pepohonan asri, enak sore-sore nongkrong di taman."),
        ("c_002", "Karsa Garden Cafe", "Area luar sejuk, pemandangannya asri dan smoking area nyaman."),
        ("c_003", "Kopi Kalibrasi", "Interior minimalis estetik banget! Banyak spot foto buat ootd."),
        ("c_003", "Kopi Kalibrasi", "Konsepnya cakep, sangat instagramable dan modern."),
        ("c_004", "Warung Kopi Rakyat", "Harganya murah meriah cocok buat kantong mahasiswa. Porsi banyak dan hemat."),
        ("c_004", "Warung Kopi Rakyat", "Tempat ngopi paling terjangkau, ramah kantong banget."),
        ("c_005", "Ruang Temu Studio", "Bisa nugas sekalian foto-foto karena estetik. Colokan ada di tiap meja."),
    ]
    return pd.DataFrame(sample_data, columns=["cafe_id", "cafe_name", "review_text"])


# ─── Tag Extraction Logic ─────────────────────────────────────────────────────
def compile_regex_patterns(lexicons: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    """Compiles keyword terms into regex patterns with word boundary constraints."""
    compiled = {}
    for vibe, keywords in lexicons.items():
        patterns = [
            re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            for kw in keywords
        ]
        compiled[vibe] = patterns
    return compiled


def extract_vibes_for_text(text: str, compiled_patterns: Dict[str, List[re.Pattern]]) -> Dict[str, int]:
    """Counts matching keyword occurrences per vibe category in a single text."""
    if not isinstance(text, str):
        return {v: 0 for v in compiled_patterns}

    text_lower = text.lower()
    counts = {}
    for vibe, patterns in compiled_patterns.items():
        match_count = sum(len(p.findall(text_lower)) for p in patterns)
        counts[vibe] = match_count
    return counts


