"""
Cafind — Cafe Category & Vibe Tag Enrichment Script
=====================================================
Analyzes customer reviews from the database to extract multi-category / vibe tags
for every cafe in Kota Malang. Enriches `cafes.categories` from the raw single
tag (e.g. ["Kafe"]) into a rich list of 4–6 descriptive tags.

Extracted Tags Taxonomy:
  - WFC (Work From Cafe / Nugas)
  - Fast Wifi
  - Santai (Cozy / Nongkrong)
  - Estetik (Aesthetic / Instagramable)
  - Affordable (Ramah Kantong / Murah)
  - Pricey
  - Keluarga (Family Friendly / Luas)
  - Kopi Enak (Specialty Coffee)
  - Outdoor (Taman / Rooftop)
  - Pastry & Snacks

Usage:
    python machine_learning/scripts/enrich_cafe_tags.py
"""

import asyncio
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv

# Ensure backend and machine_learning are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
ML_DIR = PROJECT_ROOT / "machine_learning"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ML_DIR))

load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cafind.enrich_tags")

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.db.models import Cafe, Review


# ─── Vibe Detection Lexicons ──────────────────────────────────────────────────
TAG_LEXICONS: Dict[str, List[str]] = {
    "WFC": [
        "wfc", "nugas", "laptop", "colokan", "kerja", "meja kerja", "stopkontak",
        "stop kontak", "coworking", "buka laptop", "fokus", "meeting", "colokan banyak"
    ],
    "Fast Wifi": [
        "wifi", "wi-fi", "internet", "kencang", "cepat", "sinyal", "speed", "wifi kencang",
        "wifi cepat", "kenceng"
    ],
    "Santai": [
        "santai", "nongkrong", "cozy", "nyaman", "adem", "tenang", "kongkow", "rileks",
        "homy", "asri", "nyaman banget", "betah"
    ],
    "Estetik": [
        "estetik", "aesthetic", "foto", "instagramable", "instagenic", "cantik",
        "view", "pemandangan", "interior", "dekorasi", "spot foto", "ootd", "cakep", "kece"
    ],
    "Affordable": [
        "murah", "terjangkau", "ramah kantong", "hemat", "kantong mahasiswa",
        "kantong pelajar", "pas dikantong", "merakyat", "murah meriah", "harga bersahabat"
    ],
    "Pricey": [
        "mahal", "pricey", "agak mahal", "lumayan mahal", "kurang worth it", "kantong kering"
    ],
    "Keluarga": [
        "keluarga", "anak", "anak-anak", "luas", "family", "rombongan", "family friendly"
    ],
    "Kopi Enak": [
        "kopi enak", "ngopi", "manual brew", "espresso", "latte", "cappuccino",
        "kopi mantap", "kopinya", "racikan kopi", "barista"
    ],
    "Outdoor": [
        "outdoor", "rooftop", "taman", "halaman", "asri", "terbuka", "smoking area", "angin sepoi"
    ],
    "Pastry & Snacks": [
        "pastry", "croissant", "cake", "roti", "snack", "camilan", "makanan enak", "kue"
    ],
}


def compile_tag_patterns() -> Dict[str, List[re.Pattern]]:
    compiled = {}
    for tag, keywords in TAG_LEXICONS.items():
        patterns = []
        for kw in keywords:
            # Word boundary regex constraints
            pattern = re.compile(r"(?:\b|_)" + re.escape(kw) + r"(?:\b|_)", re.IGNORECASE)
            patterns.append(pattern)
        compiled[tag] = patterns
    return compiled


def get_async_engine():
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:andre123@localhost:5434/cafind"
    )
