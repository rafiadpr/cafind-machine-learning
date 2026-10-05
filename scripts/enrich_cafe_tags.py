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
