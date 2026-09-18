"""
Cafind — Text Preprocessing Pipeline
======================================
Cleans and normalizes raw review text for:
  1. IndoBERT fine-tuning (ABSA)
  2. SBERT embedding generation (semantic search)

Run after scraping / loading your dataset:
    python scripts/preprocess.py

Output:
    data/processed/cafes_clean.csv     — Cafe metadata, deduplicated
    data/processed/reviews_clean.csv   — Cleaned reviews with text stats
    data/processed/absa_train.csv      — ABSA training-ready format

Indonesian NLP notes:
    - We use Sastrawi for stemming (Indonesian morphological analyzer)
    - Slang/informal words are normalized via a custom slang dictionary
    - Emojis are converted to text sentiment markers, not stripped
      (emojis carry strong sentiment signals in Indonesian reviews)
"""

import logging
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

CAFES_CSV = DATA_RAW / "cafes.csv"
REVIEWS_CSV = DATA_RAW / "reviews.csv"

CAFES_CLEAN = DATA_PROCESSED / "cafes_clean.csv"
REVIEWS_CLEAN = DATA_PROCESSED / "reviews_clean.csv"
ABSA_TRAIN = DATA_PROCESSED / "absa_train.csv"

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("cafind.preprocess")

# ─── Indonesian Slang Dictionary (Informal → Formal) ─────────────────────────
# Common informal Indonesian words used in cafe reviews.
# This is intentionally short — extend it based on your corpus.

SLANG_MAP = {
    "gak": "tidak",
    "ga": "tidak",
    "g": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "gk": "tidak",
    "tp": "tapi",
    "tpi": "tapi",
    "udh": "sudah",
    "udah": "sudah",
    "blm": "belum",
    "blom": "belum",
    "bgt": "banget",
    "bngt": "banget",
    "bgt": "banget",
    "skrg": "sekarang",
    "sdh": "sudah",
    "sy": "saya",
    "ak": "aku",
    "gue": "aku",
    "gw": "aku",
    "krn": "karena",
    "karna": "karena",
    "krena": "karena",
    "klo": "kalau",
    "klu": "kalau",
    "kl": "kalau",
    "hrs": "harus",
    "sm": "sama",
    "bisa di": "bisa",
    "bs": "bisa",
    "yg": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "utk": "untuk",
    "tdk": "tidak",
    "jg": "juga",
    "lg": "lagi",
    "sdg": "sedang",
    "dr": "dari",
    "pd": "pada",
    "ttg": "tentang",
    "org": "orang",
    "tmpt": "tempat",
    "tmpat": "tempat",
    "wkwk": "",           # Laughter filler — remove
    "wkwkwk": "",
    "hehe": "",
    "hihi": "",
    "haha": "",
    "kwkwkw": "",
}

# ─── Emoji Sentiment Markers ──────────────────────────────────────────────────
# Replace common emojis with text sentiment signals instead of stripping them.
# This preserves sentiment information for the ABSA model.

EMOJI_MAP = {
    "😍": " sangat_suka ",
    "😊": " senang ",
    "🥰": " sangat_suka ",
    "👍": " bagus ",
    "❤️": " suka ",
    "💕": " suka ",
    "✨": " bagus ",
    "🔥": " populer ",
    "😋": " enak ",
    "☕": " kopi ",
    "😢": " sedih ",
    "😞": " kecewa ",
    "👎": " buruk ",
    "😡": " marah ",
    "💸": " mahal ",
    "😱": " terkejut ",
    "🤔": " ragu ",
}

# ─── Text Cleaning Functions ───────────────────────────────────────────────────

