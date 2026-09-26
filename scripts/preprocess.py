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

def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (handle composed/decomposed forms)."""
    return unicodedata.normalize("NFKC", text)


def replace_emojis(text: str) -> str:
    """Replace emojis with sentiment text markers before stripping."""
    for emoji, replacement in EMOJI_MAP.items():
        text = text.replace(emoji, replacement)
    # Strip any remaining unicode emoji (outside BMP)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def normalize_slang(text: str) -> str:
    """Replace informal Indonesian words with their formal equivalents.
    
    We apply word-boundary matching to avoid replacing substrings.
    e.g. 'dr' should not replace 'drama'.
    """
    words = text.split()
    return " ".join(SLANG_MAP.get(w.lower(), w) for w in words)


def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single review text.
    
    Order matters:
    1. Unicode normalization
    2. Emoji → text sentiment markers
    3. Lowercase
    4. Slang normalization
    5. Remove URLs, emails, phone numbers
    6. Remove special characters (keep Indonesian letters, numbers, spaces)
    7. Collapse whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = normalize_unicode(text)
    text = replace_emojis(text)
    text = text.lower()
    text = normalize_slang(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Remove phone numbers
    text = re.sub(r"(\+62|08)\d{8,11}", " ", text)
    # Keep Indonesian letters (a-z, accented), digits, and basic punctuation
    text = re.sub(r"[^a-z0-9\s.,!?'\-]", " ", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_text_stats(text: str) -> dict:
    """Compute basic statistics useful for filtering low-quality reviews."""
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "avg_word_len": round(sum(len(w) for w in words) / max(len(words), 1), 2),
    }

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def preprocess_cafes(df_cafes: pd.DataFrame) -> pd.DataFrame:
    """Clean cafe metadata — deduplicate and fill missing coordinates."""
    log.info(f"Processing {len(df_cafes):,} cafes...")

    # Deduplicate by name (keep highest review_count if duplicated)
    df_cafes = (
        df_cafes
        .sort_values("review_count", ascending=False)
        .drop_duplicates(subset="name", keep="first")
        .reset_index(drop=True)
    )

    # Clean name and address
    df_cafes["name"] = df_cafes["name"].str.strip().str.title()
    df_cafes["address"] = df_cafes["address"].fillna("").str.strip()
    df_cafes["categories"] = df_cafes["categories"].fillna("").str.lower().str.strip()

    # Flag cafes with missing coordinates (will need geocoding)
    df_cafes["needs_geocoding"] = df_cafes["latitude"].isna() | df_cafes["longitude"].isna()
    missing_coords = df_cafes["needs_geocoding"].sum()
    if missing_coords > 0:
        log.warning(f"  {missing_coords} cafes have missing coordinates — flag 'needs_geocoding' set")

    log.info(f"  Cafes after dedup: {len(df_cafes):,}")
    return df_cafes


def preprocess_reviews(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """Clean review text and compute quality metrics."""
    log.info(f"Processing {len(df_reviews):,} reviews...")

    tqdm.pandas(desc="Cleaning review text")

    # Apply full text cleaning pipeline
    df_reviews["text_clean"] = df_reviews["text"].progress_apply(clean_text)

    # Compute text stats for quality filtering
    stats = df_reviews["text_clean"].apply(compute_text_stats).apply(pd.Series)
    df_reviews = pd.concat([df_reviews, stats], axis=1)

    # Filter: remove reviews that are too short (< 5 words) or empty after cleaning
    before = len(df_reviews)
    df_reviews = df_reviews[df_reviews["word_count"] >= 5].reset_index(drop=True)
    log.info(f"  Removed {before - len(df_reviews):,} too-short reviews (< 5 words)")

    # Normalize rating to float
    df_reviews["rating"] = pd.to_numeric(df_reviews["rating"], errors="coerce")

    log.info(f"  Reviews after cleaning: {len(df_reviews):,}")
    log.info(f"  Avg review length: {df_reviews['word_count'].mean():.1f} words")

    return df_reviews


def create_absa_template(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Create the ABSA training template CSV.

    Format expected by IndoBERT fine-tuning (train_absa.py):
        review_id, cafe_id, text_clean, aspect, sentiment, confidence

    At this stage, 'aspect' and 'sentiment' are empty — they will be filled
    either by manual labeling (data/labeled/) or by weak supervision rules.

    Common ABSA aspects for cafes (domain-specific):
        - wifi         (speed, stability)
        - harga        (price, value for money)
        - suasana      (ambience, atmosphere)
        - pelayanan    (service, staff)
        - makanan      (food quality, taste)
        - minuman/kopi (coffee quality)
        - tempat       (seating, space, capacity)
        - kebersihan   (cleanliness)
        - lokasi       (accessibility, parking)

    These aspects come from analysis of common Indonesian cafe review keywords.
    """
    df_absa = df_reviews[["review_id", "cafe_id", "text_clean", "rating"]].copy()
    df_absa["aspect"] = ""        # To be labeled
    df_absa["sentiment"] = ""     # To be labeled
    df_absa["confidence"] = None  # Model confidence (filled during inference)
    return df_absa


def run() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # ── Load raw data ──
    if not CAFES_CSV.exists() or not REVIEWS_CSV.exists():
        log.error(
            "Raw data not found! Run the scraper first:\n"
            "  python scripts/scraper.py\n"
            "Or load a Kaggle dataset:\n"
            "  python scripts/dataset_loader.py --input data/raw/your_dataset.csv"
