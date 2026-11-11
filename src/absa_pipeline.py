"""
Cafind — Aspect-Based Sentiment Analysis (ABSA) Pipeline
=========================================================
Extracts 4 core cafe aspects and calculates sentiment polarity & confidence:
  1. ambience      (suasana, estetik, cozy, bising, sejuk, outdoor)
  2. facility      (wifi, colokan, stopkontak, meja kerja, parkir, musholla)
  3. price         (harga, ramah kantong, mahal, worth it)
  4. food_beverage (kopi, rasa makanan, menu, pastry, manis, pahit)

Supports hybrid extraction with Indonesian domain-specific sentiment lexicons,
negation detection, and aspect clause segmentation.
Directly syncs results with the PostgreSQL `aspect_sentiments` table.
"""

import logging
import re
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cafind.absa")

# ─── 4 Core ABSA Aspects & Keywords ──────────────────────────────────────────
ASPECT_LEXICONS: Dict[str, List[str]] = {
    "ambience": [
        "suasana", "tempat", "estetik", "aesthetic", "esthetic", "cozy", "nyaman",
        "tenang", "hening", "damai", "bising", "ramai", "berisik", "sejuk", "adem",
        "panas", "pengap", "outdoor", "indoor", "luar ruangan", "dalam ruangan",
        "taman", "garden", "rooftop", "view", "pemandangan", "dekorasi", "interior",
        "spot foto", "foto", "ootd", "instagramable", "instagenic", "vintage",
        "industrial", "konsep", "bersih", "kotor", "rapi", "nuansa", "vibes", "vibe"
    ],
    "facility": [
        "wifi", "wi-fi", "internet", "koneksi", "colokan", "stopkontak", "stop kontak",
        "steker", "listrik", "charger", "meja", "meja kerja", "kursi", "tempat duduk",
        "nugas", "wfc", "kerja", "laptop", "parkir", "parkiran", "tukang parkir",
        "musholla", "mushola", "tempat ibadah", "sholat", "toilet", "kamar mandi",
        "wc", "ac", "pendingin", "kipas", "smoking area", "area merokok"
    ],
    "price": [
        "harga", "biaya", "bayar", "tarif", "murah", "terjangkau", "ramah kantong",
        "ramah dompet", "kantong mahasiswa", "hemat", "murmer", "bersahabat",
        "merakyat", "pas di kantong", "mahal", "pricey", "kemahalan", "kantong jebol",
        "worth it", "sebanding", "padan", "lumayan mahal", "agak mahal", "standar", "kantong"
    ],
    "food_beverage": [
        "kopi", "coffee", "espresso", "latte", "americano", "cappuccino", "manual brew",
        "v60", "mocktail", "teh", "tea", "minuman", "makanan", "menu", "cemilan",
        "snack", "pastry", "croissant", "kue", "roti", "nasi", "mie", "makan", "minum",
        "rasa", "taste", "enak", "lezat", "nikmat", "mantap", "gurih", "manis",
        "pahit", "asam", "kecut", "hambar", "sedap", "porsi"
    ],
}

# ─── Sentiment Polarities ────────────────────────────────────────────────────
POSITIVE_TERMS: set[str] = {
    "enak", "lezat", "nikmat", "mantap", "juara", "segar", "pas", "kencang", "kenceng",
    "cepat", "banyak", "melimpah", "lengkap", "luas", "lega", "nyaman", "tenang",
    "adem", "sejuk", "asri", "estetik", "bagus", "cantik", "kece", "keren", "cozy",
    "murah", "terjangkau", "hemat", "worth it", "ramah", "bersih", "rapi",
    "recommended", "rekomendasi", "suka", "puas", "top", "oke", "ok", "senang",
    "bersahabat", "wangi", "nagih", "sempurna", "love", "favorit", "fresh", "lezatos"
}

NEGATIVE_TERMS: set[str] = {
    "mahal", "kemahalan", "pricey", "kurang", "lemot", "lelet", "lambat", "rusak",
    "sedikit", "minim", "sempit", "panas", "pengap", "bising", "berisik", "kotor",
    "bau", "pahit", "hambar", "asam", "kecewa", "buruk", "jelek", "parah",
    "keasinan", "kelamaan", "jorok", "gelap", "bau", "ribet", "padat", "engap",
    "kemiskinan", "boros", "rugi"
}

NEGATION_TERMS: set[str] = {
    "tidak", "tak", "nggak", "ngga", "gak", "ga", "gk", "tdk", "bukan", "belum", "blm", "kurang"
}

INTENSIFIERS: set[str] = {
    "sangat", "banget", "bgt", "bngt", "pol", "sekali", "amat", "super", "terlalu", "parah"
}


@dataclass
class AspectResult:
    aspect: str
    sentiment: str  # positive, neutral, negative
    confidence: float
    evidence: str


class ABSAPipeline:
    """Aspect-Based Sentiment Analysis engine for Indonesian cafe reviews."""

    def __init__(self):
        self.compiled_aspects = self._compile_aspects()

    def _compile_aspects(self) -> Dict[str, List[re.Pattern]]:
        compiled = {}
        for aspect, terms in ASPECT_LEXICONS.items():
            compiled[aspect] = [
                re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                for kw in terms
            ]
        return compiled

    def _split_into_clauses(self, text: str) -> List[str]:
        """Splits full review text into semantic clauses using punctuation and conjunctions."""
        # Replace common conjunctions with delimiters
        delims = r"[.,;!?\n\r]+|\b(?:tetapi|tapi|namun|sedangkan|padahal|walaupun|meskipun)\b"
        raw_clauses = re.split(delims, text, flags=re.IGNORECASE)
        cleaned = [c.strip() for c in raw_clauses if len(c.strip()) > 2]
        return cleaned if cleaned else [text.strip()]

    def analyze_clause_sentiment(self, clause: str, target_aspect: str) -> Tuple[str, float]:
        """
        Calculates sentiment polarity and confidence for a single clause
        targeting a specific aspect.
        """
        tokens = re.findall(r"\b\w+\b", clause.lower())
        if not tokens:
            return "neutral", 0.60

        pos_score = 0.0
        neg_score = 0.0

        for i, word in enumerate(tokens):
            # Check negation window (previous 1-2 words)
            prev_window = tokens[max(0, i - 2):i]
            is_negated = any(neg in NEGATION_TERMS for neg in prev_window)

            # Check intensifier window
            next_window = tokens[i + 1:min(len(tokens), i + 3)]
            is_intensified = (
                any(intens in INTENSIFIERS for intens in prev_window) or
                any(intens in INTENSIFIERS for intens in next_window)
            )
            weight = 1.4 if is_intensified else 1.0

            if word in POSITIVE_TERMS:
                if is_negated:
                    neg_score += 1.2 * weight
                else:
                    pos_score += 1.0 * weight
            elif word in NEGATIVE_TERMS:
                if is_negated:
                    # e.g., "tidak mahal", "gak bising" -> positive
                    pos_score += 1.1 * weight
                else:
                    neg_score += 1.2 * weight

        # Special domain heuristic: "colokan banyak", "wifi kencang"
        if target_aspect == "facility":
            if "banyak" in tokens and any(w in tokens for w in ["colokan", "stopkontak", "kursi", "meja"]):
                pos_score += 1.5
            if "kencang" in tokens or "kenceng" in tokens or "cepat" in tokens:
                pos_score += 1.5
            if "susah" in tokens or "jarang" in tokens or "dikit" in tokens:
                neg_score += 1.5

        if target_aspect == "price":
            if any(w in tokens for w in ["terjangkau", "murah", "ramah", "hemat", "pas"]):
                if not any(neg in NEGATION_TERMS for neg in tokens[:3]):
                    pos_score += 1.5
            if any(w in tokens for w in ["mahal", "pricey", "kemahalan"]):
                if not any(neg in NEGATION_TERMS for neg in tokens[:3]):
                    neg_score += 1.5

        diff = pos_score - neg_score
        total = pos_score + neg_score

        if total == 0:
            return "neutral", 0.70

        if diff > 0.3:
            conf = min(0.98, 0.75 + (diff / (total + 1.0)) * 0.23)
            return "positive", round(conf, 4)
        elif diff < -0.3:
            conf = min(0.98, 0.75 + (abs(diff) / (total + 1.0)) * 0.23)
            return "negative", round(conf, 4)
        else:
            return "neutral", 0.72

    def extract_aspects(self, text: str) -> List[AspectResult]:
        """
        Extracts aspects mentioned in the review text and determines their polarity.
        Returns a list of AspectResult (up to 4, one per detected aspect).
        """
        if not text or not isinstance(text, str):
            return []

        clauses = self._split_into_clauses(text)
        aspect_matches: Dict[str, List[Tuple[str, str, float]]] = {
            a: [] for a in ASPECT_LEXICONS
        }

        for clause in clauses:
            for aspect, patterns in self.compiled_aspects.items():
                if any(p.search(clause) for p in patterns):
                    sentiment, conf = self.analyze_clause_sentiment(clause, aspect)
                    aspect_matches[aspect].append((clause, sentiment, conf))

        results: List[AspectResult] = []

        for aspect, matches in aspect_matches.items():
            if not matches:
                continue

            # Determine dominant sentiment across matching clauses
            pos_confs = [c for cl, s, c in matches if s == "positive"]
            neg_confs = [c for cl, s, c in matches if s == "negative"]
            neu_confs = [c for cl, s, c in matches if s == "neutral"]

            if len(pos_confs) > len(neg_confs):
                final_sentiment = "positive"
                final_conf = max(pos_confs)
            elif len(neg_confs) > len(pos_confs):
                final_sentiment = "negative"
                final_conf = max(neg_confs)
            elif pos_confs and not neg_confs:
                final_sentiment = "positive"
                final_conf = max(pos_confs)
            elif neg_confs and not pos_confs:
                final_sentiment = "negative"
                final_conf = max(neg_confs)
            else:
                final_sentiment = "neutral"
                final_conf = 0.70

            evidence_clause = matches[0][0]
            results.append(
                AspectResult(
                    aspect=aspect,
                    sentiment=final_sentiment,
                    confidence=final_conf,
                    evidence=evidence_clause
                )
            )

        return results


# ─── Database Sync Helpers (Asyncpg / SQLAlchemy) ───────────────────────────
async def process_reviews_and_save_to_db(
    db_session: Any,
    limit: Optional[int] = None,
    batch_size: int = 100
) -> Dict[str, int]:
    """
    Reads reviews from the database without aspect_sentiments (or is_processed=False),
    extracts aspects using ABSAPipeline, and inserts into `aspect_sentiments`.
    Updates review `is_processed = True`.
    """
    from sqlalchemy import select
    from app.db.models import AspectEnum, AspectSentiment, Review, SentimentEnum

    pipeline = ABSAPipeline()

    # Query reviews needing ABSA processing
    stmt = (
        select(Review)
        .where(Review.is_processed == False)  # noqa: E712
        .order_by(Review.created_at.asc())
    )
    if limit:
        stmt = stmt.limit(limit)

    result = await db_session.execute(stmt)
    unprocessed_reviews = result.scalars().all()

    total_reviews = len(unprocessed_reviews)
    logger.info(f"Found {total_reviews} reviews to process for ABSA.")

    if total_reviews == 0:
        return {"processed_reviews": 0, "inserted_aspects": 0}

    inserted_aspects_count = 0
    aspect_records = []

    for rev in unprocessed_reviews:
        aspect_results = pipeline.extract_aspects(rev.text or rev.raw_text or "")
        for ar in aspect_results:
            try:
                aspect_enum_val = AspectEnum(ar.aspect)
                sentiment_enum_val = SentimentEnum(ar.sentiment)
            except ValueError:
                continue

            as_row = AspectSentiment(
                id=uuid.uuid4(),
                review_id=rev.id,
                cafe_id=rev.cafe_id,
                aspect=aspect_enum_val,
                sentiment=sentiment_enum_val,
                confidence=float(ar.confidence),
            )
            aspect_records.append(as_row)
            inserted_aspects_count += 1

        rev.is_processed = True

        if len(aspect_records) >= batch_size:
            db_session.add_all(aspect_records)
            await db_session.commit()
            aspect_records = []

    if aspect_records:
        db_session.add_all(aspect_records)
        await db_session.commit()

    await db_session.commit()
    logger.info(
        f"ABSA completed: processed {total_reviews} reviews, inserted {inserted_aspects_count} aspect records."
    )
    return {
        "processed_reviews": total_reviews,
        "inserted_aspects": inserted_aspects_count,
    }
