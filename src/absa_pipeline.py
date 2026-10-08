"""Early Cafind ML subset. Advanced search, ranking and database sync are pending."""
import logging
import re
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger('cafind.absa')
ASPECT_LEXICONS: Dict[str, List[str]] = {'ambience': ['suasana', 'tempat', 'estetik', 'aesthetic', 'esthetic', 'cozy', 'nyaman', 'tenang', 'hening', 'damai', 'bising', 'ramai', 'berisik', 'sejuk', 'adem', 'panas', 'pengap', 'outdoor', 'indoor', 'luar ruangan', 'dalam ruangan', 'taman', 'garden', 'rooftop', 'view', 'pemandangan', 'dekorasi', 'interior', 'spot foto', 'foto', 'ootd', 'instagramable', 'instagenic', 'vintage', 'industrial', 'konsep', 'bersih', 'kotor', 'rapi', 'nuansa', 'vibes', 'vibe'], 'facility': ['wifi', 'wi-fi', 'internet', 'koneksi', 'colokan', 'stopkontak', 'stop kontak', 'steker', 'listrik', 'charger', 'meja', 'meja kerja', 'kursi', 'tempat duduk', 'nugas', 'wfc', 'kerja', 'laptop', 'parkir', 'parkiran', 'tukang parkir', 'musholla', 'mushola', 'tempat ibadah', 'sholat', 'toilet', 'kamar mandi', 'wc', 'ac', 'pendingin', 'kipas', 'smoking area', 'area merokok'], 'price': ['harga', 'biaya', 'bayar', 'tarif', 'murah', 'terjangkau', 'ramah kantong', 'ramah dompet', 'kantong mahasiswa', 'hemat', 'murmer', 'bersahabat', 'merakyat', 'pas di kantong', 'mahal', 'pricey', 'kemahalan', 'kantong jebol', 'worth it', 'sebanding', 'padan', 'lumayan mahal', 'agak mahal', 'standar', 'kantong'], 'food_beverage': ['kopi', 'coffee', 'espresso', 'latte', 'americano', 'cappuccino', 'manual brew', 'v60', 'mocktail', 'teh', 'tea', 'minuman', 'makanan', 'menu', 'cemilan', 'snack', 'pastry', 'croissant', 'kue', 'roti', 'nasi', 'mie', 'makan', 'minum', 'rasa', 'taste', 'enak', 'lezat', 'nikmat', 'mantap', 'gurih', 'manis', 'pahit', 'asam', 'kecut', 'hambar', 'sedap', 'porsi']}
POSITIVE_TERMS: set[str] = {'enak', 'lezat', 'nikmat', 'mantap', 'juara', 'segar', 'pas', 'kencang', 'kenceng', 'cepat', 'banyak', 'melimpah', 'lengkap', 'luas', 'lega', 'nyaman', 'tenang', 'adem', 'sejuk', 'asri', 'estetik', 'bagus', 'cantik', 'kece', 'keren', 'cozy', 'murah', 'terjangkau', 'hemat', 'worth it', 'ramah', 'bersih', 'rapi', 'recommended', 'rekomendasi', 'suka', 'puas', 'top', 'oke', 'ok', 'senang', 'bersahabat', 'wangi', 'nagih', 'sempurna', 'love', 'favorit', 'fresh', 'lezatos'}
NEGATIVE_TERMS: set[str] = {'mahal', 'kemahalan', 'pricey', 'kurang', 'lemot', 'lelet', 'lambat', 'rusak', 'sedikit', 'minim', 'sempit', 'panas', 'pengap', 'bising', 'berisik', 'kotor', 'bau', 'pahit', 'hambar', 'asam', 'kecewa', 'buruk', 'jelek', 'parah', 'keasinan', 'kelamaan', 'jorok', 'gelap', 'bau', 'ribet', 'padat', 'engap', 'kemiskinan', 'boros', 'rugi'}
NEGATION_TERMS: set[str] = {'tidak', 'tak', 'nggak', 'ngga', 'gak', 'ga', 'gk', 'tdk', 'bukan', 'belum', 'blm', 'kurang'}
INTENSIFIERS: set[str] = {'sangat', 'banget', 'bgt', 'bngt', 'pol', 'sekali', 'amat', 'super', 'terlalu', 'parah'}

@dataclass
class AspectResult:
    aspect: str
    sentiment: str
    confidence: float
    evidence: str

class ABSAPipeline:

    def __init__(self):
        self.compiled_aspects = self._compile_aspects()

    def _compile_aspects(self) -> Dict[str, List[re.Pattern]]:
        compiled = {}
        for aspect, terms in ASPECT_LEXICONS.items():
            compiled[aspect] = [re.compile('\\b' + re.escape(kw) + '\\b', re.IGNORECASE) for kw in terms]
        return compiled

    def _split_into_clauses(self, text: str) -> List[str]:
        """Splits full review text into semantic clauses using punctuation and conjunctions."""
        delims = '[.,;!?\\n\\r]+|\\b(?:tetapi|tapi|namun|sedangkan|padahal|walaupun|meskipun)\\b'
        raw_clauses = re.split(delims, text, flags=re.IGNORECASE)
        cleaned = [c.strip() for c in raw_clauses if len(c.strip()) > 2]
        return cleaned if cleaned else [text.strip()]

    def analyze_clause_sentiment(self, clause: str, target_aspect: str) -> Tuple[str, float]:
        """
        Calculates sentiment polarity and confidence for a single clause
        targeting a specific aspect.
        """
        tokens = re.findall('\\b\\w+\\b', clause.lower())
        if not tokens:
            return ('neutral', 0.6)
        pos_score = 0.0
        neg_score = 0.0
        for i, word in enumerate(tokens):
            prev_window = tokens[max(0, i - 2):i]
            is_negated = any((neg in NEGATION_TERMS for neg in prev_window))
            next_window = tokens[i + 1:min(len(tokens), i + 3)]
            is_intensified = any((intens in INTENSIFIERS for intens in prev_window)) or any((intens in INTENSIFIERS for intens in next_window))
            weight = 1.4 if is_intensified else 1.0
            if word in POSITIVE_TERMS:
                if is_negated:
                    neg_score += 1.2 * weight
                else:
                    pos_score += 1.0 * weight
            elif word in NEGATIVE_TERMS:
                if is_negated:
                    pos_score += 1.1 * weight
                else:
                    neg_score += 1.2 * weight
        if target_aspect == 'facility':
            if 'banyak' in tokens and any((w in tokens for w in ['colokan', 'stopkontak', 'kursi', 'meja'])):
                pos_score += 1.5
            if 'kencang' in tokens or 'kenceng' in tokens or 'cepat' in tokens:
                pos_score += 1.5
            if 'susah' in tokens or 'jarang' in tokens or 'dikit' in tokens:
                neg_score += 1.5
        if target_aspect == 'price':
            if any((w in tokens for w in ['terjangkau', 'murah', 'ramah', 'hemat', 'pas'])):
                if not any((neg in NEGATION_TERMS for neg in tokens[:3])):
                    pos_score += 1.5
            if any((w in tokens for w in ['mahal', 'pricey', 'kemahalan'])):
                if not any((neg in NEGATION_TERMS for neg in tokens[:3])):
                    neg_score += 1.5
        diff = pos_score - neg_score
        total = pos_score + neg_score
        if total == 0:
            return ('neutral', 0.7)
        if diff > 0.3:
            conf = min(0.98, 0.75 + diff / (total + 1.0) * 0.23)
            return ('positive', round(conf, 4))
        elif diff < -0.3:
            conf = min(0.98, 0.75 + abs(diff) / (total + 1.0) * 0.23)
            return ('negative', round(conf, 4))
        else:
            return ('neutral', 0.72)

    def extract_aspects(self, text: str) -> List[AspectResult]:
        """
        Extracts aspects mentioned in the review text and determines their polarity.
        Returns a list of AspectResult (up to 4, one per detected aspect).
        """
        if not text or not isinstance(text, str):
            return []
        clauses = self._split_into_clauses(text)
        aspect_matches: Dict[str, List[Tuple[str, str, float]]] = {a: [] for a in ASPECT_LEXICONS}
        for clause in clauses:
            for aspect, patterns in self.compiled_aspects.items():
                if any((p.search(clause) for p in patterns)):
                    sentiment, conf = self.analyze_clause_sentiment(clause, aspect)
                    aspect_matches[aspect].append((clause, sentiment, conf))
        results: List[AspectResult] = []
        for aspect, matches in aspect_matches.items():
            if not matches:
                continue
            pos_confs = [c for cl, s, c in matches if s == 'positive']
            neg_confs = [c for cl, s, c in matches if s == 'negative']
            neu_confs = [c for cl, s, c in matches if s == 'neutral']
            if len(pos_confs) > len(neg_confs):
                final_sentiment = 'positive'
                final_conf = max(pos_confs)
            elif len(neg_confs) > len(pos_confs):
                final_sentiment = 'negative'
                final_conf = max(neg_confs)
            elif pos_confs and (not neg_confs):
                final_sentiment = 'positive'
                final_conf = max(pos_confs)
            elif neg_confs and (not pos_confs):
                final_sentiment = 'negative'
                final_conf = max(neg_confs)
            else:
                final_sentiment = 'neutral'
                final_conf = 0.7
            evidence_clause = matches[0][0]
            results.append(AspectResult(aspect=aspect, sentiment=final_sentiment, confidence=final_conf, evidence=evidence_clause))
        return results
