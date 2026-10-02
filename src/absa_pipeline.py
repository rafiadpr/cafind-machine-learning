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
