"""Parsing helpers shared by cafe metadata and review extraction."""

import re
from typing import Optional

# Malang bounding box — filter out off-target results
Malang_BOUNDS = {
    "lat_min": -8.25,
    "lat_max": -7.75,
    "lng_min": 112.40,
    "lng_max": 112.90,
}

def parse_rating(text: str) -> Optional[float]:
    """Extract a float rating from strings like '4,3' or '4.3 stars'."""
    match = re.search(r"(\d)[,.](\d)", text)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        return float(match.group(1))
    return None


def parse_review_count(text: str) -> int:
    """Extract integer from '(1.234)' or '1,234 ulasan' or '1234 reviews'."""
    # Remove dots/commas used as thousand separators in Indonesian
    text = text.replace(".", "").replace(",", "")
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else 0


PRICE_PATTERN = re.compile(
    r"(?:Rp|IDR)\s*(\d+(?:[.,]\d+)*)(?:\s*(ribu|rb|k)\b)?"
    r"(?:\s*(?:[–—−-]|to|sampai|hingga)\s*(?:(?:Rp|IDR)\s*)?"
    r"(\d+(?:[.,]\d+)*)(?:\s*(ribu|rb|k)\b)?)?",
    re.I,
)


def parse_price_range(raw: Optional[str]) -> tuple[Optional[str], Optional[float], Optional[float]]:
    """Read literal rupiah amounts, including shared K/rb suffixes, without invented bands."""
    if not raw:
        return None, None, None
    clean = " ".join(re.sub(r"[\ue000-\uf8ff]", "", raw).split())
    if re.fullmatch(r"\${1,4}", clean):
        return clean, None, None  # A category alone does not specify rupiah amounts.
    match = PRICE_PATTERN.search(clean)
    if not match:
        return None, None, None
    low_raw, low_unit, high_raw, high_unit = match.groups()

    def amount(number: str, unit: Optional[str]) -> float:
        # A 3-digit group is a thousands separator; other decimal groups
        # with an explicit K/rb unit are fractional thousands (25,5 rb).
        if unit and re.fullmatch(r"\d+[.,]\d{1,2}", number):
            value = float(number.replace(",", "."))
        else:
            value = float(number.replace(".", "").replace(",", ""))
        return value * 1000 if unit else value

    # Rp 25–50K shares K; Rp 25.000–50K already expresses its lower bound in full.
    low_shared = high_unit if not re.search(r"[.,]\d{3}(?:$|[.,])", low_raw) else None
    high_shared = low_unit if high_raw and not re.search(r"[.,]\d{3}(?:$|[.,])", high_raw) else None
    low = amount(low_raw, low_unit or low_shared)
    high = amount(high_raw, high_unit or high_shared) if high_raw else low
    return match.group(0).strip(), min(low, high), max(low, high)


def parse_coordinates(url: str) -> tuple[Optional[float], Optional[float]]:
    """Extract lat/lng from a Google Maps URL."""
    if not url:
        return None, None

    # Format 1: .../@-7.7956,110.3695,17z/...
    match = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    # Format 2: .../data=!3d-7.9734363!4d112.6326229
    match_3d = re.search(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)", url)
    if match_3d:
        return float(match_3d.group(1)), float(match_3d.group(2))
        
    # Format 3: ...center=-7.9734%2C112.6326 or center=-7.9734,112.6326
    match_center = re.search(r"center=(-?\d+\.\d+)(?:%2C|,)(-?\d+\.\d+)", url)
    if match_center:
        return float(match_center.group(1)), float(match_center.group(2))

    return None, None


def is_in_Malang(lat: Optional[float], lng: Optional[float]) -> bool:
    """Filter out cafes that are clearly outside Malang."""
    if lat is None or lng is None or lat == 0.0 or lng == 0.0:
        return True  # Give benefit of the doubt if we can't parse coords
    b = Malang_BOUNDS
    return b["lat_min"] <= lat <= b["lat_max"] and b["lng_min"] <= lng <= b["lng_max"]


