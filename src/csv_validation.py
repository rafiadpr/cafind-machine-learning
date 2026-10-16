import csv
import math

FIELDS = ('latitude', 'longitude', 'rating', 'review_count')

def validate_cafe_csv(path):
    count = 0
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            try:
                lat, lon, rating, reviews = (float(row[field]) for field in FIELDS)
                valid = all(math.isfinite(v) for v in (lat, lon, rating, reviews))
                valid = valid and -90 <= lat <= 90 and -180 <= lon <= 180
                valid = valid and 0 <= rating <= 5 and reviews >= 0 and reviews.is_integer()
            except (ValueError, TypeError):
                valid = False
            if not valid:
                raise ValueError(f'Invalid numeric cafe values on CSV line {line}')
            count += 1
    if not count:
        raise ValueError('CSV contains no cafe rows')
    return count
