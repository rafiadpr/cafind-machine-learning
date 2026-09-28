
"""Local JSON input for the early Cafind ML workflow. Database sync is pending."""
import argparse
import json
from pathlib import Path

def load_cafes(path):
    cafes = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(cafes, list) or not cafes:
        raise ValueError('Input must contain a non-empty list of cafes')
    for cafe in cafes:
        if not isinstance(cafe, dict) or not cafe.get('cafe_id') or not cafe.get('name'):
            raise ValueError('Every cafe needs cafe_id and name')
        if not isinstance(cafe.get('reviews', []), list) or any(not isinstance(review, str) for review in cafe.get('reviews', [])):
            raise ValueError('Cafe reviews must be a list of text strings')
    return cafes

def main():
    parser = argparse.ArgumentParser(description='Cafind local review preview')
    parser.add_argument('--input', required=True, help='JSON list of cafes and review strings')

    args = parser.parse_args()
    cafes = load_cafes(args.input)
    print(json.dumps({'cafes': len(cafes), 'reviews': sum(len(c.get('reviews', [])) for c in cafes)}))

if __name__ == '__main__':
    main()
