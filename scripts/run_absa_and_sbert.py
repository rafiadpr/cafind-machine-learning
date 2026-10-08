
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

def run_preview(cafes, query, index_dir, aspect=None, sentiment='positive', limit=5):
    from collections import Counter
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.absa_pipeline import ABSAPipeline
    from src.sbert_embedder import SemanticVibeEngine
    pipeline = ABSAPipeline()
    documents = []
    for cafe in cafes:
        votes = {}
        for review in cafe.get('reviews', []):
            for result in pipeline.extract_aspects(review):
                votes.setdefault(result.aspect, Counter())[result.sentiment] += 1
        profile = {aspect: counts.most_common(1)[0][0] for aspect, counts in votes.items()}
        reviews = [{'text': text} for text in cafe.get('reviews', [])]
        documents.append(dict(cafe, absa_profile=profile,
                              document_text=SemanticVibeEngine.construct_cafe_document(cafe, reviews)))
    engine = SemanticVibeEngine(Path(index_dir) / 'cafe_embeddings.npy', Path(index_dir) / 'cafe_id_map.json')
    engine.build_and_save_index(documents)
    filters = {aspect: sentiment} if aspect else None
    return engine.search_vibe(query, aspect_filters=filters, limit=limit)

def main():
    parser = argparse.ArgumentParser(description='Cafind local review preview')
    parser.add_argument('--input', required=True, help='JSON list of cafes and review strings')

    parser.add_argument('--query', required=True)
    parser.add_argument('--index-dir', default='data/processed')
    parser.add_argument('--aspect', choices=['ambience', 'facility', 'price', 'food_beverage'])
    parser.add_argument('--sentiment', choices=['positive', 'negative', 'neutral'], default='positive')
    parser.add_argument('--limit', type=int, default=5)

    args = parser.parse_args()
    cafes = load_cafes(args.input)
    print(json.dumps(run_preview(cafes, args.query, args.index_dir, args.aspect, args.sentiment, args.limit), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
