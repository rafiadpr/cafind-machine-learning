"""Read-only, nested-sample ABSA/SBERT benchmarks; no production-index writes."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import statistics
import time

import numpy as np

ASPECTS = ('ambience', 'facility', 'price', 'food_beverage')


def validate_sizes(sizes):
    values = sorted(set(int(value) for value in sizes))
    if not values or len(values) > 8 or any(n < 10 or n > 10000 for n in values):
        raise ValueError('Pilih 1 sampai 8 ukuran sampel, masing-masing 10 sampai 10000 ulasan.')
    return values


def clean_snapshot(reviews, cafes):
    rows = []
    seen = set()
    for review in reviews:
        rid = str(review['review_id'])
        cid = str(review['cafe_id'])
        text = (review.get('text') or review.get('raw_text') or '').strip()
        if not text or cid not in cafes or rid in seen:
            continue
        seen.add(rid)
        rows.append(dict(review, review_id=rid, cafe_id=cid, text=text))
    return sorted(rows, key=lambda row: row['review_id'])


def nested_samples(rows, sizes, seed):
    sizes = validate_sizes(sizes)
    if max(sizes) > len(rows):
        raise ValueError(f'Sampel terbesar {max(sizes)} melebihi {len(rows)} ulasan valid pada snapshot.')
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return [(size, shuffled[:size]) for size in sizes]


def compare_review_sizes(reviews, cafes, sizes, query, output_dir, encoder,
                         pipeline, document_builder, seed=42, repeats=3,
                         batch_size=32, metadata=None):
    sizes = validate_sizes(sizes)
    if repeats < 1 or repeats > 5 or not 1 <= batch_size <= 128:
        raise ValueError('Pengulangan harus 1 sampai 5; batch embedding 1 sampai 128.')
    if not query.strip() or len(query) > 512:
        raise ValueError('Query wajib berisi 1 sampai 512 karakter.')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'result.json').exists():
        raise ValueError('Direktori eksperimen sudah berisi hasil; gunakan run baru.')
    cafes = {str(key): value for key, value in cafes.items()}
    rows = clean_snapshot(reviews, cafes)
    nested_samples(rows, sizes, seed)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row['review_id'], row['cafe_id'], row['text'],
            row.get('rating'), row.get('likes')], ensure_ascii=False).encode('utf-8'))
    trials = []
    for repeat in range(repeats):
        print(f'Pengulangan {repeat+1}/{repeats}', flush=True)
        per_repeat = []
        for size, sample in nested_samples(rows, sizes, seed + repeat):
            print(f'  Sampel {size} ulasan', flush=True)
            started = time.perf_counter()
            counts, by_cafe, profiles = Counter(), defaultdict(list), defaultdict(lambda: defaultdict(Counter))
            covered = 0
            for row in sample:
                aspects = pipeline.extract_aspects(row['text'])
                covered += bool(aspects)
                by_cafe[row['cafe_id']].append(row)
                for aspect in aspects:
                    counts[aspect.sentiment] += 1
                    profiles[row['cafe_id']][aspect.aspect][aspect.sentiment] += 1
            absa_seconds = time.perf_counter() - started
            document_started = time.perf_counter()
            documents, records = [], []
            quoted_reviews = 0
            for cid in sorted(by_cafe):
                votes = profiles[cid]
                profile = {}
                for aspect, labels in votes.items():
                    best = max(labels.values())
                    winners = sorted(label for label, count in labels.items() if count == best)
                    profile[aspect] = winners[0] if len(winners) == 1 else 'neutral'
                ordered = sorted(by_cafe[cid], key=lambda r: (
                    -float(r.get('rating') or 0), -int(r.get('likes') or 0), r['review_id']))
                snippets = [{'text': r['text']} for r in ordered if len(r['text']) > 15][:8]
                quoted_reviews += len(snippets)
                cafe = cafes[cid]
                documents.append(document_builder(cafe, snippets, profile))
                records.append({'cafe_id': cid, 'name': cafe.get('name', cid), 'absa_profile': profile})
            document_seconds = time.perf_counter() - document_started
            embedding_started = time.perf_counter()
            vectors = np.asarray(encoder.encode(documents, batch_size=batch_size,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != len(records) or not np.isfinite(vectors).all():
                raise ValueError('Embedding tidak valid atau jumlah dokumen tidak cocok.')
            embedding_seconds = time.perf_counter() - embedding_started
            query_started = time.perf_counter()
            query_vector = np.asarray(encoder.encode([query.strip()], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)[0]
            scores = vectors @ query_vector
            top = np.argsort(-scores, kind='stable')[:5]
            ranking = [dict(records[int(i)], similarity=round(float(scores[int(i)]), 6)) for i in top]
            query_seconds = time.perf_counter() - query_started
            trial_dir = root / f'trial-{repeat+1:02d}' / f'n-{size}'
            trial_dir.mkdir(parents=True, exist_ok=True)
            np.save(trial_dir / 'cafe_embeddings.npy', vectors)
            (trial_dir / 'cafe_id_map.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
            result = {'reviews': size, 'repeat': repeat+1, 'seed': seed+repeat,
                'sample_sha256': hashlib.sha256('\n'.join(r['review_id'] for r in sample).encode()).hexdigest(),
                'cafes': len(records), 'quoted_reviews': quoted_reviews,
                'absa_seconds': absa_seconds, 'document_seconds': document_seconds,
                'sbert_seconds': embedding_seconds, 'query_seconds': query_seconds,
                'total_seconds': time.perf_counter()-started,
                'aspect_coverage': covered/size, 'aspect_results': sum(counts.values()),
                'sentiments': dict(counts), 'ranking': ranking,
                'index_bytes': (trial_dir/'cafe_embeddings.npy').stat().st_size}
            per_repeat.append(result)
        reference = {r['cafe_id'] for r in per_repeat[-1]['ranking']}
        for result in per_repeat:
            current = {r['cafe_id'] for r in result['ranking']}
            result['top5_jaccard'] = len(current & reference)/len(current | reference) if current | reference else None
            trials.append(result)
    numeric = ('cafes','quoted_reviews','absa_seconds','document_seconds','sbert_seconds',
               'query_seconds','total_seconds','aspect_coverage','aspect_results','index_bytes','top5_jaccard')
    summaries = []
    for size in sizes:
        selected = [r for r in trials if r['reviews'] == size]
        summary = {'reviews': size, 'repeats': repeats}
        for key in numeric:
            values = [r[key] for r in selected if r[key] is not None]
            summary[key+'_mean'] = statistics.mean(values) if values else None
            summary[key+'_std'] = statistics.stdev(values) if len(values)>1 else 0.0
        summary['sentiments_mean'] = {label: statistics.mean(r['sentiments'].get(label,0) for r in selected)
                                     for label in ('positive','negative','neutral')}
        summaries.append(summary)
    report = {'schema_version': 1, 'status': 'completed',
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'query': query.strip(), 'sizes': sizes, 'seed': seed, 'repeats': repeats,
        'available_reviews': len(rows), 'snapshot_sha256': digest.hexdigest(),
        'metadata': metadata or {}, 'summary': summaries, 'trials': trials,
        'method': {'sampling':'Nested prefixes of the same seeded shuffle within each repeat.',
                   'timing':'Warm pretrained encoder; model loading/warmup excluded from trial timings.',
                   'documents':'One document per sampled cafe; at most eight review snippets longer than 15 characters. ABSA profile uses all sampled reviews.',
                   'stability':'Top-5 Jaccard against the largest sample in the same repeat; not relevance accuracy.',
                   'quality':'No manual aspect/relevance labels supplied; accuracy/F1 are not computed.',
                   'persistence':'Read-only database snapshot; experimental indexes are isolated per run.'}}
    (root/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    fields = [key for key in summaries[0] if key != 'sentiments_mean']
    with (root/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:v for k,v in row.items() if k in fields} for row in summaries)
    return report