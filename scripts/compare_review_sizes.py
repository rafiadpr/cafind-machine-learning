"""Benchmark ABSA/SBERT on review counts without updating production data."""
import argparse
import asyncio
import os
from pathlib import Path
import platform
import sys
import time

ML_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(os.environ.get('CAFIND_BACKEND_DIR', str(ML_DIR.parent/'backend'))).resolve()
sys.path.insert(0,str(ML_DIR))
sys.path.insert(0,str(BACKEND_DIR))


async def load_snapshot():
    from sqlalchemy import select, func
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from app.db.models import Cafe, Review
    engine=create_async_engine(os.environ['DATABASE_URL'])
    try:
        async with async_sessionmaker(engine)() as session:
            text=func.coalesce(func.nullif(Review.text,''),Review.raw_text,'')
            rows=(await session.execute(select(Review.id,Review.cafe_id,text.label('text'),
                Review.rating,Review.likes).where(func.length(func.trim(text))>0)
                .order_by(Review.id).limit(20000))).mappings().all()
            cafes=(await session.execute(select(Cafe.id,Cafe.name,Cafe.address,Cafe.categories))).mappings().all()
        return [dict(r,review_id=str(r['id']),cafe_id=str(r['cafe_id'])) for r in rows], {
            str(c['id']):dict(c) for c in cafes}
    finally:
        await engine.dispose()


def main():
    parser=argparse.ArgumentParser(description='Perbandingan jumlah ulasan, bukan training ulang model.')
    parser.add_argument('--sizes',nargs='+',type=int,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--query',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    allowed=(BACKEND_DIR/'.runtime/ml-comparisons/runs').resolve()
    if not args.output_dir.resolve().is_relative_to(allowed):
        parser.error('Output harus berada di .runtime/ml-comparisons/runs backend.')
    from src.review_size_comparison import validate_sizes,clean_snapshot,compare_review_sizes
    validate_sizes(args.sizes)
    print('--- STEP 1: Membaca snapshot database tanpa perubahan ---',flush=True)
    reviews,cafes=asyncio.run(load_snapshot())
    eligible=clean_snapshot(reviews,cafes)
    if max(args.sizes)>len(eligible):
        raise ValueError(f'Sampel terbesar melebihi {len(eligible)} ulasan valid.')
    print(f'Snapshot valid: {len(eligible)} ulasan.',flush=True)
    print('--- STEP 2: Memuat model dari cache lokal ---',flush=True)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    from sentence_transformers import SentenceTransformer
    from src.absa_pipeline import ABSAPipeline
    from src.sbert_embedder import SemanticVibeEngine,DEFAULT_MODEL_NAME
    started=time.perf_counter()
    try:
        encoder=SentenceTransformer(DEFAULT_MODEL_NAME,device='cpu',local_files_only=True)
    except Exception as error:
        raise RuntimeError('Model SBERT belum tersedia di cache interpreter ML; siapkan model sebelum benchmark.') from error
    load_seconds=time.perf_counter()-started
    encoder.encode(['uji pemanasan model'],normalize_embeddings=True,show_progress_bar=False)
    print('--- STEP 3: Menjalankan sampel dan menyimpan indeks eksperimen ---',flush=True)
    result=compare_review_sizes(eligible,cafes,args.sizes,args.query,args.output_dir,
        encoder,ABSAPipeline(),SemanticVibeEngine.construct_cafe_document,
        seed=args.seed,repeats=args.repeats,batch_size=args.batch_size,
        metadata={'source':'PostgreSQL read-only snapshot, capped at 20000 ordered review IDs',
                  'model':DEFAULT_MODEL_NAME,'model_load_seconds':load_seconds,
                  'device':'cpu','python':platform.python_version(),'torch':torch.__version__,
                  'absa':'lexicon/rule-based; not IndoBERT fine-tuning'})
    print('--- STEP 4: Laporan perbandingan selesai ---',flush=True)
    print('Ukuran sampel:',result['sizes'],flush=True)


if __name__=='__main__':
    main()
