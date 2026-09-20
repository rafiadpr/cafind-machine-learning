"""
Cafind — Cafe Clustering Pipeline (K-Means with Multiple Distance Metrics)
===========================================================================
Clusters cafe listings into distinct profiles based on:
  1. Average Rating (1.0 - 5.0)
  2. Price Estimate / Normalized Expense Level
  3. Positive Sentiment Score (from ABSA aggregate confidence)
  4. Total Review Volume (Log-scaled)

Supports:
  - Euclidean distance (L2 norm) via standard K-Means.
  - Manhattan distance (L1 norm / Minkowski distance with p=1)
    using pairwise L1 distance matrix and K-Medoids / Medoid assignment.
  - Performance evaluation using Silhouette Score and cluster centroid profiling.

Usage:
    python scripts/clustering.py --metric euclidean --k 4
    python scripts/clustering.py --metric manhattan --k 4
    python scripts/clustering.py --compare
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cafind.clustering")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DEFAULT_INPUT_FILE = DATA_PROCESSED / "cafes_clean.csv"


# ─── Synthetic Data Generator (Fallback when CSV not yet scraped) ────────────
def generate_sample_dataset(n_samples: int = 120) -> pd.DataFrame:
    """Generates synthetic cafe data for standalone verification and unit tests."""
    np.random.seed(42)
    names = [f"Cafe {i+1}" for i in range(n_samples)]
    ratings = np.clip(np.random.normal(4.3, 0.4, n_samples), 3.0, 5.0)
    prices = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.3, 0.45, 0.2, 0.05])
    review_counts = np.random.exponential(scale=150, size=n_samples).astype(int) + 10
    sentiment_scores = np.clip(ratings / 5.0 + np.random.normal(0, 0.05, n_samples), 0.5, 1.0)

    return pd.DataFrame({
        "cafe_id": [f"c_{i+1:03d}" for i in range(n_samples)],
        "name": names,
        "rating": np.round(ratings, 2),
        "price_level": prices,
        "review_count": review_counts,
        "sentiment_score": np.round(sentiment_scores, 3),
    })


def load_features(input_path: Optional[Path] = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Loads cafe records and prepares standardized numerical feature matrix."""
    file_path = input_path or DEFAULT_INPUT_FILE
    if file_path.exists():
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
    else:
        logger.warning(f"File '{file_path}' not found. Using representative sample dataset.")
        df = generate_sample_dataset()

    feature_cols = []
    # Identify available numeric features
    for col in ["rating", "price_level", "sentiment_score", "review_count"]:
        if col in df.columns:
            feature_cols.append(col)

    if not feature_cols:
        df = generate_sample_dataset()
        feature_cols = ["rating", "price_level", "sentiment_score", "review_count"]

    df_clean = df.dropna(subset=feature_cols).copy()

    # Preprocessing: log-transform review_count to prevent high-popularity bias
    if "review_count" in df_clean.columns:
        df_clean["log_review_count"] = np.log1p(df_clean["review_count"])
        feature_cols_scaled = [c if c != "review_count" else "log_review_count" for c in feature_cols]
    else:
        feature_cols_scaled = feature_cols

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_cols_scaled])

    logger.info(f"Feature matrix shape: {X_scaled.shape} using features: {feature_cols_scaled}")
