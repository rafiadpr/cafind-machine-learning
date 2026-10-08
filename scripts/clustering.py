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
    return df_clean, X_scaled, feature_cols_scaled


# ─── Distance Metric Implementations ──────────────────────────────────────────
def compute_minkowski_l1_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Computes Manhattan (Minkowski distance with p=1 / ord=1) between samples and centroids.
    Formula: D(x, c) = sum(|x_i - c_i|) for each dimension i.
    """
    # Pairwise Manhattan distance matrix: (n_samples, n_clusters)
    return pairwise_distances(X, centers, metric="minkowski", p=1)


class ManhattanKMeans:
    """
    K-Means clustering optimized for Manhattan (L1 norm / Minkowski p=1) metric.
    Uses median centroid updates for robust cluster representations with outliers.
    """
    def __init__(self, n_clusters: int = 4, max_iter: int = 300, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # Initialize centers with random sample points
        init_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[init_indices].copy()

        for iteration in range(self.max_iter):
            # Compute Minkowski L1 distances using ord=1
            dist_matrix = compute_minkowski_l1_distances(X, self.cluster_centers_)
            labels = np.argmin(dist_matrix, axis=1)

            # Update centroids using component-wise median (L1 median property)
            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                cluster_members = X[labels == k]
                if len(cluster_members) > 0:
                    new_centers[k] = np.median(cluster_members, axis=0)
                else:
                    new_centers[k] = X[np.random.choice(n_samples)]

            # Check convergence
            if np.allclose(self.cluster_centers_, new_centers, atol=1e-4):
                logger.debug(f"Manhattan K-Means converged in {iteration + 1} iterations.")
                break

            self.cluster_centers_ = new_centers

        self.labels_ = labels
        return labels


# ─── Training & Evaluation Routine ────────────────────────────────────────────
def train_and_evaluate(
    X: np.ndarray,
    df: pd.DataFrame,
    features: List[str],
    k: int = 4,
    metric: str = "euclidean"
) -> Tuple[np.ndarray, float, pd.DataFrame]:
    """Fits K-Means with specified distance metric and outputs performance statistics."""
    metric = metric.lower()

    if metric == "manhattan":
        model = ManhattanKMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        # Silhouette Score calculated with Manhattan metric
        sil_score = silhouette_score(X, labels, metric="manhattan")
    elif metric == "euclidean":
        model = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        labels = model.fit_predict(X)
        sil_score = silhouette_score(X, labels, metric="euclidean")
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Choose 'euclidean' or 'manhattan'.")

    # Generate Cluster Profiles
    df_result = df.copy()
    df_result["cluster"] = labels

    summary = df_result.groupby("cluster")[features].mean()
    summary["cafe_count"] = df_result.groupby("cluster").size()

    return labels, sil_score, summary


def compare_metrics(X: np.ndarray, df: pd.DataFrame, features: List[str], k_range: range = range(2, 6)):
    """Compares clustering effectiveness across Euclidean and Manhattan distance metrics."""
    logger.info("=" * 60)
    logger.info(f"{'K':<4} | {'Metric':<12} | {'Silhouette Score':<18} | {'Status':<10}")
    logger.info("=" * 60)

    for k in k_range:
        # Euclidean
        _, sil_euclid, _ = train_and_evaluate(X, df, features, k=k, metric="euclidean")
        logger.info(f"{k:<4} | {'Euclidean':<12} | {sil_euclid:<18.4f} | Optimal" if sil_euclid > 0.4 else f"{k:<4} | {'Euclidean':<12} | {sil_euclid:<18.4f} | Pass")

        # Manhattan (Minkowski L1)
        _, sil_manh, _ = train_and_evaluate(X, df, features, k=k, metric="manhattan")
        logger.info(f"{k:<4} | {'Manhattan L1':<12} | {sil_manh:<18.4f} | Robust (Outliers)" if sil_manh > 0.35 else f"{k:<4} | {'Manhattan L1':<12} | {sil_manh:<18.4f} | Pass")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Cafind Cafe Clustering Module")
    parser.add_argument("--metric", choices=["euclidean", "manhattan"], default="euclidean", help="Distance metric to use")
    parser.add_argument("--k", type=int, default=4, help="Number of clusters (default: 4)")
    parser.add_argument("--compare", action="store_true", help="Compare Euclidean vs Manhattan metrics across k=2..5")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV file")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    df, X_scaled, features = load_features(input_path)

    if args.compare:
        compare_metrics(X_scaled, df, features)
        return

    logger.info(f"Training K-Means with metric='{args.metric}', K={args.k}...")
    labels, sil, summary = train_and_evaluate(X_scaled, df, features, k=args.k, metric=args.metric)

    logger.info(f"\nModel Silhouette Score ({args.metric}): {sil:.4f}")
    logger.info("\nCluster Profile Summary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
