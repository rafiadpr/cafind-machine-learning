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
