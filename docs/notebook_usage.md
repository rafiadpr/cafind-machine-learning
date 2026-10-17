# Clustering experiment report

The nine notebooks compare DBSCAN, HDBSCAN, and K-Means variants using
Euclidean, Manhattan, and cosine distances. Use the same input CSV for each run.

## Input and execution

Install `requirements.txt`. Supply a CSV containing latitude, longitude,
rating, and review_count. Keep the dataset outside Git when it contains collected data.

```bash
python scripts/check_cafe_csv.py /path/to/cafes.csv
python scripts/summarize_notebooks.py --data /path/to/cafes.csv --output-dir reports
```

The command executes the notebooks and writes notebook_summary.csv plus figures.
For interactive use, set DATA_PATH and FIGURES_DIR in the notebook kernel first.

## Reading the results

Report cluster count, noise percentage, silhouette score, elapsed time, and
Python allocation peak. Tracemalloc does not measure total process RAM.
Record the dataset version and row count alongside the exported summary.
A silhouette score may be unavailable when only one usable cluster is found.
Do not substitute estimated numbers for a failed or incomplete experiment.
