# Cafind Machine Learning

Data collection and clustering experiments for the Cafind cafe application.

## Setup

Use Python 3.11 or newer and a virtual environment.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Project structure

- `scripts/`: data collection and command-line tools.
- `src/`: initial aspect sentiment and embedding modules.
- `experiments/`: nine notebooks covering three algorithms and distance metrics.
- `data/`: local input and generated data; large datasets are not tracked.

## Collect a small sample

The scraper accepts one query and exports visible results to local CSV files.
This is an initial workflow; resume and backend synchronization are pending.

```bash
python scripts/scraper.py --query "cafe Kota Malang" --max-cafes 5 --headless
```

Preview CSV files replace the previous preview run. Save a separate copy before
collecting another sample if the previous result is still needed.

## Clustering notebooks

Open the notebooks in `experiments/` with Jupyter.
Provide `cafes.csv` in the kernel working directory, or set `DATA_PATH` before
running the first code cell. Numeric input fields are `latitude`, `longitude`,
`rating`, and `review_count`. Set `FIGURES_DIR` to choose an output directory.

## Local aspect and search preview

Input is a JSON list of cafes containing `cafe_id`, `name`, and review strings.

```bash
python scripts/run_absa_and_sbert.py --input cafes.json --query "tempat nyaman"
```

The first embedding run needs the model download unless it is already cached.
One aspect filter is supported. Multiple aspect filters, advanced ranking,
and database synchronization are still under development.

## Development

Keep model weights, credentials, browser profiles, and large datasets local.
Check results on the intended dataset before using experiment measurements.
