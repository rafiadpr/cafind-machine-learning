"""Restore failed local CSV writes; never contacts the backend."""

import argparse
from pathlib import Path

if __package__:
    from .scraper_storage import replay_csv_recovery
else:
    from scraper_storage import replay_csv_recovery


def main():
    raw = Path(__file__).resolve().parent.parent / "data" / "raw"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=raw / "csv_recovery")
    parser.add_argument("--output-dir", type=Path, default=raw)
    args = parser.parse_args()
    recovered, added = replay_csv_recovery(args.directory, args.output_dir)
    print(f"Recovered batches: {recovered}; CSV reviews added: {added}")


if __name__ == "__main__":
    main()
