"""Gabungkan summary 9 notebook. Jalankan: python summarize_notebooks.py.

Letakkan cafes.csv di data/ atau di sebelah skrip, bersama sembilan notebook, atau gunakan
--data untuk CSV di lokasi lain. Tidak memerlukan struktur folder repo tertentu.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Jalankan tanpa membuka jendela grafik.
import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / 'experiments'
NOTEBOOK_NAMES = [
    f"{algorithm}_{metric}.ipynb"
    for algorithm in ("kmeans", "dbscan", "hdbscan")
    for metric in ("euclidean", "manhattan", "cosine")
]


def collect_summaries(data_path, output_dir):
    rows = []
    previous_cwd = Path.cwd()
    try:
        os.chdir(EXPERIMENTS_DIR)
        for algorithm in ("kmeans", "dbscan", "hdbscan"):
            for metric in ("euclidean", "manhattan", "cosine"):
                path = EXPERIMENTS_DIR / f"{algorithm}_{metric}.ipynb"
                print(f"Menjalankan {path.name} ...", flush=True)
                notebook = json.loads(path.read_text(encoding="utf-8"))
                namespace = {
                    "__name__": "__main__",
                    "DATA_PATH": data_path,
                    "FIGURES_DIR": output_dir / "figures",
                }

                try:
                    for index, cell in enumerate(notebook["cells"], start=1):
                        if cell["cell_type"] != "code":
                            continue
                        source = "".join(cell["source"])
                        with contextlib.redirect_stdout(io.StringIO()):
                            exec(compile(source, f"{path.name}:sel-{index}", "exec"), namespace)
                    summary = namespace[f"{algorithm.upper()}_SUMMARY"][metric]
                    rows.append({
                        "Notebook": path.name,
                        "Algoritma": summary["algo"],
                        "Metrik": summary["metric"],
                        "Parameter": summary["param"],
                        "Waktu (detik)": summary["time"],
                        "Memori puncak (MB)": summary["mem_mb"],
                        "Jumlah cluster": summary["clusters"],
                        "Noise (%)": summary["noise_pct"],
                        "Silhouette": summary["silhouette"],
                    })
                except Exception as exc:
                    raise RuntimeError(f"Gagal mengambil summary {path.name}: {exc}") from exc
                finally:
                    plt.close("all")
    finally:
        os.chdir(previous_cwd)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Tabel summary dari 9 notebook di folder skrip.")
    default_data = next(
        (path for path in (EXPERIMENTS_DIR / "data" / "cafes.csv", EXPERIMENTS_DIR / "cafes.csv")
         if path.is_file()),
        EXPERIMENTS_DIR / "data" / "cafes.csv",
    )
    parser.add_argument("--data", type=Path, default=default_data,
                        help="Lokasi CSV; default: data/cafes.csv atau cafes.csv di sebelah skrip.")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS_DIR / "reports",
                        help="Folder hasil; default: reports di sebelah skrip.")
    args = parser.parse_args()
    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_path.is_file():
        parser.error(f"CSV tidak ditemukan: {data_path}. Letakkan cafes.csv di data/ atau di sebelah skrip, atau gunakan --data.")
    missing = [name for name in NOTEBOOK_NAMES if not (EXPERIMENTS_DIR / name).is_file()]
    if missing:
        parser.error("Notebook harus berada di sebelah skrip. Belum ditemukan: " + ", ".join(missing))

    table = collect_summaries(data_path, output_dir)
    output_path = output_dir / "notebook_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8-sig", na_rep="N/A")
    print("\nTABEL SUMMARY 9 NOTEBOOK\n")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}", na_rep="N/A"))
    print(f"\nCSV disimpan di: {output_path}")


if __name__ == "__main__":
    main()
