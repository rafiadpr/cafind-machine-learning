# Pembelajaran mesin Cafind

Modul ini mengumpulkan data, melatih model, dan menjalankan inferensi untuk rekomendasi kafe di Cafind.

## Struktur folder

```text
machine_learning/
├── data/
│   ├── raw/            ← CSV dari Google Maps atau Kaggle (cafes.csv, reviews.csv)
│   ├── processed/      ← Data siap latih setelah pembersihan dan tokenisasi
│   └── labeled/        ← Data ABSA berlabel manual (aspek, sentimen)
├── notebooks/          ← Notebook Jupyter untuk eksplorasi data dan eksperimen
├── scripts/
│   ├── scraper.py          ← Pengambilan data Google Maps dengan Playwright
│   ├── dataset_loader.py   ← Konversi data Kaggle ke format CSV Cafind
│   ├── preprocess.py       ← Pembersihan dan tokenisasi teks
│   ├── clustering.py       ← Klasterisasi K-Means (Euclidean dan Manhattan L1)
│   ├── vibe_tagger.py      ← Ekstraksi kata kunci suasana
│   ├── train_absa.py       ← Pelatihan lanjutan IndoBERT untuk ABSA
│   ├── build_embeddings.py ← Pembuatan embedding SBERT dan indeks FAISS
│   └── evaluate.py         ← Metrik evaluasi model
├── models/
│   ├── indobert_absa/  ← Bobot IndoBERT hasil pelatihan lanjutan
│   └── sbert_index/    ← Indeks vektor FAISS
├── inference/
│   ├── absa_inferencer.py    ← Inferensi ABSA untuk backend
│   └── semantic_search.py    ← Pencarian semantik SBERT untuk backend
└── requirements.txt
```

## Mulai

### 1. Pasang dependensi

Jalankan perintah berikut dari folder `machine_learning`.

```bash
pip install -r requirements.txt
playwright install chromium
```

### 2a. Ambil data dari Google Maps

*Scraper* membatasi hasil ke wilayah Malang berdasarkan koordinat. Skrip membuka **Selengkapnya** untuk mengambil teks utuh bagi ABSA.

```bash
# Bawaan (7 pencarian, maksimal 60 kafe per pencarian dan 20 ulasan per kafe)
python scripts/scraper.py

# Kata kunci dan kuota khusus
python scripts/scraper.py --query "cafe Kota Malang" --max-cafes 5 --reviews-per-cafe 20

# Lanjutkan pengambilan data
python scripts/scraper.py --resume

# Jalankan tanpa jendela peramban
python scripts/scraper.py --headless
```

Agar hasil terkirim ke API, jalankan *backend* pada `BACKEND_URL` di `.env`. Jika *backend* tidak tersedia, skrip tetap menyimpan hasil ke CSV. Simpan berkas `data/raw/` di luar Git. Lihat `.gitignore`.

#### Masuk ke Google Maps lewat Chromium

*Scraper* berhenti saat Maps meminta login atau menampilkan pesan **tampilan terbatas**. Jika akses ulasan terhalang, simpan sesi login dengan perintah berikut.

```powershell
python scripts/scraper.py --login-only
```

1. Klik **Login** di Google Maps pada jendela Chromium yang terbuka. Masuk dengan akun Google dan selesaikan verifikasi dua langkah jika diminta.
2. Buka **Ulasan** di salah satu kafe, lalu klik **Lihat ulasan lainnya** jika tersedia. Pastikan daftar ulasan bisa dibuka dan digulir.
3. Biarkan Chromium terbuka, kembali ke terminal, lalu tekan **Enter**. Skrip menutup Chromium untuk menyimpan profil.
4. Uji sesi dengan satu kafe, lalu jalankan lagi dengan kuota yang diinginkan.

```powershell
python scripts/scraper.py --query "Warkop Kota Malang" --max-cafes 1 --reviews-per-cafe 5 --resume
python scripts/scraper.py --query "Warkop Kota Malang" --max-cafes 40 --reviews-per-cafe 30 --resume
```

Chromium memakai profil `data/.browser_profile/` untuk login dan pengambilan data. Gunakan satu proses per profil. Git mengabaikan folder profil, yang berisi sesi akun. Jaga agar folder itu tetap lokal.

Opsi `--login-only` hanya menyiapkan sesi. Jalankan dengan jendela peramban terbuka, tanpa `--headless`. Ulangi login jika sesi kedaluwarsa. Jika Google menolak login atau ulasan tetap terhalang, coba lewat Chrome.

#### Masuk lewat Chrome jika Google memblokir Chromium

Google dapat menolak login dari peramban otomatis meski kredensial dimasukkan manual. [Panduan browser Google](https://support.google.com/accounts/answer/7675428?hl=en)

```powershell
python scripts/scraper.py --login-only --browser chrome
```

Perintah ini membuka Chrome yang sudah terpasang tanpa Playwright atau *remote debugging*. Chrome memakai profil `data/.browser_profile_chrome/`, yang juga diabaikan Git. Masuk lewat **Login** di Maps dan pastikan ulasan bisa dibuka. Tutup semua jendela Chrome yang memakai profil ini sebelum menekan **Enter** di terminal.

Gunakan `--browser chrome` saat mengambil data agar skrip memakai profil yang sama.

```powershell
python scripts/scraper.py --browser chrome --query "Warkop Kota Malang" --max-cafes 1 --reviews-per-cafe 5 --resume
```

Jika skrip gagal menemukan Chrome, atur lokasi aplikasinya di `.env`.

```dotenv
CHROME_PATH=C:/Program Files/Google/Chrome/Application/chrome.exe
```

Google masih bisa membatasi akses saat Playwright memakai sesi tersimpan. Jika login lewat Chrome juga ditolak, ikuti panduan Google di atas dan periksa akses ulasan sebelum menjalankan *scraper*.

#### Melanjutkan pengambilan data

`--resume` mencocokkan nama dan alamat kafe, lalu memperbarui detail dengan `cafe_id` lama. Skrip mencoba lagi kafe yang belum punya ulasan teks. Jika ulasan sudah tersimpan, skrip mempertahankan jumlahnya dan hanya mengirim detail terbaru ke *backend* (`reviews: []`). Ringkasan mencatat `Details-only updates`. Kafe dengan 5 ulasan tetap punya 5 ulasan meski kuota naik ke 30. Sinkronisasi ulang CSV lama memerlukan proses terpisah.

#### Harga dan jam buka

Skrip membaca harga dari label atau ringkasan di tab **Overview**. Format seperti `Rp 25–50K`, `Rp 25–50 rb`, dan nominal penuh disimpan sebagai rupiah di `price_min` dan `price_max`. Kategori seperti `$$` disimpan apa adanya, dengan kolom nominal kosong.

Skrip membuka tabel jam untuk membaca jadwal harian, termasuk hari tutup dan beberapa rentang jam. Jika jadwal mingguan tidak tersedia, skrip memakai status buka saat itu dan mencatat peringatan. Lihat baris `Details:` di log. Saat memperbarui CSV berdasarkan ID, nilai kosong dari Maps membiarkan detail lama tetap tersimpan.

#### Kegagalan pengambilan data

`Timeout scraping cafe details` muncul sebelum pengambilan ulasan. Skrip mencoba navigasi hingga dua kali dan menunggu dokumen mulai dimuat (`commit`) serta judul kafe tampil. Jika keduanya siap, skrip bisa lanjut sebelum `DOMContentLoaded`.

Skrip memeriksa panel lewat kartu dan kontrol ulasan. Filter **All** (`aria-label="All reviews"`) bisa tetap terlihat saat panel terbuka. Skrip membiarkan dialog ulasan terbuka dan menghitung kuota berdasarkan ID unik karena satu kartu bisa memuat beberapa elemen `data-review-id` serta teks terjemahan.

Jika panel gagal dibuka atau dibaca, skrip mencatat `Review failures` dan menahan pengiriman hasil kafe ke *backend*. Rincian galat dan tangkapan layar tersimpan di `debug/scraper/`, yang diabaikan Git. Ringkasan memisahkan kafe yang dilewati dari kegagalan detail atau pencarian. Jika terjadi kegagalan dan tidak ada kafe yang berhasil diproses, skrip keluar dengan kode 1.

#### Bahasa ulasan dan koreksi data

*Scraper* memakai setelan bahasa Indonesia (`id-ID`) dan mengganti parameter bahasa pada URL pencarian atau tempat menjadi `hl=id`. Skrip membaca teks yang tampil dan mencoba **Lihat aslinya / Show original** jika tersedia. Pertahankan ulasan asli penulis meski berbahasa Inggris.

Ambil ulang ulasan ke folder pemeriksaan dengan `--output-dir` dan `--no-backend`. Gunakan folder baru tanpa `--resume`.

```powershell
python scripts/scraper.py --browser chrome --query "nama kafe dan alamat lengkap" --max-cafes 1 --reviews-per-cafe 30 --output-dir data/review_repair/pemeriksaan --no-backend
```

Gunakan `--place-url "URL Google Maps lengkap"` sebagai pengganti kata kunci untuk membuka tempat tertentu. Pencarian juga bisa langsung membuka satu tempat. Cocokkan nama, alamat, dan koordinat dengan data kafe di basis data.

`scripts/prepare_review_language_repair.py` membuat perbandingan teks, cadangan CSV, dan dua berkas SQL secara lokal. Jika data yang akan diperbaiki tidak ada di CSV utama, gunakan `--database-snapshot` yang berisi kafe dan ulasannya. Skrip memeriksa identitas kafe sebelum membuat rencana.

```powershell
python scripts/prepare_review_language_repair.py --staged-dir data/review_repair/pemeriksaan --database-snapshot path/ke/database_snapshot.json --db-cafe-id UUID_KAFE_DB
```

1. Periksa `proposed_changes.csv`, lalu jalankan `01_preview.sql` di PostgreSQL. Pratinjau berakhir dengan `ROLLBACK`.
2. Jalankan `02_apply.sql` setelah seluruh perubahan disetujui. SQL memeriksa ID, penulis, nilai, dan teks lama sebelum membuat tabel cadangan. SQL mempertahankan ID ulasan, memperbarui teks, dan mengatur ulang hasil ABSA terkait.
3. Ikuti README di folder rencana untuk memperbarui CSV dan menjalankan ulang pemrosesan. Git mengabaikan berkas pemeriksaan dan cadangan di `data/review_repair/`.

SQL menyimpan teks sebagai byte UTF-8 agar editor Windows tidak mengubah LF, CRLF, atau emoji. Baca teksnya di `proposed_changes.csv`. Jika rencana lama dengan teks multibaris gagal saat mencocokkan teks, jalankan `ROLLBACK` dan buat ulang rencana dengan generator terbaru. Pertahankan pemeriksaan kecocokan teks.

#### Modul dan uji regresi

| Berkas di `scripts/` | Tugas |
| --- | --- |
| `scraper.py` | Menjalankan *scraper* dan mengambil metadata kafe |
| `scraper_reviews.py` | Membuka panel dan mengambil ulasan |
| `scraper_browser.py` | Mengelola peramban dan sesi login |
| `scraper_parsers.py` | Mengurai data dari Google Maps |
| `scraper_storage.py` | Membaca status `--resume` dan memperbarui CSV |
| `scraper_details.py` | Membaca harga dan jam buka dari **Overview** |

Uji regresi memakai HTML lokal dan folder sementara tanpa mengakses Google Maps atau *backend*. Setelah memasang Chromium lewat Playwright, jalankan perintah berikut dari folder utama repositori.

```powershell
python -m unittest machine_learning.tests.test_scraper_regressions -v
```

### 2b. Muat data dari Kaggle

Simpan CSV Kaggle di `data/raw/` dan pilih perintah yang sesuai.

```bash
# Deteksi format otomatis (satu baris per ulasan)
python scripts/dataset_loader.py --input data/raw/your_dataset.csv

# Data kafe saja, tanpa ulasan
python scripts/dataset_loader.py --input data/raw/places.csv --mode places_only

# Tambahkan data Kaggle ke CSV yang sudah ada
python scripts/dataset_loader.py --input data/raw/kaggle.csv --append
```

## Format CSV

### `data/raw/cafes.csv`

| Kolom | Tipe | Keterangan |
| --- | --- | --- |
| `cafe_id` | UUID | Kunci utama |
| `name` | str | Nama kafe |
| `address` | str | Alamat lengkap |
| `latitude` | float | Lintang GPS |
| `longitude` | float | Bujur GPS |
| `rating` | float | Nilai kafe (1 sampai 5 bintang) |
| `review_count` | int | Jumlah ulasan di Maps |
| `categories` | str | Kategori yang dipisahkan koma |
| `phone` | str | Nomor telepon |
| `website` | str | URL situs web |
| `price_range` | str | Rentang harga di Maps |
| `price_min` | float | Harga terendah dalam rupiah, jika tersedia |
| `price_max` | float | Harga tertinggi dalam rupiah, jika tersedia |
| `opening_hours` | str | Jadwal mingguan atau status buka saat ini |
| `scraped_at` | ISO8601 | Waktu pengambilan data |
| `source_query` | str | Kata kunci yang menemukan kafe |

### `data/raw/reviews.csv`

| Kolom | Tipe | Keterangan |
| --- | --- | --- |
| `review_id` | UUID | Kunci utama |
| `cafe_id` | UUID | Kunci asing ke `cafes.csv` |
| `author_name` | str | Nama tampilan penulis |
| `rating` | float | Nilai ulasan |
| `text` | str | Teks ulasan lengkap |
| `published_at` | str | Tanggal ulasan dalam bentuk teks |
| `likes` | int | Jumlah suara bahwa ulasan membantu |
| `scraped_at` | ISO8601 | Waktu pengambilan data |

## Penyimpanan dan pemulihan CSV

### Migrasi dan ringkasan

Saat mulai, skrip memigrasikan CSV kafe dari 12 ke 16 kolom, termasuk berkas dengan campuran baris lama dan baru. Skrip menyimpan salinan asli di `data/raw/.csv_backups/`. Jika format tidak dikenali, skrip berhenti sebelum mengubah kolom.

Ringkasan mencatat jumlah ulasan yang diambil (`Reviews collected`), ditambahkan ke CSV (`CSV reviews added`), dan diterima atau dilewati oleh *backend* (`Backend inserted/skipped`). *Backend* melewati ulasan jika nama penulis dan tanggalnya sudah ada pada kafe yang sama.

### Pulihkan CSV dari salinan lokal

Jika gagal menulis CSV, skrip mencatat galat dan menyimpan salinan hasil ke `data/raw/csv_recovery/` selama folder bisa ditulis. Setelah memperbaiki penyebab galat, jalankan perintah berikut dari `machine_learning`.

```powershell
python scripts/replay_csv_recovery.py
```

Perintah ini hanya memulihkan CSV. Skrip melewati ID ulasan yang sudah ada dan memindahkan hasil yang berhasil dipulihkan ke subfolder `processed`. Jika proses lama gagal sebelum fitur ini tersedia, ekspor data dari basis data atau ambil ulang dari Maps.

### Ekspor CSV dari basis data

Jika hasil sudah masuk ke *backend* tetapi gagal tersimpan ke CSV, ekspor dari basis data. Dari `machine_learning`, jalankan Python milik *backend* yang sudah memiliki `asyncpg` dan `python-dotenv`.

```powershell
..\backend\.venv\Scripts\python.exe scripts/export_database_csv.py
```

Hasil bawaan ada di `data/db_export/<waktu-ekspor>/`, berisi `cafes.csv`, `reviews.csv`, dan `database_export.json`. Skrip mengambil data dalam satu transaksi baca saja dan memakai `raw_text` atau `text` jika `raw_text` tidak tersedia. Skrip memeriksa jumlah baris, isi CSV, dan relasi ulasan ke kafe. Gunakan kedua CSV sebagai pasangan karena ID di basis data berbeda dari ID *scraper*.

Ekspor mencakup data yang tersimpan di basis data. `source_query` dibiarkan kosong. `scraped_at` memakai waktu pembuatan data di basis data. `published_at` mengikuti tanggal yang tersimpan, termasuk tanggal pengganti dari proses pemasukan data sebelumnya.

Pilih folder tujuan dengan `--output-dir <folder>`. Sebelum mengganti CSV di folder itu, skrip mencadangkannya ke `.csv_backups/`. Simpan CSV lokal lama sampai ulasan yang hanya ada di sana selesai dibandingkan. Gabungkan data lokal dan hasil ekspor secara terpisah jika diperlukan.
