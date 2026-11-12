"""Shared persistent browser configuration and interactive session setup."""

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path

from playwright.async_api import BrowserContext, Page, Playwright

log = logging.getLogger("cafind.scraper")


def find_chrome() -> Path:
    """Find installed Chrome without using or copying the user's everyday profile."""
    candidates = []
    if os.getenv("CHROME_PATH"):
        candidates.append(Path(os.environ["CHROME_PATH"]))
    for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        if os.getenv(variable):
            candidates.append(Path(os.environ[variable]) / "Google/Chrome/Application/chrome.exe")
    if executable := shutil.which("chrome"):
        candidates.append(Path(executable))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("Google Chrome tidak ditemukan. Instal Chrome atau set CHROME_PATH ke lokasi chrome.exe.")


def login_in_regular_chrome(profile_dir: Path) -> None:
    """Open Chrome normally for manual sign-in, without Playwright or remote debugging."""
    executable = find_chrome()
    profile_dir.mkdir(parents=True, exist_ok=True)
    log.info("Membuka Chrome biasa dengan profil scraper: %s", profile_dir)
    log.info(
        "Login di halaman Google Maps dan coba buka semua ulasan. "
        "Setelah selesai, TUTUP SEMUA jendela Chrome dari profil scraper ini, "
        "lalu kembali ke terminal dan tekan Enter."
    )
    try:
        process = subprocess.Popen([
            str(executable), f"--user-data-dir={profile_dir.resolve()}",
            "--no-first-run", "--no-default-browser-check", "--new-window",
            "https://www.google.com/maps?hl=id",
        ])
    except OSError as exc:
        raise RuntimeError(f"Tidak dapat membuka Chrome: {exc}") from exc
    input("Setelah login dan semua jendela profil scraper ditutup, tekan Enter: ")
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Chrome masih terbuka. Tutup jendela profil scraper sebelum menjalankan scraping.") from exc
    if process.returncode:
        raise RuntimeError(f"Chrome keluar dengan kode {process.returncode}; periksa browser sebelum scraping.")
    log.info("Profil Chrome siap dipakai dengan --browser chrome. Keberhasilan login perlu diperiksa manual.")


async def launch_browser(
    pw: Playwright, profile_dir: Path, *, headless: bool = False, browser: str = "chromium",
) -> BrowserContext:
    """Reuse the dedicated profile and the selected browser's real user agent."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    options = {"executable_path": str(find_chrome())} if browser == "chrome" else {}
    return await pw.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        headless=headless,
        args=["--lang=id-ID"],
        locale="id-ID",
        timezone_id="Asia/Jakarta",
        geolocation={"latitude": -7.9734, "longitude": 112.6326},
        permissions=["geolocation"],
        viewport={"width": 1280, "height": 900},
        **options,
    )


async def login_manually(page: Page, profile_dir: Path) -> None:
    """Leave authentication to the user; closing the context flushes the profile."""
    await page.goto("https://www.google.com/maps?hl=id", wait_until="domcontentloaded", timeout=30_000)
    log.info("Profil login: %s", profile_dir)
    log.info(
        "Di Chromium: klik Login di Google Maps, selesaikan login/2FA sendiri, "
        "lalu buka sebuah tempat > Ulasan > Lihat ulasan lainnya. "
        "Pastikan daftar ulasan lengkap bisa dibuka. Jangan tutup browser dahulu."
    )
    await asyncio.to_thread(input, "Setelah selesai, kembali ke terminal dan tekan Enter untuk menyimpan profil: ")
    if page.is_closed():
        raise RuntimeError("Browser login sudah ditutup. Ulangi --login-only dan tekan Enter sebelum menutup browser.")
    log.info("Sesi akan disimpan saat browser ditutup. Status login tidak diverifikasi otomatis.")
