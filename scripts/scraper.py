"""
Usage:
    python scraper.py                          # Run full scrape
    python scraper.py --query "cafe sleman"   # Single custom query
    python scraper.py --max-cafes 50          # Limit number of cafes
    python scraper.py --resume                # Retry cafes without text reviews
    python scraper.py --login-only            # Manually sign in and retain the session

Output:
    data/raw/cafes.csv    — One row per cafe
    data/raw/reviews.csv  — One row per review, foreign key: cafe_id
"""

import asyncio
import csv
import json
import logging
import os
import random
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from playwright.async_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)
from tqdm import tqdm
from dotenv import load_dotenv

if __package__:
    from .scraper_parsers import (
        is_in_Malang, parse_coordinates, parse_price_range, parse_rating, parse_review_count,
    )
    from .scraper_reviews import ReviewAccessError, ReviewPanelError, dismiss_popups, scrape_reviews
    from .scraper_storage import (CAFE_FIELDS, REVIEW_FIELDS, cafe_key, load_resume_state, upsert_cafe_csv, ensure_cafe_csv_schema, append_review_rows, save_csv_recovery)
    from .scraper_browser import launch_browser, login_manually, login_in_regular_chrome
    from .scraper_details import extract_overview_details
    from .scraper_navigation import PLACE_HEADING, indonesian_maps_url, open_cafe_page, save_failure_diagnostics
else:
    from scraper_parsers import (
        is_in_Malang, parse_coordinates, parse_price_range, parse_rating, parse_review_count,
    )
    from scraper_reviews import ReviewAccessError, ReviewPanelError, dismiss_popups, scrape_reviews
    from scraper_storage import (CAFE_FIELDS, REVIEW_FIELDS, cafe_key, load_resume_state, upsert_cafe_csv, ensure_cafe_csv_schema, append_review_rows, save_csv_recovery)
    from scraper_browser import launch_browser, login_manually, login_in_regular_chrome
    from scraper_details import extract_overview_details
    from scraper_navigation import PLACE_HEADING, indonesian_maps_url, open_cafe_page, save_failure_diagnostics

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "change_me_before_production")

# ─── Configuration ────────────────────────────────────────────────────────────

# Output paths (relative to the machine_learning/ directory)
BASE_DIR = Path(__file__).resolve().parent.parent        # machine_learning/
DATA_RAW = BASE_DIR / "data" / "raw"
CAFES_CSV = DATA_RAW / "cafes.csv"
REVIEWS_CSV = DATA_RAW / "reviews.csv"
USER_DATA_DIR = BASE_DIR / "data" / ".browser_profile"
CHROME_DATA_DIR = BASE_DIR / "data" / ".browser_profile_chrome"
DIAGNOSTICS_DIR = BASE_DIR / "debug" / "scraper"

# Scraping targets — multiple queries for broad coverage of Malang cafes
SEARCH_QUERIES = [
    "cafe Kota Malang",
    "coffee shop Malang",
    "kedai kopi Malang",
    "kafe kayutangan Malang",
    "warung kopi Malang",
    "specialty coffee Malang",
    "kafe aesthetic Malang",
]

# Limits
MAX_CAFES_PER_QUERY = 60      # Google Maps typically shows up to 60 per search
MAX_REVIEWS_PER_CAFE = 20     # 20 reviews per cafe default as required
SCROLL_PAUSE_MS = 1200        # Milliseconds between scrolls (be polite to Google)
ACTION_DELAY_MS = (800, 1800) # Random delay range between actions (min, max)

# ─── Logging ──────────────────────────────────────────────────────────────────

DATA_RAW.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_RAW / "scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("cafind.scraper")


class CafeOutsideTarget(ValueError):
    """A successfully loaded place was intentionally filtered by coordinates."""

# ─── Helpers ──────────────────────────────────────────────────────────────────

def ensure_csv_headers(cafes_path: Path | None = None, reviews_path: Path | None = None) -> None:
    """Create missing CSVs and migrate recognized legacy cafe schemas safely."""
    cafes_path = cafes_path or CAFES_CSV
    reviews_path = reviews_path or REVIEWS_CSV
    cafes_path.parent.mkdir(parents=True, exist_ok=True)
    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_cafe_csv_schema(cafes_path, CAFE_FIELDS)
    if not reviews_path.exists():
        with open(reviews_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
            writer.writeheader()


def append_cafe(cafe: dict) -> None:
    with open(CAFES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAFE_FIELDS)
        writer.writerow(cafe)


def append_reviews(reviews: list[dict], path: Path | None = None) -> int:
    return append_review_rows(path or REVIEWS_CSV, reviews, REVIEW_FIELDS)

# ─── Page Interactions ────────────────────────────────────────────────────────

async def search_cafes(page: Page, query: str) -> None:
    """Navigate to Google Maps and search for [query]."""
    url = indonesian_maps_url(f"https://www.google.com/maps/search/{query.replace(' ', '+')}")
    log.info(f"Searching: {query} -> {url}")
    await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    await page.wait_for_timeout(random.uniform(1500, 2500))

    # Dismiss cookie consent if it appears (EU/Indonesia regions)
    try:
        accept_btn = page.get_by_role("button", name=re.compile(r"accept|setuju|terima", re.I))
        if await accept_btn.is_visible(timeout=3000):
            await accept_btn.click()
            await page.wait_for_timeout(500)
    except PlaywrightTimeoutError:
        pass
    await dismiss_popups(page)



async def scroll_results_panel(page: Page, target_cafes: int = MAX_CAFES_PER_QUERY) -> None:
    """Scroll the left-side results feed until target_cafes or no more results load.
    
    Google Maps lazily loads results in batches (typically 20 per batch).
    We scroll using hover, gradual mouse wheel gestures, and keyboard PageDown
    until we have enough place cards or hit the end of results.
    """
    feed = page.locator('[role="feed"]').first
    await page.wait_for_timeout(random.uniform(1500, 2500))
    await dismiss_popups(page)
    if not await feed.count() and "/maps/place/" in page.url:
        log.info("Search opened one place directly; no result feed to scroll.")
        return

    prev_count = 0
    stuck_attempts = 0
    max_scrolls = max(25, (target_cafes // 5) * 3 + 5)

    for scroll_idx in range(max_scrolls):
        # Auto-dismiss modal popups if they appear mid-feed
        await dismiss_popups(page)

        anchors = await page.locator('[role="feed"] a[href*="/maps/place/"]').all()
        unique_links = set()
        for a in anchors:
            h = await a.get_attribute("href")
            if h:
                unique_links.add(h)
        count = len(unique_links)

        # Stop as soon as target_cafes unique places are reached
        if count >= target_cafes:
            log.info(f"Results panel reached target: {count}/{target_cafes} items")
            break

        if count == prev_count:
            stuck_attempts += 1
            if stuck_attempts >= 6:
                log.info("No more new results loaded after 6 scroll attempts.")
                break
        else:
            stuck_attempts = 0
        prev_count = count

        # 1. Scroll last card into view, hover, and gradual mouse wheel
        last_card = page.locator('[role="feed"] a[href*="/maps/place/"]').last
        if await last_card.count() > 0:
            try:
                await last_card.scroll_into_view_if_needed(timeout=2000)
                await last_card.hover(timeout=1000)
            except Exception:
                pass

        for _ in range(random.randint(2, 4)):
            await page.mouse.wheel(0, random.randint(300, 600))
            await page.wait_for_timeout(random.uniform(100, 250))

        try:
            await page.keyboard.press("PageDown")
        except Exception:
            pass

        # 2. Smoothly scroll the feed element directly via JavaScript
        if await feed.count() > 0:
            try:
                await feed.evaluate("el => el.scrollBy({top: 800, behavior: 'smooth'})")
            except Exception:
                pass

        # Give Google Maps enough time (1.5-2.5s) to fetch and render the next batch
        await page.wait_for_timeout(random.uniform(1500, 2500))

        # 3. Check for "end of list" indicator
        end_text = page.get_by_text(re.compile(r"end of.{0,20}list|Anda telah mencapai", re.I))
        if await end_text.count() > 0:
            log.info("Reached end of results panel.")
            break

    log.info(f"Results panel: ~{count} unique places available")


async def collect_cafe_links(page: Page, max_cafes: int) -> list[str]:
    """Collect hrefs from all place cards in the results feed."""
    links: list[str] = []
    # Each result card is an <a> inside the feed with a /maps/place/ href
    anchors = await page.locator('[role="feed"] a[href*="/maps/place/"]').all()
    if not anchors and "/maps/place/" in page.url:
        return [indonesian_maps_url(page.url)]
    seen: set[str] = set()
    for anchor in anchors:
        if len(links) >= max_cafes:
            break
        href = await anchor.get_attribute("href")
        if href and href not in seen:
            seen.add(href)
            links.append(href)
    log.info(f"Collected {len(links)} unique cafe links")
    return links


async def scrape_cafe_details(page: Page, url: str, query: str) -> Optional[dict]:
    """Navigate to a cafe's Google Maps page and extract all metadata."""
    stage = "opening place page / waiting for cafe heading"
    try:
        await open_cafe_page(page, url)
        stage = "extracting cafe metadata"

        # ── Name ──
        name_el = page.locator(PLACE_HEADING).first
        name = (await name_el.inner_text()).strip()

        if not name:
            return None

        # ── Rating & Review Count ──
        rating = None
        review_count = 0
        rating_container = page.locator('div.F7nice').first
        if await rating_container.count() > 0:
            rating_text = await rating_container.inner_text()
            # Expecting "4.5\n(123)" or "4,5\n(1.234)"
            r_match = re.search(r"(\d[\.,]\d)", rating_text)
            if r_match:
                rating = float(r_match.group(1).replace(',', '.'))
            
            rev_match = re.search(r"\(([\d\.,]+)\)", rating_text)
            if rev_match:
                review_count = int(rev_match.group(1).replace('.', '').replace(',', ''))
        
        # Fallback for rating/review count
        if rating is None:
            rating_el = page.locator('[data-hide-tooltip-on-mouse-move] span[aria-hidden="true"]').first
            r_text = await rating_el.inner_text() if await rating_el.count() > 0 else ""
            rating = parse_rating(r_text)
        if review_count == 0:
            review_count_el = page.locator('[data-hide-tooltip-on-mouse-move] button span').first
            rc_text = await review_count_el.inner_text() if await review_count_el.count() > 0 else "0"
            review_count = parse_review_count(rc_text)

        # ── Address ──
        address = ""
        addr_btn = page.locator('[data-item-id="address"] .Io6YTe').first
        if await addr_btn.count() > 0:
            address = (await addr_btn.inner_text()).strip()

        # ── Categories ──
        categories = ""
        cat_el = page.locator('button[jsaction*="category"]').first
        if await cat_el.count() > 0:
            categories = (await cat_el.inner_text()).strip()
        else:
            subtitle = page.locator(".DkEaL").first
            if await subtitle.count() > 0:
                categories = (await subtitle.inner_text()).strip()

        # ── Phone ──
        phone = ""
        phone_el = page.locator('[data-item-id^="phone"] .Io6YTe').first
        if await phone_el.count() > 0:
            phone = (await phone_el.inner_text()).strip()

        # ── Website ──
        website = ""
        website_el = page.locator('[data-item-id="authority"] .Io6YTe').first
        if await website_el.count() > 0:
            website = (await website_el.inner_text()).strip()
            # Note: Google Maps often lists instagram.com if the owner set it as their website!

        # ── Coordinates ──
        try:
            # Wait briefly until the URL updates to include coordinates
            await page.wait_for_url(re.compile(r"(@|!3d|center=)-?\d+\.\d+"), timeout=3500)
        except PlaywrightTimeoutError:
            pass
        
        current_url = page.url
        lat, lng = parse_coordinates(current_url)
        if lat is None or lng is None:
            lat, lng = parse_coordinates(url)

        # Filter out-of-Malang results
        if not is_in_Malang(lat, lng):
            log.warning(f"Out of bounds: {name} @ ({lat}, {lng}) — skipping")
            raise CafeOutsideTarget(name)

        stage = "extracting price and weekly opening hours"
        details = await extract_overview_details(page)
        log.info("  Details: price=%s | min=%s | max=%s | hours=%s",
                 details["price_range"], details["price_min"], details["price_max"], details["opening_hours"])

        cafe_id = str(uuid.uuid4())
        return {
            "cafe_id": cafe_id,
            "name": name,
            "address": address,
            "latitude": lat if lat is not None else 0.0,
            "longitude": lng if lng is not None else 0.0,
            "rating": rating if rating is not None else 0.0,
            "review_count": review_count,
            "categories": [categories] if categories else [],
            "phone": phone,
            "website": website,
            **details,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "source_query": query,
        }

    except (ReviewAccessError, CafeOutsideTarget):
        raise
    except PlaywrightTimeoutError as exc:
        log.warning("Timeout scraping cafe details during %s: %s | %s", stage, url, exc)
        await save_failure_diagnostics(page, DIAGNOSTICS_DIR, url=url, stage=stage, error=exc)
        return None
    except Exception as e:
        if page.is_closed():
            raise
        log.error(f"Error scraping cafe {url}: {e}")
        await save_failure_diagnostics(page, DIAGNOSTICS_DIR, url=url, stage=stage, error=e)
        return None


# ─── Main Orchestrator ────────────────────────────────────────────────────────

async def run_scraper(
    queries: list[str],
    max_cafes_per_query: int,
    reviews_per_cafe: int = MAX_REVIEWS_PER_CAFE,
    resume: bool = False,
    headless: bool = False,
    login_only: bool = False,
    browser: str = "chromium",
    output_dir: Path | None = None,
    no_backend: bool = False,
    place_url: str | None = None,
) -> None:
    """Main scraping orchestrator.
    
    For each search query:
      1. Open Google Maps and search
      2. Scroll the results panel to reveal all listings
      3. Collect all place-page URLs
      4. For each URL: scrape cafe details + reviews (up to reviews_per_cafe)
      5. Append results to CSV incrementally and POST to backend
    """
    if login_only and headless:
        raise ValueError("--login-only tidak dapat digabung dengan --headless")
    if browser not in {"chromium", "chrome"}:
        raise ValueError("Browser harus chromium atau chrome")
    profile_dir = CHROME_DATA_DIR if browser == "chrome" else USER_DATA_DIR
    if login_only and browser == "chrome":
        await asyncio.to_thread(login_in_regular_chrome, profile_dir)
        return
    cafes_path = Path(output_dir) / "cafes.csv" if output_dir else CAFES_CSV
    reviews_path = Path(output_dir) / "reviews.csv" if output_dir else REVIEWS_CSV
    if not login_only:
        ensure_csv_headers(cafes_path, reviews_path)
    existing = load_resume_state(cafes_path, reviews_path) if resume and not login_only else {}
    attempted = set()
    total_cafes_scraped = 0
    total_reviews_scraped = 0
    detail_failures = 0
    query_failures = 0
    review_failures = 0
    details_only_updates = 0
    skipped_cafes = 0
    csv_failures = csv_reviews_written = 0
    backend_inserted = backend_skipped = backend_failures = 0

    async with async_playwright() as pw:
        context = await launch_browser(pw, profile_dir, headless=headless, browser=browser)
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            if login_only:
                await login_manually(page, profile_dir)
                return

            for query in queries:
                log.info(f"\n{'='*60}")
                log.info(f"Query: {query}")
                log.info(f"{'='*60}")

                try:
                    if place_url:
                        cafe_urls = [indonesian_maps_url(place_url)]
                    else:
                        await search_cafes(page, query)
                        await scroll_results_panel(page, target_cafes=max_cafes_per_query)
                        cafe_urls = await collect_cafe_links(page, max_cafes_per_query)

                    for url in tqdm(cafe_urls, desc=f"Scraping: {query[:30]}"):
                        await asyncio.sleep(random.randint(*ACTION_DELAY_MS) / 1000)

                        # ── Scrape cafe details ──
                        try:
                            cafe = await scrape_cafe_details(page, url, query)
                        except CafeOutsideTarget:
                            skipped_cafes += 1
                            continue
                        if cafe is None:
                            detail_failures += 1
                            continue

                        key = cafe_key(cafe)
                        previous = existing.get(key)
                        if key in attempted:
                            skipped_cafes += 1
                            log.info("  Skipping (already has reviews or attempted this run): %s", cafe["name"])
                            continue
                        attempted.add(key)
                        if previous:
                            cafe["cafe_id"] = previous.cafe_id
                            log.info("  Reusing existing cafe ID: %s", cafe["name"])

                        # ── Scrape reviews (per-cafe quota) ──
                        try:
                            if previous and previous.has_reviews:
                                log.info("  Updating cafe details; saved reviews are retained: %s", cafe["name"])
                                details_only_updates += 1
                                reviews = []
                            else:
                                reviews = await scrape_reviews(page, cafe["cafe_id"], max_reviews=reviews_per_cafe)
                        except ReviewPanelError as exc:
                            review_failures += 1
                            log.error("Review scraping failed for %s: %s", cafe["name"], exc)
                            await save_failure_diagnostics(page, DIAGNOSTICS_DIR, url=url, stage="opening/extracting reviews", error=exc)
                            continue
                        if not reviews and not (previous and previous.has_reviews):
                            log.warning("No text reviews collected for %s; this does not imply the place has no reviews.", cafe["name"])

                        total_cafes_scraped += 1
                        total_reviews_scraped += len(reviews)

                        log.info(
                            f"  * {cafe['name']} | Rating:{cafe['rating']} "
                            f"| {len(reviews)} reviews | "
                            f"({cafe['latitude']}, {cafe['longitude']})"
                        )

                        # ── Save locally to CSV ──
                        try:
                            upsert_cafe_csv(cafes_path, cafe, CAFE_FIELDS)
                            csv_reviews_written += append_reviews(reviews, reviews_path)
                        except Exception as e:
                            csv_failures += 1
                            log.error("    Failed to save CSV: %s", e)
                            try:
                                recovery = save_csv_recovery(cafes_path.parent / "csv_recovery", cafe, reviews, e)
                                log.warning("    Local recovery saved: %s", recovery)
                            except Exception as recovery_error:
                                log.error("    Failed to save recovery data: %s", recovery_error)

                        # ── Post to FastAPI Backend ──
                        payload = cafe.copy()
                        payload["reviews"] = reviews

                        if not no_backend:
                            try:
                                api_res = await context.request.post(
                                    f"{BACKEND_URL}/api/admin/insert_reviews",
                                    headers={"X-API-Key": ADMIN_API_KEY},
                                    data=payload,
                                )
                                if api_res.ok:
                                    res_json = await api_res.json()
                                    backend_inserted += res_json.get("reviews_inserted", 0)
                                    backend_skipped += res_json.get("reviews_skipped", 0)
                                    log.info(f"    Backend: {res_json.get('message')}")
                                else:
                                    backend_failures += 1
                                    err_text = await api_res.text()
                                    log.error(f"    Backend Error ({api_res.status}): {err_text}")
                            except Exception as e:
                                backend_failures += 1
                                log.error(f"    Failed to contact backend: {e}")

                        await page.wait_for_timeout(random.uniform(500, 1000))

                except ReviewAccessError:
                    raise
                except Exception as e:
                    if page.is_closed():
                        raise RuntimeError("Browser ditutup; scraping dihentikan. Jalankan kembali dengan --resume.") from e
                    log.error(f"Error processing query '{query}': {e}")
                    query_failures += 1
                    continue

        finally:
            await context.close()

    log.info(f"\n{'='*60}")
    log.info("Scraping finished%s", " with failures" if detail_failures or query_failures or review_failures or csv_failures or backend_failures else "")
    log.info(f"   Cafes processed   : {total_cafes_scraped}")
    log.info(f"   Reviews collected : {total_reviews_scraped}")
    log.info("   CSV reviews added : %s", csv_reviews_written)
    log.info("   CSV save failures : %s", csv_failures)
    if not no_backend:
        log.info("   Backend inserted  : %s", backend_inserted)
        log.info("   Backend skipped   : %s", backend_skipped)
        log.info("   Backend failures  : %s", backend_failures)
    log.info("   Cafes skipped     : %s", skipped_cafes)
    log.info("   Detail failures   : %s", detail_failures)
    log.info("   Query failures    : %s", query_failures)
    log.info("   Review failures   : %s", review_failures)
    log.info("   Details-only updates: %s", details_only_updates)
    if csv_failures or backend_failures:
        raise RuntimeError("Sebagian penyimpanan gagal. Periksa ringkasan CSV/backend dan data/raw/csv_recovery.")
    if total_cafes_scraped == 0 and (detail_failures or query_failures or review_failures):
        raise RuntimeError("Tidak ada kafe yang berhasil diproses karena kegagalan. Periksa error dan folder debug/scraper.")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Cafind — Google Maps scraper for Malang cafes"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single custom query instead of the default list",
    )
    parser.add_argument(
        "--max-cafes",
        type=int,
        default=MAX_CAFES_PER_QUERY,
        help=f"Max cafes to scrape per query (default: {MAX_CAFES_PER_QUERY})",
    )
    parser.add_argument(
        "--reviews-per-cafe",
        type=int,
        default=MAX_REVIEWS_PER_CAFE,
        help=f"Max reviews to scrape per cafe (default: {MAX_REVIEWS_PER_CAFE})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Refresh cafe details with existing IDs; retain saved reviews and retry zero-review cafes",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no visible window)",
    )
    parser.add_argument(
        "--login-only",
        action="store_true",
        help="Set up manual Google login; use --browser chrome for regular Chrome without Playwright",
    )
    parser.add_argument(
        "--browser", choices=["chromium", "chrome"], default="chromium",
        help="Browser/profile to use (default: chromium); chrome requires installed Google Chrome",
    )
    parser.add_argument("--output-dir", type=Path, help="Write cafe/review CSVs into a separate directory")
    parser.add_argument("--no-backend", action="store_true", help="Save CSVs only; do not contact the backend")
    parser.add_argument("--place-url", help="Scrape one full Google Maps place URL instead of search results")
    args = parser.parse_args()
    if args.login_only and args.headless:
        parser.error("--login-only memerlukan browser terlihat; hapus --headless")
    if args.max_cafes < 1 or args.reviews_per_cafe < 1:
        parser.error("--max-cafes dan --reviews-per-cafe harus lebih dari 0")

    queries = [args.query] if args.query else (["direct place"] if args.place_url else SEARCH_QUERIES)

    log.info("Cafind Google Maps Scraper (API Pipeline)")
    log.info(f"  Queries           : {queries}")
    log.info(f"  Max cafes/query   : {args.max_cafes}")
    log.info(f"  Reviews per cafe  : {args.reviews_per_cafe}")
    log.info(f"  Resume mode       : {args.resume}")
    log.info(f"  Headless mode     : {args.headless}")
    log.info(f"  Browser           : {args.browser}")
    log.info(f"  Backend URL       : {BACKEND_URL}")

    asyncio.run(
        run_scraper(
            queries=queries,
            max_cafes_per_query=args.max_cafes,
            reviews_per_cafe=args.reviews_per_cafe,
            resume=args.resume,
            headless=args.headless,
            login_only=args.login_only,
            browser=args.browser,
            output_dir=args.output_dir,
            no_backend=args.no_backend,
            place_url=args.place_url,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ReviewAccessError, RuntimeError) as exc:
        log.error("Scraping dihentikan: %s", exc)
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        log.info("Dibatalkan. Jalankan kembali dengan --resume setelah sesi siap.")
        sys.exit(130)
