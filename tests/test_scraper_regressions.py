"""Offline regressions: python -m unittest machine_learning.tests.test_scraper_regressions -v."""

import csv
import asyncio
import subprocess
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

from machine_learning.scripts import scraper
from machine_learning.scripts import scraper_browser
from machine_learning.scripts.scraper_browser import launch_browser
from machine_learning.scripts.scraper_reviews import (
    ReviewAccessError,
    ReviewPanelError,
    dismiss_popups,
    open_more_reviews_if_needed,
    scrape_reviews,
    review_counts,
    original_review_text,
)
from machine_learning.scripts.scraper_storage import load_resume_state, upsert_cafe_csv
from machine_learning.scripts.scraper_details import extract_overview_details
from machine_learning.scripts.scraper_parsers import parse_price_range
from machine_learning.scripts.scraper_navigation import open_cafe_page, same_place, save_failure_diagnostics, indonesian_maps_url
from machine_learning.scripts.prepare_review_language_repair import propose_changes, build_sql, sql_text


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cafes = Path(self.temp.name) / "cafes.csv"
        self.reviews = Path(self.temp.name) / "reviews.csv"

    def write_csv(self, path, fields, rows):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_missing_files(self):
        self.assertEqual(load_resume_state(self.cafes, self.reviews), {})

    def test_zero_or_blank_reviews_remain_retryable(self):
        self.write_csv(self.cafes, ["cafe_id", "name", "address"], [
            {"cafe_id": "old", "name": "Cafe", "address": "A"},
        ])
        self.write_csv(self.reviews, ["cafe_id", "text"], [{"cafe_id": "old", "text": " "}])
        state = load_resume_state(self.cafes, self.reviews)[("cafe", "a")]
        self.assertEqual(state.cafe_id, "old")
        self.assertFalse(state.has_reviews)

    def test_historical_duplicates_prefer_id_with_reviews(self):
        self.write_csv(self.cafes, ["cafe_id", "name", "address"], [
            {"cafe_id": "reviewed", "name": "Cafe", "address": "A"},
            {"cafe_id": "empty", "name": " CAFE ", "address": "a"},
            {"cafe_id": "other-branch", "name": "Cafe", "address": "B"},
        ])
        self.write_csv(self.reviews, ["cafe_id", "text"], [{"cafe_id": "reviewed", "text": "Enak"}])
        state = load_resume_state(self.cafes, self.reviews)
        self.assertEqual(len(state), 2)
        self.assertEqual(state[("cafe", "a")].cafe_id, "reviewed")
        self.assertTrue(state[("cafe", "a")].has_reviews)
        self.assertFalse(state[("cafe", "b")].has_reviews)

    def test_csv_refreshes_metadata_without_duplicate_cafe_or_erasing_known_values(self):
        fields = ["cafe_id", "name", "address", "price_range", "price_min", "price_max", "opening_hours"]
        cafe = dict(cafe_id="old", name="Cafe", address="A", price_range=None, price_min=None, price_max=None, opening_hours=None)
        upsert_cafe_csv(self.cafes, cafe, fields)
        cafe.update(price_range="Rp 25–50K", price_min=25000.0, price_max=50000.0, opening_hours="Monday: Open 24 hours")
        upsert_cafe_csv(self.cafes, cafe, fields)
        cafe.update(price_range=None, price_min=None, price_max=None, opening_hours=None)
        upsert_cafe_csv(self.cafes, cafe, fields)
        with self.cafes.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["price_min"], "25000.0")
        self.assertEqual(rows[0]["price_max"], "50000.0")
        self.assertEqual(rows[0]["opening_hours"], "Monday: Open 24 hours")


class PriceParserTests(unittest.TestCase):
    def test_rupiah_formats_and_shared_units(self):
        for raw, expected in [
            ("Rp 25–50K", (25000, 50000)),
            ("Rp 25–50 rb", (25000, 50000)),
            ("Rp 25K–Rp 50K", (25000, 50000)),
            ("Rp 25.000 to Rp 50.000", (25000, 50000)),
            ("Price range, Rp\u00a025,000–50,000 per person, Reported by 61 people", (25000, 50000)),
            ("Rp 25,5–50,5 ribu", (25500, 50500)),
            ("Rp 25.000–50K", (25000, 50000)),
            ("IDR 20.000", (20000, 20000)),
            ("Rp 1 hingga Rp 25.000", (1, 25000)),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(parse_price_range(raw)[1:], expected)

    def test_missing_or_categorical_prices_do_not_invent_numeric_amounts(self):
        self.assertEqual(parse_price_range(None), (None, None, None))
        self.assertEqual(parse_price_range("$$"), ("$$", None, None))
        self.assertEqual(parse_price_range("No price information"), (None, None, None))


class LanguageRepairTests(unittest.TestCase):
    def test_maps_language_overrides_english_without_losing_place_or_account(self):
        result = indonesian_maps_url("https://www.google.com/maps/place/Cafe/data=!1s123?hl=en&authuser=0&hl=fr")
        self.assertIn("hl=id", result)
        self.assertIn("authuser=0", result)
        self.assertNotIn("hl=en", result)
        self.assertNotIn("hl=fr", result)
        self.assertIn("data=!1s123", result)
        self.assertEqual(indonesian_maps_url("https://example.test/maps/place/Cafe"), "https://example.test/maps/place/Cafe")

    def test_ambiguous_review_match_is_not_proposed(self):
        old = [dict(review_id="one", author_name="Budi", rating=5, text="Good"), dict(review_id="two", author_name="Budi", rating=5, text="Good")]
        fresh = [dict(author_name="Budi", rating=5, text="Bagus")]
        changes, unmatched = propose_changes(old, fresh)
        self.assertEqual(changes, [])
        self.assertEqual(len(unmatched), 1)

    def test_repair_sql_preserves_ids_and_guards_old_text(self):
        changes, _ = propose_changes(
            [dict(review_id="old", db_review_id="11111111-1111-4111-8111-111111111111", author_name="O'Brien", rating=5, text="Good coffee")],
            [dict(author_name="O'Brien", rating=5, text="Kopi enak")],
        )
        preview, apply = build_sql("22222222-2222-4222-8222-222222222222", "Cafe", changes, "20261017_test")
        self.assertIn("O'Brien".encode("utf-8").hex(), preview)
        self.assertIn("r.id = f.db_review_id", preview)
        self.assertIn("COALESCE(r.raw_text, r.text) = f.old_text", preview)
        self.assertTrue(preview.endswith("ROLLBACK;\n"))
        self.assertIn("CREATE TABLE review_language_backup_", apply)
        self.assertIn("is_processed = false", apply)
        self.assertNotIn("DELETE FROM reviews", apply)
        self.assertTrue(apply.endswith("COMMIT;\n"))

    def test_sql_text_survives_windows_newlines_quotes_and_emoji(self):
        import re
        original = "Line one\n\nLine two\r\nO'Brien 🥰"
        encoded = sql_text(original)
        windows_script = ("SELECT " + encoded + ";\n").replace("\n", "\r\n")
        hex_value = re.search(r"decode\('([0-9a-f]+)'", windows_script).group(1)
        self.assertEqual(bytes.fromhex(hex_value).decode("utf-8"), original)
        self.assertNotIn(original, windows_script)


class RegularChromeTests(unittest.TestCase):
    def test_manual_login_starts_regular_browser_with_dedicated_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = Path(temp) / "profile with spaces"
            with patch.object(scraper_browser, "find_chrome", return_value=Path("C:/Chrome/chrome.exe")), \
                 patch.object(scraper_browser.subprocess, "Popen") as popen, \
                 patch("builtins.input", return_value=""):
                popen.return_value.returncode = 0
                scraper_browser.login_in_regular_chrome(profile)
                command = popen.call_args.args[0]
                self.assertIn(f"--user-data-dir={profile.resolve()}", command)
                self.assertFalse(any("remote-debugging" in arg or "enable-automation" in arg for arg in command))
                popen.return_value.wait.assert_called_once_with(timeout=10)

    def test_open_chrome_is_reported_before_scraping(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(scraper_browser, "find_chrome", return_value=Path("C:/Chrome/chrome.exe")), \
                 patch.object(scraper_browser.subprocess, "Popen") as popen, \
                 patch("builtins.input", return_value=""):
                popen.return_value.wait.side_effect = subprocess.TimeoutExpired("chrome", 10)
                with self.assertRaisesRegex(RuntimeError, "Chrome masih terbuka"):
                    scraper_browser.login_in_regular_chrome(Path(temp))


class BrowserDOMTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=True)
        self.page = await self.browser.new_page()

    async def asyncTearDown(self):
        await self.browser.close()
        await self.pw.stop()


class NavigationDOMTests(BrowserDOMTestCase):
    async def test_place_is_usable_while_domcontentloaded_is_blocked(self):
        release = asyncio.Event()
        async def delayed_script(route):
            await release.wait()
            await route.fulfill(body="", content_type="application/javascript")
        await self.page.route("https://example.test/slow.js", delayed_script)
        url = "https://example.test/maps/place/Cafe/data=!1splace-one!3d-7.9!4d112.6"
        await self.page.route(url, lambda route: route.fulfill(
            body='<html><head><script defer src="/slow.js"></script></head><body><main role="main"><h1 class="DUwDvf">Cafe</h1></main></body></html>',
            content_type="text/html",
        ))
        try:
            with self.assertRaises(PlaywrightTimeoutError):
                await self.page.goto(url, wait_until="domcontentloaded", timeout=500)
            await open_cafe_page(self.page, url)
            self.assertEqual(await self.page.locator("h1").inner_text(), "Cafe")
            self.assertFalse(release.is_set())
        finally:
            release.set()
            await self.page.wait_for_load_state("domcontentloaded")

    async def test_timeout_can_recover_when_requested_place_is_already_rendered(self):
        url = "https://example.test/maps/place/Cafe/data=!1splace-one"
        await self.page.route(url, lambda route: route.fulfill(body='<h1 class="DUwDvf">Cafe</h1>', content_type="text/html"))
        await self.page.goto(url)
        with patch.object(self.page, "goto", AsyncMock(side_effect=PlaywrightTimeoutError("commit event timeout"))):
            await open_cafe_page(self.page, url, attempts=1)

    async def test_failure_diagnostics_are_local_screenshot_and_context(self):
        await self.page.set_content('<h1>Loading failed</h1>')
        with tempfile.TemporaryDirectory() as temp:
            await save_failure_diagnostics(self.page, Path(temp), url="target", stage="navigation", error=RuntimeError("timeout"))
            self.assertEqual(len(list(Path(temp).glob("*.png"))), 1)
            self.assertEqual(len(list(Path(temp).glob("*.json"))), 1)


class MetadataDOMTests(BrowserDOMTestCase):
    async def test_reads_real_overview_price_and_seven_day_schedule(self):
        days = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        rows = "".join(f'<tr><td>{day}</td><td><ul><li>7.00 am–11.00 pm</li></ul></td><td><button>Copy open hours</button></td></tr>' for day in days)
        await self.page.set_content(f'''
            <main role="main"><h1>Cafe</h1>
              <span role="img" aria-label="Rp&nbsp;25.000 to Rp&nbsp;50.000">Rp 25–50K</span>
              <div role="button" jsaction="pane.openhours.example.dropdown" aria-expanded="false"
                   onclick="const t=document.querySelector('#hours');t.hidden=!t.hidden;this.setAttribute('aria-expanded',String(!t.hidden))">
                Open · Closes 11.00 pm
              </div>
              <table aria-label="Price range histogram"><tr><td>Rp 1–25.000</td><td>8%</td></tr></table>
              <table id="hours" class="eK4R0e" hidden>{rows}</table>
              <table><tr><td>5</td><td>114 reviews</td></tr></table>
            </main>
        ''')
        details = await extract_overview_details(self.page)
        self.assertEqual(details["price_min"], 25000.0)
        self.assertEqual(details["price_max"], 50000.0)
        self.assertEqual(len(details["opening_hours"].split("; ")), 7)
        for day in days:
            self.assertIn(f"{day}: 7.00 am–11.00 pm", details["opening_hours"])
        self.assertNotIn("Copy", details["opening_hours"])
        self.assertFalse(await self.page.locator('#hours').is_visible())

    async def test_indonesian_split_intervals_and_closed_days(self):
        await self.page.set_content('''
            <main role="main"><h1>Kafe</h1><span role="img" aria-label="Rp 10–25 rb">Rp 10–25 rb</span>
              <div data-item-id="oh" aria-expanded="true">Buka</div>
              <table><tr><td>Senin</td><td><ul><li>08.00–12.00</li><li>14.00–20.00</li></ul></td><td>Salin</td></tr>
              <tr><td>Selasa</td><td>Tutup</td><td>Salin</td></tr><tr><td>Rabu</td><td>Buka 24 jam</td></tr></table>
            </main>
        ''')
        details = await extract_overview_details(self.page)
        self.assertEqual(details["opening_hours"], "Senin: 08.00–12.00 14.00–20.00; Selasa: Tutup; Rabu: Buka 24 jam")
        self.assertEqual(details["price_min"], 10000.0)


class NavigationRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_navigation_does_not_reuse_previous_cafe_heading(self):
        page = MagicMock()
        page.url = "https://example.test/maps/place/Previous/data=!1sother-place"
        page.is_closed.return_value = False
        page.goto = AsyncMock(side_effect=PlaywrightTimeoutError("no commit"))
        page.wait_for_timeout = AsyncMock()
        with self.assertRaises(PlaywrightTimeoutError):
            await open_cafe_page(page, "https://example.test/maps/place/Target/data=!1starget-place")
        self.assertEqual(page.goto.await_count, 2)
        page.locator.assert_not_called()

    def test_same_place_ignores_viewport_but_checks_place_id(self):
        target = "https://google.com/maps/place/Cafe/data=!1s0x123:0x456!3d-7.9"
        self.assertTrue(same_place("https://google.com/maps/place/Cafe/@-7.9,112.6/data=!1s0x123:0x456", target))
        self.assertFalse(same_place("https://google.com/maps/place/Cafe/data=!1sdifferent", target))


class ReviewDOMTests(BrowserDOMTestCase):

    async def test_original_text_toggle_avoids_saving_translation(self):
        await self.page.set_content('''
            <div data-review-id="r1"><span class="wiI7pd" id="translation">Cheap and cheerful</span>
              <div class="wiI7pd" id="original" hidden>Murah meriah</div>
              <button onclick="document.querySelector('#translation').hidden=true;document.querySelector('#original').hidden=false;this.remove()">Lihat aslinya</button>
            </div>
        ''')
        self.assertEqual(await original_review_text(self.page.locator('[data-review-id]')), "Murah meriah")

    async def test_original_text_ignores_hidden_copy(self):
        await self.page.set_content('<div data-review-id="r1"><span class="wiI7pd" hidden>English copy</span><span class="wiI7pd">Kopi enak</span></div>')
        self.assertEqual(await original_review_text(self.page.locator('[data-review-id]')), "Kopi enak")

    async def test_all_reviews_filter_is_not_a_panel_open_button(self):
        await self.page.set_content('''
            <button aria-label="All reviews" onclick="window.filterClicked=true">All</button>
            <button aria-label="Sort reviews">Sort</button>
            <div data-review-id="r1"><span class="wiI7pd">Enak</span></div>
        ''')
        self.assertTrue(await open_more_reviews_if_needed(self.page))
        self.assertIsNone(await self.page.evaluate("window.filterClicked"))
        self.assertEqual(len(await scrape_reviews(self.page, "c1", max_reviews=1)), 1)

    async def test_real_review_panel_can_coexist_with_background_more_button(self):
        await self.page.set_content('''
            <button onclick="window.wrongClick=true">More reviews</button>
            <button aria-label="Sort reviews">Sort</button>
            <div data-review-id="r1">Review</div>
        ''')
        self.assertTrue(await open_more_reviews_if_needed(self.page))
        self.assertIsNone(await self.page.evaluate("window.wrongClick"))

    async def test_review_dialog_is_never_dismissed_as_a_popup(self):
        await self.page.set_content('''
            <div role="dialog"><button aria-label="Close" onclick="this.parentElement.remove()">X</button>
              <div data-review-id="r1">Review</div>
            </div>
            <script>document.addEventListener('keydown',e=>{if(e.key==='Escape')document.querySelector('[role=dialog]').remove()})</script>
        ''')
        self.assertFalse(await dismiss_popups(self.page))
        self.assertTrue(await self.page.locator('[role="dialog"]').is_visible())

    async def test_counts_ignore_nested_ids_hidden_cards_and_translated_duplicates(self):
        await self.page.set_content('''
            <div data-review-id="hidden" hidden>Old review</div>
            <div data-review-id="r1"><div data-review-id="r1">
              <span class="wiI7pd">Enak</span><div class="wiI7pd">Tasty</div>
              <button data-review-id="r1">Like</button>
            </div></div>
            <div data-review-id="r2">Rating only</div>
        ''')
        self.assertEqual(await review_counts(self.page), (2, 1))

    async def test_login_gate_stops_after_one_click_and_stays_visible(self):
        await self.page.set_content('''
            <button onclick="window.clicks=(window.clicks||0)+1;document.querySelector('[role=dialog]').hidden=false">
              Lihat ulasan lainnya (1.687)
            </button>
            <div role="dialog" hidden>Login untuk menikmati fitur terbaik dari Google Maps
              <button onclick="this.parentElement.hidden=true">Tutup</button>
            </div>
        ''')
        with self.assertRaises(ReviewAccessError):
            await open_more_reviews_if_needed(self.page)
        self.assertEqual(await self.page.evaluate("window.clicks"), 1)
        self.assertTrue(await self.page.locator('[role="dialog"]').is_visible())
        with self.assertRaises(ReviewAccessError):
            await dismiss_popups(self.page)

    async def test_limited_banner_stops_before_clicking(self):
        for text in ["Anda melihat tampilan terbatas dari Google Maps.", "You're seeing a limited view of Google Maps."]:
            await self.page.set_content(f'<p>{text}</p><button onclick="window.clicked=true">More reviews</button>')
            with self.assertRaises(ReviewAccessError):
                await open_more_reviews_if_needed(self.page)
            self.assertIsNone(await self.page.evaluate("window.clicked"))

    async def test_hidden_gate_does_not_block_successful_panel(self):
        await self.page.set_content('''
            <div role="dialog" hidden>Sign in to get the best of Google Maps</div>
            <button onclick="this.remove();document.querySelector('[data-review-id]').hidden=false">See all reviews</button>
            <div data-review-id="r1" hidden>Review</div>
        ''')
        self.assertTrue(await open_more_reviews_if_needed(self.page))

    async def test_noop_click_is_not_success_even_with_preview_card(self):
        await self.page.set_content('''
            <button onclick="window.clicks=(window.clicks||0)+1">More reviews</button>
            <div data-review-id="preview">Preview card</div>
        ''')
        self.assertFalse(await open_more_reviews_if_needed(self.page))
        self.assertEqual(await self.page.evaluate("window.clicks"), 1)

    async def test_regular_popup_can_still_be_closed(self):
        await self.page.set_content('''
            <div role="dialog">Promo<button onclick="this.parentElement.remove()">Tutup</button></div>
        ''')
        self.assertTrue(await dismiss_popups(self.page))
        self.assertEqual(await self.page.locator('[role="dialog"]').count(), 0)

    async def test_text_expansion_is_scoped_to_review_cards(self):
        await self.page.set_content('''
            <button onclick="window.unrelated=true">Lainnya</button>
            <div data-review-id="r1">
              <span class="d4r55">Budi</span><span aria-label="4 bintang"></span>
              <span class="wiI7pd">Kopi</span><span class="rsqaWe">kemarin</span>
              <button aria-label="More actions" onclick="window.unrelated=true">...</button>
              <button onclick="this.parentElement.querySelector('.wiI7pd').textContent='Kopi enak';this.remove()">Selengkapnya</button>
            </div>
        ''')
        reviews = await scrape_reviews(self.page, "c1", max_reviews=1)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["text"], "Kopi enak")
        self.assertEqual(reviews[0]["cafe_id"], "c1")
        self.assertEqual(reviews[0]["rating"], 4.0)
        self.assertIsNone(await self.page.evaluate("window.unrelated"))


class PersistentProfileTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_reuses_saved_session_in_dedicated_profile(self):
        await self.check_saved_session("chromium")

    async def test_installed_chrome_reuses_saved_session_in_dedicated_profile(self):
        try:
            scraper_browser.find_chrome()
        except RuntimeError:
            self.skipTest("Google Chrome is not installed")
        await self.check_saved_session("chrome")

    async def check_saved_session(self, browser):
        with tempfile.TemporaryDirectory() as temp:
            async with async_playwright() as pw:
                profile = Path(temp) / "profile"
                context = await launch_browser(pw, profile, headless=True, browser=browser)
                try:
                    await context.add_cookies([{
                        "name": "test_session", "value": "saved", "url": "https://example.test",
                        "expires": time.time() + 3600,
                    }])
                finally:
                    await context.close()
                context = await launch_browser(pw, profile, headless=True, browser=browser)
                try:
                    cookies = await context.cookies("https://example.test")
                    self.assertTrue(any(c["name"] == "test_session" and c["value"] == "saved" for c in cookies))
                finally:
                    await context.close()


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_failure_is_reported_while_csv_is_retained(self):
        with ExitStack() as stack:
            context, cafe = self.setup_pipeline(stack)
            context.request.post.return_value.ok = False
            context.request.post.return_value.status = 500
            context.request.post.return_value.text = AsyncMock(return_value="database unavailable")
            stack.enter_context(patch.object(scraper, "scrape_reviews", AsyncMock(return_value=[{"review_id": "r1", "text": "Enak"}])))
            with self.assertRaisesRegex(RuntimeError, "Sebagian penyimpanan gagal"):
                await scraper.run_scraper(["one"], 1)
            with scraper.REVIEWS_CSV.open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)

    async def test_csv_failure_saves_recovery_and_still_sends_backend(self):
        with ExitStack() as stack:
            context, cafe = self.setup_pipeline(stack)
            reviews = [{"review_id": "r1", "cafe_id": cafe["cafe_id"], "text": "Enak"}]
            stack.enter_context(patch.object(scraper, "scrape_reviews", AsyncMock(return_value=reviews)))
            stack.enter_context(patch.object(scraper, "upsert_cafe_csv", side_effect=OSError("locked")))
            with self.assertRaisesRegex(RuntimeError, "Sebagian penyimpanan gagal"):
                await scraper.run_scraper(["one"], 1)
            context.request.post.assert_awaited_once()
            recovery = list((scraper.CAFES_CSV.parent / "csv_recovery").glob("*.json"))
            self.assertEqual(len(recovery), 1)
            import json
            self.assertEqual(json.loads(recovery[0].read_text(encoding="utf-8"))["reviews"], reviews)

    async def test_backend_skipped_is_reported_separately_without_failure(self):
        with ExitStack() as stack:
            context, cafe = self.setup_pipeline(stack)
            context.request.post.return_value.json = AsyncMock(return_value={"reviews_inserted": 0, "reviews_skipped": 30})
            reviews = [{"review_id": f"r{i}", "cafe_id": cafe["cafe_id"], "text": "Enak"} for i in range(30)]
            stack.enter_context(patch.object(scraper, "scrape_reviews", AsyncMock(return_value=reviews)))
            with self.assertLogs("cafind.scraper", level="INFO") as logs:
                await scraper.run_scraper(["one"], 1)
            self.assertIn("Backend skipped   : 30", "\n".join(logs.output))
            self.assertIn("CSV reviews added : 30", "\n".join(logs.output))

    def setup_pipeline(self, stack):
        temp = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(patch.object(scraper, "CAFES_CSV", Path(temp) / "cafes.csv"))
        stack.enter_context(patch.object(scraper, "REVIEWS_CSV", Path(temp) / "reviews.csv"))
        page = MagicMock()
        page.is_closed.return_value = False
        page.wait_for_timeout = AsyncMock()
        context = MagicMock()
        context.pages = [page]
        context.close = AsyncMock()
        context.request.post = AsyncMock()
        response = context.request.post.return_value
        response.ok = True
        response.json = AsyncMock(return_value={"message": "ok"})
        stack.enter_context(patch.object(scraper, "async_playwright"))
        stack.enter_context(patch.object(scraper, "launch_browser", AsyncMock(return_value=context)))
        stack.enter_context(patch.object(scraper.asyncio, "sleep", AsyncMock()))
        stack.enter_context(patch.object(scraper, "search_cafes", AsyncMock()))
        stack.enter_context(patch.object(scraper, "scroll_results_panel", AsyncMock()))
        stack.enter_context(patch.object(scraper, "collect_cafe_links", AsyncMock(return_value=["url"])))
        cafe = dict.fromkeys(scraper.CAFE_FIELDS, "")
        cafe.update(cafe_id="new-id", name="Cafe", address="A", rating=4, latitude=-7.9, longitude=112.6)
        stack.enter_context(patch.object(scraper, "scrape_cafe_details", AsyncMock(side_effect=lambda *args: cafe.copy())))
        return context, cafe

    async def test_login_only_does_not_create_csv_or_contact_backend(self):
        with ExitStack() as stack:
            context, _ = self.setup_pipeline(stack)
            login = stack.enter_context(patch.object(scraper, "login_manually", AsyncMock()))
            await scraper.run_scraper(["test"], 1, login_only=True)
            login.assert_awaited_once()
            self.assertFalse(scraper.CAFES_CSV.exists())
            self.assertFalse(scraper.REVIEWS_CSV.exists())
            context.request.post.assert_not_awaited()
            context.close.assert_awaited_once()

    async def test_staging_direct_place_never_contacts_backend_or_writes_main_csv(self):
        with ExitStack() as stack:
            context, cafe = self.setup_pipeline(stack)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            reviews = [dict(review_id="r1", cafe_id=cafe["cafe_id"], text="Kopi enak")]
            stack.enter_context(patch.object(scraper, "scrape_reviews", AsyncMock(return_value=reviews)))
            await scraper.run_scraper(["one"], 1, output_dir=output, no_backend=True, place_url="https://www.google.com/maps/place/Cafe?hl=en")
            context.request.post.assert_not_awaited()
            scraper.search_cafes.assert_not_awaited()
            self.assertFalse(scraper.CAFES_CSV.exists())
            self.assertFalse(scraper.REVIEWS_CSV.exists())
            self.assertTrue((output / "cafes.csv").exists())
            with (output / "reviews.csv").open(newline="", encoding="utf-8") as file:
                self.assertEqual(list(csv.DictReader(file))[0]["text"], "Kopi enak")

    async def test_regular_chrome_login_never_starts_playwright_or_pipeline(self):
        with ExitStack() as stack:
            context, _ = self.setup_pipeline(stack)
            login = stack.enter_context(patch.object(scraper, "login_in_regular_chrome"))
            await scraper.run_scraper(["test"], 1, login_only=True, browser="chrome")
            login.assert_called_once_with(scraper.CHROME_DATA_DIR)
            scraper.async_playwright.assert_not_called()
            scraper.launch_browser.assert_not_awaited()
            context.request.post.assert_not_awaited()
            self.assertFalse(scraper.CAFES_CSV.exists())

    async def test_chrome_scraping_uses_same_profile_as_manual_login(self):
        with ExitStack() as stack:
            self.setup_pipeline(stack)
            await scraper.run_scraper([], 1, browser="chrome")
            self.assertEqual(scraper.launch_browser.await_args.args[1], scraper.CHROME_DATA_DIR)
            self.assertEqual(scraper.launch_browser.await_args.kwargs["browser"], "chrome")

    async def test_resume_reuses_id_without_duplicate_cafe_and_then_skips(self):
        with ExitStack() as stack:
            context, cafe = self.setup_pipeline(stack)
            scraper.ensure_csv_headers()
            cafe["cafe_id"] = "old-id"
            scraper.append_cafe(cafe)
            async def get_reviews(page, cafe_id, **kwargs):
                return [{"review_id": "r1", "cafe_id": cafe_id, "text": "Enak"}]
            scrape = stack.enter_context(patch.object(scraper, "scrape_reviews", AsyncMock(side_effect=get_reviews)))
            await scraper.run_scraper(["one", "two"], 1, resume=True)
            scrape.assert_awaited_once()
            self.assertEqual(scrape.await_args.args[1], "old-id")
            with scraper.CAFES_CSV.open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)
            with scraper.REVIEWS_CSV.open(newline="", encoding="utf-8") as file:
                self.assertEqual(list(csv.DictReader(file))[0]["cafe_id"], "old-id")
            await scraper.run_scraper(["one"], 1, resume=True)
            scrape.assert_awaited_once()
            self.assertEqual(context.request.post.await_count, 2)
            self.assertEqual(context.request.post.await_args.kwargs["data"]["reviews"], [])

    async def test_resume_refreshes_price_hours_in_csv_and_payload_without_rescraping_reviews(self):
        with ExitStack() as stack:
            context, cafe = self.setup_pipeline(stack)
            scraper.ensure_csv_headers()
            cafe["cafe_id"] = "old-id"
            scraper.append_cafe(cafe)
            scraper.append_reviews([dict(review_id="r1", cafe_id="old-id", text="Enak")])
            cafe.update(price_range="Rp 25–50K", price_min=25000.0, price_max=50000.0, opening_hours="Senin: 07.00–23.00")
            reviews = stack.enter_context(patch.object(scraper, "scrape_reviews", AsyncMock()))
            await scraper.run_scraper(["one"], 1, resume=True)
            reviews.assert_not_awaited()
            payload = context.request.post.await_args.kwargs["data"]
            self.assertEqual(payload["cafe_id"], "old-id")
            self.assertEqual(payload["price_min"], 25000.0)
            self.assertEqual(payload["price_max"], 50000.0)
            self.assertEqual(payload["opening_hours"], "Senin: 07.00–23.00")
            with scraper.CAFES_CSV.open(newline="", encoding="utf-8") as file:
                saved = list(csv.DictReader(file))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["price_min"], "25000.0")
            self.assertEqual(saved[0]["opening_hours"], payload["opening_hours"])
            with scraper.REVIEWS_CSV.open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)

    async def test_access_gate_aborts_without_saving_empty_success(self):
        with ExitStack() as stack:
            context, _ = self.setup_pipeline(stack)
            scrape = stack.enter_context(patch.object(scraper, "scrape_reviews", AsyncMock(side_effect=ReviewAccessError("login"))))
            with self.assertRaises(ReviewAccessError):
                await scraper.run_scraper(["one", "two"], 1)
            scrape.assert_awaited_once()
            context.request.post.assert_not_awaited()
            context.close.assert_awaited_once()
            with scraper.CAFES_CSV.open(newline="", encoding="utf-8") as file:
                self.assertEqual(list(csv.DictReader(file)), [])

    async def test_failed_details_do_not_report_an_empty_success(self):
        with ExitStack() as stack:
            context, _ = self.setup_pipeline(stack)
            stack.enter_context(patch.object(scraper, "scrape_cafe_details", AsyncMock(return_value=None)))
            reviews = stack.enter_context(patch.object(scraper, "scrape_reviews", AsyncMock()))
            with self.assertRaisesRegex(RuntimeError, "Tidak ada kafe yang berhasil"):
                await scraper.run_scraper(["one"], 1)
            reviews.assert_not_awaited()
            context.request.post.assert_not_awaited()
            context.close.assert_awaited_once()

    async def test_review_panel_failure_is_not_posted_as_zero_review_success(self):
        with ExitStack() as stack:
            context, _ = self.setup_pipeline(stack)
            stack.enter_context(patch.object(scraper, "scrape_reviews", AsyncMock(side_effect=ReviewPanelError("panel failed"))))
            diagnostics = stack.enter_context(patch.object(scraper, "save_failure_diagnostics", AsyncMock()))
            with self.assertRaisesRegex(RuntimeError, "Tidak ada kafe yang berhasil"):
                await scraper.run_scraper(["one"], 1)
            context.request.post.assert_not_awaited()
            diagnostics.assert_awaited_once()
            with scraper.CAFES_CSV.open(newline="", encoding="utf-8") as file:
                self.assertEqual(list(csv.DictReader(file)), [])

    async def test_filtered_place_is_skipped_without_reporting_failure(self):
        with ExitStack() as stack:
            context, _ = self.setup_pipeline(stack)
            stack.enter_context(patch.object(scraper, "scrape_cafe_details", AsyncMock(side_effect=scraper.CafeOutsideTarget("outside Malang"))))
            reviews = stack.enter_context(patch.object(scraper, "scrape_reviews", AsyncMock()))
            await scraper.run_scraper(["one"], 1)
            reviews.assert_not_awaited()
            context.request.post.assert_not_awaited()

    async def test_closed_browser_aborts_remaining_queries(self):
        with ExitStack() as stack:
            context, _ = self.setup_pipeline(stack)
            context.pages[0].is_closed.return_value = True
            scrape = stack.enter_context(patch.object(scraper, "scrape_reviews", AsyncMock(side_effect=RuntimeError("Target closed"))))
            with self.assertRaisesRegex(RuntimeError, "Browser ditutup"):
                await scraper.run_scraper(["one", "two"], 1)
            scrape.assert_awaited_once()
            context.request.post.assert_not_awaited()
            context.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
