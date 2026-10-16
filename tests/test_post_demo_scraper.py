import csv
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from playwright.async_api import Error as BrowserError

spec = importlib.util.spec_from_file_location('cafe_scraper', Path(__file__).resolve().parents[1] / 'scripts/scraper.py')
scraper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scraper)

class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_bad_cafe_does_not_stop_the_next_cafe(self):
        async def goto(url, **kwargs):
            if url == 'bad':
                raise BrowserError('Page closed unexpectedly')
        page = SimpleNamespace(goto=goto, wait_for_timeout=AsyncMock(), set_default_timeout=lambda value: None)
        browser = SimpleNamespace(new_page=AsyncMock(return_value=page), close=AsyncMock())
        playwright = SimpleNamespace(chromium=SimpleNamespace(launch=AsyncMock(return_value=browser)))
        manager = AsyncMock()
        manager.__aenter__.return_value = playwright
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(headless=True, query='cafe', max_cafes=2, reviews_per_cafe=2, output_dir=Path(directory))
            cafe = dict(cafe_id='ok', name='Good Cafe')
            with patch('playwright.async_api.async_playwright', return_value=manager), \
                 patch.object(scraper, 'collect_links', AsyncMock(return_value=['bad', 'good'])), \
                 patch.object(scraper, 'extract_cafe', AsyncMock(return_value=cafe)), \
                 patch.object(scraper, 'collect_reviews', AsyncMock(return_value=[])):
                result = await scraper.run_scraper(args)
            self.assertEqual(result['cafes'], 1)
            with (args.output_dir / 'cafes.csv').open() as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]['cafe_id'], 'ok')
            browser.close.assert_awaited_once()
