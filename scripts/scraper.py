
"""Single-query scraper preview. Resume, multi-query jobs and backend sync are pending.

Only visible Google Maps results/reviews are collected. CSV files replace the
previous preview run; this is not the full dataset collection pipeline.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit

CAFE_FIELDS = ['cafe_id', 'name', 'address', 'latitude', 'longitude', 'rating',
               'review_count', 'source_url', 'source_query', 'scraped_at']
REVIEW_FIELDS = ['review_id', 'cafe_id', 'text', 'rating', 'scraped_at']

async def first_text(locator, default=''):
    return (await locator.first.inner_text()).strip() if await locator.count() else default

async def collect_links(page, limit):
    links = {}
    for link in await page.locator('a[href*="/maps/place/"]').all():
        href = await link.get_attribute('href')
        if not href:
            continue
        parsed = urlsplit(href)
        if parsed.scheme != 'https' or parsed.hostname != 'www.google.com':
            continue
        key = parsed.path.rstrip('/')
        links.setdefault(key, href)
        if len(links) >= limit:
            break
    return list(links.values())

async def run_scraper(args):
    from playwright.async_api import async_playwright, TimeoutError as PageTimeout
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=args.headless)
        try:
            page = await browser.new_page(locale='id-ID')
            page.set_default_timeout(10000)
            await page.goto('https://www.google.com/maps/search/' + quote(args.query, safe=''),
                            wait_until='domcontentloaded')
            await page.wait_for_timeout(1500)
            links = await collect_links(page, args.max_cafes)

            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / 'cafe_links.json').write_text(
                json.dumps(links, ensure_ascii=False, indent=2), encoding='utf-8')
            return links

        finally:
            await browser.close()

def main():
    parser = argparse.ArgumentParser(description='Cafind single-query scraper preview')
    parser.add_argument('--query', default='cafe Kota Malang')
    parser.add_argument('--max-cafes', type=int, default=5)
    parser.add_argument('--reviews-per-cafe', type=int, default=5)
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw/preview'))
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.max_cafes < 1 or args.reviews_per_cafe < 1:
        parser.error('Cafe and review limits must be positive')
    print(json.dumps(asyncio.run(run_scraper(args)), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
