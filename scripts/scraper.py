
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
    if not await locator.count():
        return default
    try:
        return (await locator.first.inner_text(timeout=3000)).strip()
    except Exception as error:
        from playwright.async_api import TimeoutError as PageTimeout
        if not isinstance(error, PageTimeout):
            raise
        return default

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

def parse_coords(url):
    match = re.search(r'!3d(-?[0-9.]+)!4d(-?[0-9.]+)', url)
    if not match:
        match = re.search(r'@(-?[0-9.]+),(-?[0-9.]+)', url)
    return (float(match[1]), float(match[2])) if match else (None, None)

def parse_rating(text):
    match = re.search(r'([0-5](?:[.,][0-9])?)', text)
    return float(match[1].replace(',', '.')) if match else None

async def extract_cafe(page, query):
    name = await first_text(page.locator('h1'))
    if not name:
        raise ValueError('Cafe name is missing')
    latitude, longitude = parse_coords(page.url)
    rating_text = await first_text(page.locator('div.F7nice'))
    return dict(cafe_id=hashlib.sha256((urlsplit(page.url).path + str(latitude) + str(longitude)).encode()).hexdigest()[:16], name=name,
                address=await first_text(page.locator('[data-item-id="address"]')),
                latitude=latitude, longitude=longitude, rating=parse_rating(rating_text),
                review_count=None, source_url=page.url, source_query=query,
                scraped_at=datetime.now(timezone.utc).isoformat())

def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

async def visible_reviews(page, cafe_id):
    rows = []
    for card in await page.locator('[data-review-id]').all():
        review_id = await card.get_attribute('data-review-id')
        text = await first_text(card.locator('.wiI7pd'))
        if not review_id or not text:
            continue
        stars = card.locator('[aria-label*="star"], [aria-label*="bintang"]').first
        label = await stars.get_attribute('aria-label') if await stars.count() else ''
        rows.append(dict(review_id=review_id, cafe_id=cafe_id, text=text,
                         rating=parse_rating(label or ''),
                         scraped_at=datetime.now(timezone.utc).isoformat()))
    return rows

async def collect_reviews(page, cafe_id, limit):
    tab = page.get_by_role('tab', name=re.compile('ulasan|review', re.I))
    if await tab.count():
        await tab.first.click()
        await page.wait_for_timeout(1200)
    found = {}
    for attempt in range(5):
        before = len(found)
        for review in await visible_reviews(page, cafe_id):
            found.setdefault(review['review_id'], review)
        if len(found) >= limit or (attempt > 0 and len(found) == before):
            break
        cards = page.locator('[data-review-id]')
        if not await cards.count():
            break
        await cards.last.scroll_into_view_if_needed()
        await page.mouse.wheel(0, 700)
        await page.wait_for_timeout(1200)
    return list(found.values())[:limit]

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

            cafes, reviews = [], []
            for link in links:
                try:
                    await page.goto(link, wait_until='domcontentloaded')
                    await page.wait_for_timeout(1000)
                    cafe = await extract_cafe(page, args.query)
                    cafes.append(cafe)
                    reviews.extend(await collect_reviews(page, cafe['cafe_id'], args.reviews_per_cafe))

                except (PageTimeout, ValueError) as error:
                    print(f'Skipping incomplete cafe {link}: {error}')

            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_csv(args.output_dir / 'cafes.csv', CAFE_FIELDS, cafes)
            write_csv(args.output_dir / 'reviews.csv', REVIEW_FIELDS, reviews)
            return {'cafes': len(cafes), 'reviews': len(reviews)}

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
