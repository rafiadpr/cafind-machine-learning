"""Read cafe prices and the weekly schedule from the Overview panel."""

import logging
import re

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

if __package__:
    from .scraper_parsers import parse_price_range
    from .scraper_reviews import ReviewAccessError, raise_if_review_access_blocked
else:
    from scraper_parsers import parse_price_range
    from scraper_reviews import ReviewAccessError, raise_if_review_access_blocked

log = logging.getLogger("cafind.scraper")
HOURS_CONTROL = '[role="button"][jsaction*="pane.openhours"], [data-item-id="oh"]'
DAY = re.compile(r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu)\b", re.I)


def clean_text(text: str) -> str:
    return " ".join(re.sub(r"[\ue000-\uf8ff]", "", text).split())


async def read_schedule(page: Page) -> str | None:
    """Read day/time cells, ignoring copy buttons and unrelated price/rating tables."""
    for table in await page.locator('table:visible').all():
        rows = []
        for row in await table.locator('tr').all():
            cells = row.locator('td, th')
            if await cells.count() < 2:
                continue
            day = clean_text(await cells.nth(0).inner_text())
            if not DAY.match(day):
                continue
            hours = clean_text(await cells.nth(1).inner_text())
            if hours:
                rows.append(f"{day}: {hours}")
        if rows:
            return "; ".join(dict.fromkeys(rows))
    return None


async def extract_overview_details(page: Page) -> dict:
    """Wait for overview controls, prefer price aria labels, then expand weekly hours."""
    overview = page.get_by_role('tab', name=re.compile(r'^(Overview|Ringkasan)$', re.I)).first
    if await overview.is_visible() and await overview.get_attribute('aria-selected') != 'true':
        await overview.click(timeout=3000)
    scope = page.locator('[role="main"]:visible').first
    if not await scope.count():
        scope = page.locator('body')

    # The heading can render before optional metadata controls are hydrated.
    price_selector = (
        '[role="button"][aria-label*="Price range" i], '
        '[role="button"][aria-label*="Rentang harga" i], '
        '[role="img"][aria-label^="Rp" i], [role="img"][aria-label^="IDR" i]'
    )
    price = (None, None, None)
    for _ in range(12):
        await raise_if_review_access_blocked(page, include_limited=False)
        for element in await scope.locator(price_selector).all():
            if not await element.is_visible():
                continue
            price = parse_price_range(await element.get_attribute('aria-label') or await element.inner_text())
            if price[0]:
                break
        if price[0] and await page.locator(HOURS_CONTROL).count():
            break
        await page.wait_for_timeout(250)

    if not price[0]:
        # Avoid extracting food prices mentioned by reviewers or histogram bands.
        text = await scope.evaluate("""root => {
            const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
            const texts = []; let node;
            while (node = walker.nextNode()) {
                if (!node.parentElement.closest('[data-review-id], table') && node.textContent.trim())
                    texts.push(node.textContent.trim());
            }
            return texts.join(' ');
        }""")
        price = parse_price_range(text)

    hours = None
    control = page.locator(f':is({HOURS_CONTROL}):visible').first
    if await control.count():
        status = clean_text(await control.inner_text())
        was_expanded = await control.get_attribute('aria-expanded') == 'true'
        opened = False
        try:
            if not was_expanded:
                await control.click(timeout=3000)
                opened = True
            for _ in range(12):
                await raise_if_review_access_blocked(page, include_limited=False)
                hours = await read_schedule(page)
                if hours:
                    break
                await page.wait_for_timeout(250)
        except ReviewAccessError:
            raise
        except PlaywrightTimeoutError as exc:
            log.warning("Could not expand weekly hours: %s", exc)
        finally:
            if opened and await control.get_attribute('aria-expanded') == 'true':
                await control.click(timeout=2000)
        hours = hours or status or None
    else:
        log.warning("Opening-hours control not available in Overview.")

    if not price[0]:
        log.warning("Price range not available in Overview; numeric prices remain empty.")
    if hours and not DAY.match(hours):
        log.warning("Only current opening status is available; weekly schedule could not be read.")
    return dict(price_range=price[0], price_min=price[1], price_max=price[2], opening_hours=hours)
