"""Google Maps review navigation, access checks, scrolling, and extraction."""

import logging
import random
import re
import uuid
from datetime import datetime, timezone

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

if __package__:
    from .scraper_parsers import parse_review_count
else:
    from scraper_parsers import parse_review_count

log = logging.getLogger("cafind.scraper")


class ReviewPanelError(RuntimeError):
    """The reviews UI could not be opened or parsed; this is not an empty success."""


class ReviewAccessError(RuntimeError):
    """Maps requires authentication or is serving a limited view."""


MORE_REVIEWS_SELECTOR = ', '.join([
    'button:has-text("Lihat ulasan lainnya")',
    'button:has-text("Lihat semua ulasan")',
    'button:has-text("More reviews")',
    'button:has-text("See all reviews")',
    'button[aria-label*="ulasan lainnya" i]',
    'button[aria-label*="semua ulasan" i]',
    'button[aria-label*="more reviews" i]',
    'button[aria-label*="see all reviews" i]',
    'button[aria-label*="view all reviews" i]',
    'button[jsaction*="moreReviews"]',
])
REVIEW_SELECTOR = '[data-review-id]:visible:not([data-review-id] [data-review-id])'
REVIEW_SEARCH_SELECTOR = 'input[aria-label*="Search reviews" i], input[aria-label*="Cari ulasan" i], button[aria-label*="Search reviews" i], button[aria-label*="Telusuri ulasan" i]'
SORT_SELECTOR = 'button[aria-label*="Urutkan" i], button[aria-label*="Sort" i], [data-value="Sort"]'


async def raise_if_review_access_blocked(page: Page, *, include_limited: bool = True) -> None:
    """Inspect visible access messages before any generic popup dismissal."""
    dialogs = page.locator('[role="dialog"], [aria-modal="true"]').filter(
        has_text=re.compile(r"(?:login|sign\s*in|masuk).*?(?:Google\s*Maps|ulasan|reviews|photos)", re.I | re.S)
    )
    for dialog in await dialogs.all():
        if await dialog.is_visible():
            raise ReviewAccessError(
                "Google Maps meminta login untuk mengakses ulasan. "
                "Jalankan --login-only dengan pilihan --browser yang sama, lalu ulangi dengan --resume. "
                "Jika login Chromium ditolak, coba --login-only --browser chrome; "
                "gunakan --browser chrome juga saat scraping."
            )
    if include_limited:
        messages = page.get_by_text(re.compile(
            r"(?:tampilan|akses) terbatas.{0,40}Google Maps|limited (?:view|access).{0,40}Google Maps",
            re.I | re.S,
        ))
        for message in await messages.all():
            if await message.is_visible():
                raise ReviewAccessError(
                    "Google Maps menampilkan akses terbatas. Jalankan --login-only dan "
                    "pastikan seluruh ulasan bisa dibuka manual sebelum mencoba --resume."
                )


async def dismiss_popups(page: Page) -> bool:
    """Close ordinary overlays, preserving review panels and authentication gates."""
    await raise_if_review_access_blocked(page, include_limited=False)
    for dialog in await page.locator('[role="dialog"], [aria-modal="true"]').all():
        if not await dialog.is_visible():
            continue
        # Reviews may be rendered as a dialog. Never close one or press Escape
        # indiscriminately: that would undo opening the full reviews panel.
        if (await dialog.locator('[data-review-id]').count()
                or await dialog.locator(SORT_SELECTOR).count()
                or re.search(r"reviews|ulasan", await dialog.get_attribute("aria-label") or "", re.I)):
            continue
        close = dialog.get_by_role("button", name=re.compile(
            r"^(?:Tutup|Dismiss|Close|Batalkan|Cancel|Tidak, terima kasih|No thanks)(?: dialog)?$", re.I,
        ))
        for button in await close.all():
            if await button.is_visible():
                log.info("Dismissing ordinary modal")
                await button.click(timeout=1000)
                return True
    return False


async def review_panel_ready(page: Page, *, initial_ids: set[str] | None = None) -> bool:
    """Use visible review cards and panel controls, not a background button's disappearance."""
    cards = page.locator(REVIEW_SELECTOR)
    if not await cards.count():
        return False
    if any([await control.is_visible() for control in await page.locator(f"{SORT_SELECTOR}, {REVIEW_SEARCH_SELECTOR}").all()]):
        return True
    for dialog in await page.locator('[role="dialog"]:visible, [aria-modal="true"]:visible').all():
        if await dialog.locator(REVIEW_SELECTOR).count():
            return True
    if initial_ids is not None:
        current_ids = {await card.get_attribute('data-review-id') for card in await cards.all()}
        if current_ids - initial_ids:
            return True
    return not any([await b.is_visible() for b in await page.locator(MORE_REVIEWS_SELECTOR).all()])


async def review_counts(page: Page) -> tuple[int, int]:
    """Count unique review cards, not nested buttons/containers or translated text nodes."""
    counts = await page.locator(REVIEW_SELECTOR).evaluate_all("""cards => {
        const all = new Set(), withText = new Set();
        for (const card of cards) {
            const id = card.getAttribute('data-review-id');
            all.add(id);
            if ([...card.querySelectorAll('.wiI7pd')].some(el => el.textContent.trim())) withText.add(id);
        }
        return [all.size, withText.size];
    }""")
    return tuple(counts)


async def original_review_text(card) -> str:
    """Ask Maps for the author's original text before reading the visible review body."""
    original = card.get_by_role("button", name=re.compile(
        r"^(?:See original|Show original|View original|Lihat (?:yang )?asli(?:nya)?|Lihat versi asli|Tampilkan (?:teks )?asli)(?:\s*\(.*\))?$", re.I,
    )).or_(card.locator('button[jsaction*="showReviewInOriginal"], button[jsaction*="review.original"]'))
    for button in await original.all():
        if await button.is_visible():
            await button.click(timeout=2000)
            break
    # Hidden copies can contain translations or truncated text.
    text = card.locator('.wiI7pd:visible').first
    return (await text.inner_text()).strip() if await text.count() else ""


async def open_more_reviews_if_needed(page: Page) -> bool:
    """Reuse an already open panel, otherwise click once and wait for actual review content."""
    await raise_if_review_access_blocked(page)
    if await review_panel_ready(page):
        return True
    initial_ids = {
        await card.get_attribute('data-review-id') for card in await page.locator(REVIEW_SELECTOR).all()
    }
    for button in await page.locator(MORE_REVIEWS_SELECTOR).all():
        if await button.is_visible():
            log.info("Opening full reviews panel...")
            await button.scroll_into_view_if_needed(timeout=2000)
            await button.click(timeout=3000)
            break

    for _ in range(40):
        await raise_if_review_access_blocked(page)
        # Do not dismiss dialogs here: the newly opened review list may itself
        # be a dialog. The sign-in gate is handled explicitly above.
        if await review_panel_ready(page, initial_ids=initial_ids):
            return True
        await page.wait_for_timeout(250)
    return False


async def scrape_reviews(page: Page, cafe_id: str, max_reviews: int = 20) -> list[dict]:
    """Scrape the reviews tab of the currently open cafe page.
    
    Navigates to the reviews section, scrolls the review panel until `max_reviews`
    reviews with text are loaded or no more load, clicks 'More' on long reviews,
    and returns parsed reviews with deduplication.
    """
    reviews: list[dict] = []

    try:
        # Ordinary overlays can be dismissed; authentication gates propagate.
        await dismiss_popups(page)

        # ── 1. Open reviews tab / panel ──
        reviews_tab = page.get_by_role("tab", name=re.compile(r"ulasan|review", re.I)).first
        if await reviews_tab.count() == 0:
            reviews_tab = page.locator('button[role="tab"]:has-text("Ulasan"), button[role="tab"]:has-text("Reviews")').first
        if await reviews_tab.count() == 0:
            reviews_tab = page.locator('button[jsaction*="pane.rating.moreReviews"], button[jsaction*="pane.reviewChart.moreReviews"]').first

        if await reviews_tab.count() > 0:
            try:
                # Human behavior: mouse jitter and 2–4s pause before clicking reviews tab
                await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                await page.wait_for_timeout(random.uniform(2000, 4000))
                await reviews_tab.click(timeout=3000)
                await page.wait_for_timeout(random.uniform(1000, 2000))
            except Exception as e:
                log.debug(f"Could not click reviews tab: {e}")

        # Dismiss modal if triggered by tab switch
        await dismiss_popups(page)

        if not await open_more_reviews_if_needed(page):
            raise ReviewPanelError(f"Panel ulasan tidak siap untuk cafe {cafe_id}")

        # ── 2. Sort by newest (optional, best-effort) ──
        try:
            sort_btn = page.locator(SORT_SELECTOR).first
            if await sort_btn.is_visible(timeout=1500):
                await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                await page.wait_for_timeout(random.uniform(500, 1000))
                await sort_btn.click(timeout=1500)
                await page.wait_for_timeout(random.uniform(500, 1000))
                newest_opt = page.get_by_role("menuitemradio", name=re.compile(r"newest|terbaru", re.I)).first
                if await newest_opt.is_visible(timeout=1500):
                    await newest_opt.click(timeout=1500)
                    await page.wait_for_timeout(random.uniform(1000, 2000))
        except Exception:
            pass

        # ── 3. Dynamic scroll loop for reviews panel ──
        scroll_attempts = 0
        max_scrolls = max(20, (max_reviews // 5) * 4 + 10)
        stuck_count = 0
        prev_card_count = 0

        while scroll_attempts < max_scrolls:
            scroll_attempts += 1

            await raise_if_review_access_blocked(page)
            await dismiss_popups(page)

            cur_cards, cur_text_cards = await review_counts(page)

            # Break if we already have enough review cards with text
            if cur_text_cards >= max_reviews:
                break

            if cur_cards == prev_card_count:
                stuck_count += 1
                if stuck_count >= 6:
                    break
            else:
                stuck_count = 0
            prev_card_count = cur_cards

            # 1. Hover on last review card to position pointer inside reviews container
            last_card = page.locator(REVIEW_SELECTOR).last
            if await last_card.count() > 0:
                try:
                    await last_card.hover(timeout=1000)
                except Exception:
                    pass

            # 2. Gradual mouse wheel scrolling (no instant scrollTo)
            for _ in range(random.randint(2, 4)):
                await page.mouse.wheel(0, random.randint(300, 600))
                await page.wait_for_timeout(random.uniform(120, 250))

            # 3. Smooth fallback scroll on the reviews scrollable container
            if await last_card.count():
                await last_card.evaluate("""rev => {
                    let el = rev.parentElement;
                    while (el) {
                        if (el.scrollHeight > el.clientHeight && el.clientHeight > 0) {
                            el.scrollBy({top: 400, behavior: 'smooth'});
                            break;
                        }
                        el = el.parentElement;
                    }
                }""")

            # Jeda antar scroll pada container ulasan: 1.5–3 detik (simulasikan pembacaan bertahap)
            await page.wait_for_timeout(random.uniform(1500, 3000))


        # ── 4. Expand truncated reviews ("Lainnya" / "Selengkapnya" / "More") ──
        await dismiss_popups(page)
        cards = page.locator(REVIEW_SELECTOR)
        more_buttons = await cards.locator('button.w8nwRe').or_(
            cards.get_by_role("button", name=re.compile(r"^(?:Lainnya|Selengkapnya|More|See more)$", re.I))
        ).all()
        for btn in more_buttons:
            try:
                if await btn.is_visible():
                    # Random mouse movement before clicking expansion buttons
                    await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                    await page.wait_for_timeout(random.uniform(150, 350))
                    await btn.click(timeout=800)
                    await page.wait_for_timeout(random.uniform(100, 250))
            except Exception:
                pass

        # ── 5. Extract reviews with deduplication ──
        seen_rids = set()
        review_els = await page.locator(REVIEW_SELECTOR).all()
        for el in review_els:
            if len(reviews) >= max_reviews:
                break

            try:
                # Deduplicate by Google Maps review ID attribute
                rid_attr = await el.get_attribute("data-review-id")
                if rid_attr:
                    if rid_attr in seen_rids:
                        continue

                # Review text (.wiI7pd)
                text = await original_review_text(el)
                if not text:
                    continue  # Require non-empty text for ABSA NLP training
                if rid_attr:
                    seen_rids.add(rid_attr)

                # Author name (.d4r55)
                author_el = el.locator(".d4r55").first
                author = (await author_el.inner_text()).strip() if await author_el.count() > 0 else "Anonymous"

                # Star rating
                review_rating = None
                star_container = el.locator('[aria-label*="bintang" i], [aria-label*="star" i]').first
                if await star_container.count() > 0:
                    label = await star_container.get_attribute("aria-label") or ""
                    match = re.search(r"(\d+)", label)
                    if match:
                        review_rating = float(match.group(1))

                # Date/time
                date_el = el.locator(".rsqaWe").first
                published_at = (await date_el.inner_text()).strip() if await date_el.count() > 0 else ""

                # Likes
                likes_el = el.locator(".pkWtMe").first
                likes_text = (await likes_el.inner_text()).strip() if await likes_el.count() > 0 else "0"
                likes = parse_review_count(likes_text)

                reviews.append({
                    "review_id": str(uuid.uuid4()),
                    "cafe_id": cafe_id,
                    "author_name": author,
                    "rating": review_rating if review_rating is not None else 5.0,
                    "text": text,
                    "published_at": published_at,
                    "likes": likes,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                })

            except Exception as e:
                log.debug(f"Error parsing review element: {e}")
                continue

    except (ReviewAccessError, ReviewPanelError):
        raise
    except PlaywrightTimeoutError as exc:
        raise ReviewPanelError(f"Timeout memuat ulasan cafe {cafe_id}: {exc}") from exc
    except Exception as e:
        if page.is_closed():
            raise
        raise ReviewPanelError(f"Gagal mengambil ulasan cafe {cafe_id}: {e}") from e

    await raise_if_review_access_blocked(page)
    log.info(f"  → {len(reviews)}/{max_reviews} reviews collected")
    return reviews


