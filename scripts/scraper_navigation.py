"""Wait for place content, with bounded retries and local failure diagnostics."""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qsl, urlencode, urlunsplit

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

if __package__:
    from .scraper_reviews import ReviewAccessError, dismiss_popups
else:
    from scraper_reviews import ReviewAccessError, dismiss_popups

log = logging.getLogger("cafind.scraper")
PLACE_HEADING = 'h1.DUwDvf, [role="main"] h1'


def indonesian_maps_url(url: str) -> str:
    """Keep account/place parameters, overriding any copied hl=en setting."""
    parts = urlsplit(url)
    if parts.hostname not in {"google.com", "www.google.com", "google.co.id", "www.google.co.id"} or not parts.path.startswith("/maps"):
        return url
    params = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key != "hl"]
    params.append(("hl", "id"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


def same_place(current_url: str, target_url: str) -> bool:
    """Only recover a navigation timeout if the current document is the requested place."""
    current, target = unquote(current_url), unquote(target_url)
    current_id = re.search(r"!1s([^!/?]+)", current)
    target_id = re.search(r"!1s([^!/?]+)", target)
    if current_id and target_id:
        return current_id.group(1) == target_id.group(1)
    return current == target


async def open_cafe_page(page: Page, url: str, *, attempts: int = 2) -> None:
    """Wait for document commit and a visible place heading instead of DOMContentLoaded.

    A timed-out document event alone is not proof that Maps failed to render.
    Never reuse a heading left on the previous cafe's document after a failed goto.
    """
    url = indonesian_maps_url(url)
    for attempt in range(1, attempts + 1):
        try:
            log.info("Opening cafe details (attempt %s/%s)", attempt, attempts)
            try:
                await page.goto(url, wait_until="commit", timeout=30_000)
            except PlaywrightTimeoutError:
                if page.is_closed() or not same_place(page.url, url):
                    raise
                log.warning("Navigation timed out on the requested place URL; checking rendered content.")
            if urlsplit(page.url).hostname == "accounts.google.com":
                raise ReviewAccessError("Sesi Google meminta login kembali. Ulangi --login-only dengan --browser yang sama.")
            await dismiss_popups(page)
            await page.locator(PLACE_HEADING).first.wait_for(state="visible", timeout=20_000)
            await dismiss_popups(page)
            return
        except ReviewAccessError:
            raise
        except PlaywrightTimeoutError as exc:
            if page.is_closed() or attempt == attempts:
                raise
            log.warning("Cafe details not ready on attempt %s: %s. Retrying once.", attempt, exc)
            await page.wait_for_timeout(1000)


async def save_failure_diagnostics(page: Page, directory: Path, *, url: str, stage: str, error: Exception) -> None:
    """Save a local screenshot and minimal context; never dump cookies or storage state."""
    if page.is_closed():
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        prefix = directory / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
        details = {
            "stage": stage, "requested_url": url, "current_url": page.url,
            "error": str(error),
        }
        prefix.with_suffix(".json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
        await page.screenshot(path=str(prefix.with_suffix(".png")), timeout=5000)
        log.warning("Failure diagnostics saved: %s (.json and .png)", prefix)
    except Exception as exc:
        log.warning("Could not save failure diagnostics: %s", exc)
