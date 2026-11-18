"""Prepare a reviewable text correction plan and PostgreSQL scripts; never connect to the DB."""

import argparse
import csv
import json
import shutil
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def review_key(row: dict) -> tuple[str, float]:
    return row["author_name"].strip(), float(row["rating"])


def propose_changes(old: list[dict], fresh: list[dict]) -> tuple[list[dict], list[dict]]:
    """Author/rating only proposes a match. Ambiguous matches are left for manual review."""
    old_by_key, new_by_key = defaultdict(list), defaultdict(list)
    for row in old:
        old_by_key[review_key(row)].append(row)
    for row in fresh:
        new_by_key[review_key(row)].append(row)
    changes, unmatched = [], []
    for key, candidates in new_by_key.items():
        previous = old_by_key.get(key, [])
        if len(previous) != 1 or len(candidates) != 1:
            unmatched.append({"author_name": key[0], "reason": "missing or ambiguous author/rating match"})
            continue
        before, after = previous[0], candidates[0]
        if before["text"] != after["text"]:
            changes.append({
                "reference_review_id": before["review_id"], "db_review_id": before.get("db_review_id", ""), "author_name": before["author_name"],
                "rating": before["rating"], "old_text": before["text"], "new_text": after["text"],
            })
    return changes, unmatched


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_text(value: str) -> str:
    """Preserve UTF-8 bytes, including LF/CRLF and emoji, across Windows SQL editors."""
    return f"convert_from(decode('{value.encode('utf-8').hex()}', 'hex'), 'UTF8')"


def build_sql(cafe_id: str, cafe_name: str, changes: list[dict], suffix: str) -> tuple[str, str]:
    """Guard exact old text, author, rating, cafe ID/name, and expected row count before writes."""
    cafe_id = str(uuid.UUID(cafe_id))
    if not changes:
        raise ValueError("No proposed text changes; no SQL should be generated")
    values = ",\n".join(
        "(" + ", ".join([sql_literal(str(uuid.UUID(c["db_review_id"]))) + "::uuid" if c.get("db_review_id") else "NULL", sql_text(c["author_name"]), str(float(c["rating"])), sql_text(c["old_text"]), sql_text(c["new_text"])]) + ")"
        for c in changes
    )
    setup = f"""-- Review proposed_changes.csv first. No review rows are deleted.
BEGIN;
SET LOCAL standard_conforming_strings = on;
CREATE TEMP TABLE repair_context (cafe_id uuid, cafe_name text) ON COMMIT DROP;
INSERT INTO repair_context VALUES ({sql_literal(cafe_id)}::uuid, {sql_text(cafe_name)});
CREATE TEMP TABLE language_fix (db_review_id uuid, author_name text, rating double precision, old_text text, new_text text) ON COMMIT DROP;
INSERT INTO language_fix VALUES
{values};
DO $guard$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM cafes c JOIN repair_context x ON c.id = x.cafe_id AND c.name = x.cafe_name) THEN
        RAISE EXCEPTION 'Cafe ID/name does not match. Nothing was changed.';
    END IF;
END; $guard$;
SELECT r.id FROM reviews r JOIN repair_context x ON r.cafe_id = x.cafe_id FOR UPDATE OF r;
CREATE TEMP TABLE repair_targets ON COMMIT DROP AS
SELECT r.id, r.author_name, f.old_text, f.new_text
FROM reviews r JOIN repair_context x ON r.cafe_id = x.cafe_id
JOIN language_fix f ON r.author_name = f.author_name AND r.rating = f.rating
    AND COALESCE(r.raw_text, r.text) = f.old_text AND (f.db_review_id IS NULL OR r.id = f.db_review_id);
DO $guard$
DECLARE matched_count integer; unique_count integer;
BEGIN
    SELECT count(*), count(DISTINCT id) INTO matched_count, unique_count FROM repair_targets;
    IF matched_count <> {len(changes)} OR unique_count <> {len(changes)} THEN
        RAISE EXCEPTION 'Expected % exact review matches; found % (% distinct). ROLLBACK and inspect.', {len(changes)}, matched_count, unique_count;
    END IF;
END; $guard$;
"""
    preview = setup + "SELECT * FROM repair_targets ORDER BY author_name;\nROLLBACK;\n"
    apply = setup + f"""
-- Durable backups are committed together with the correction.
CREATE TABLE review_language_backup_{suffix} AS
SELECT r.* FROM reviews r JOIN repair_targets t ON r.id = t.id;
CREATE TABLE aspect_language_backup_{suffix} AS
SELECT a.* FROM aspect_sentiments a JOIN repair_targets t ON a.review_id = t.id;
UPDATE reviews r SET text = t.new_text, raw_text = t.new_text, is_processed = false
FROM repair_targets t WHERE r.id = t.id;
-- Only derived ABSA results of these corrected reviews are invalidated.
DELETE FROM aspect_sentiments a USING repair_targets t WHERE a.review_id = t.id;
SELECT count(*) AS corrected_reviews FROM repair_targets;
COMMIT;
"""
    return preview, apply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-dir", required=True, type=Path)
    parser.add_argument("--db-cafe-id", required=True)
    parser.add_argument("--database-snapshot", type=Path, help="Read an exported DB snapshot instead of assuming the main CSV contains the affected batch")
    parser.add_argument("--raw-dir", type=Path, default=BASE_DIR / "data/raw")
    args = parser.parse_args()
    uuid.UUID(args.db_cafe_id)
    _, cafes = read_csv(args.staged_dir / "cafes.csv")
    if len(cafes) != 1:
        parser.error("Staged data must contain exactly one cafe")
    cafe = cafes[0]
    _, existing_cafes = read_csv(args.raw_dir / "cafes.csv")
    local_ids = {c["cafe_id"] for c in existing_cafes if c["name"].strip().casefold() == cafe["name"].strip().casefold()}
    if len(local_ids) != 1 and not args.database_snapshot:
        parser.error("Local cafe name is missing or ambiguous; inspect cafe identity manually")
    fields, existing_reviews = read_csv(args.raw_dir / "reviews.csv")
    _, fresh = read_csv(args.staged_dir / "reviews.csv")
    old = [r for r in existing_reviews if r["cafe_id"] in local_ids]
    if args.database_snapshot:
        snapshot = json.loads(args.database_snapshot.read_text(encoding="utf-8"))
        original_cafe = snapshot["cafe"]
        if str(original_cafe["id"]) != str(uuid.UUID(args.db_cafe_id)) or original_cafe["name"] != cafe["name"]:
            parser.error("DB cafe ID/name does not match staged cafe; never mix similarly named places")
        if any(abs(float(original_cafe[key]) - float(cafe[key])) > 0.001 for key in ("latitude", "longitude")):
            parser.error("DB and staged cafe coordinates do not match")
        old = [{**r, "review_id": r["id"], "db_review_id": r["id"], "text": r.get("raw_text") or r["text"]} for r in snapshot["reviews"]]
    changes, unmatched = propose_changes(old, fresh)
    if not changes:
        parser.error("No unambiguous changes found; do not change the DB")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output = args.staged_dir / f"repair_plan_{stamp}"
    output.mkdir()
    for name in ("cafes.csv", "reviews.csv"):
        shutil.copy2(args.raw_dir / name, output / f"{name}.before")
    if args.database_snapshot:
        shutil.copy2(args.database_snapshot, output / "database_snapshot.before.json")
    write_csv(output / "proposed_changes.csv", list(changes[0]), changes)
    (output / "unmatched.json").write_text(json.dumps(unmatched, ensure_ascii=False, indent=2), encoding="utf-8")
    preview, apply = build_sql(args.db_cafe_id, cafe["name"], changes, stamp)
    (output / "01_preview.sql").write_text(preview, encoding="utf-8")
    (output / "02_apply.sql").write_text(apply, encoding="utf-8")
    by_key = {(c["author_name"], float(c["rating"]), c["old_text"]): c for c in changes}
    corrected = []
    local_changes = 0
    for row in existing_reviews:
        updated = row.copy()
        change = by_key.get((row["author_name"], float(row["rating"]), row["text"]))
        if row["cafe_id"] in local_ids and change:
            updated["text"] = change["new_text"]
            local_changes += 1
        corrected.append(updated)
    write_csv(output / "reviews.corrected.csv", fields, corrected)
    instructions = f"""# Proposed review language correction

Cafe: {cafe['name']}
DB cafe ID: {args.db_cafe_id}
Proposed changes: {len(changes)}. Unmatched/ambiguous fresh reviews: {len(unmatched)}.
Affected rows in the main local CSV: {local_changes}.

Nothing has been changed in the database or main CSVs.

1. Inspect proposed_changes.csv. Author/rating matching only suggests a correspondence;
   confirm each old/new text is the same review in its original language. If any is
   wrong, do not run the generated apply script; regenerate the plan after resolving it.
2. Run 01_preview.sql in the PostgreSQL client. It displays exact old/new DB matches
   and ends with ROLLBACK. If an error occurs, run ROLLBACK before doing anything else.
3. After approving the complete plan, run 02_apply.sql. It creates permanent backup
   tables, updates only exact matches, preserves review IDs/dates/ratings, resets
   is_processed, and invalidates only the affected ABSA results. It does not delete reviews.
4. ONLY if affected rows in the main local CSV is greater than zero: with scraping
   stopped, after a successful COMMIT, replace data/raw/reviews.csv
   with reviews.corrected.csv ONLY IF the main reviews.csv has not changed since
   reviews.csv.before was created. Keep the backup. If it has changed, regenerate.
5. Rerun the ABSA/embedding pipeline for the corrected data before using derived results.

The database scripts were generated for review; they have not been executed.
"""
    (output / "README.md").write_text(instructions, encoding="utf-8")
    print(json.dumps({"plan": str(output), "proposed_changes": len(changes), "unmatched": len(unmatched), "local_changes": local_changes}, ensure_ascii=True))


if __name__ == "__main__":
    main()
