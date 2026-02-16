#!/usr/bin/env python3
"""
Stage 1: The Morning Harvest (Aggregation)

Build a "flat pool" of K unique stories for the day and persist it as JSON.

Default behavior:
- Reads from data/scraped_articles.csv (which you can refresh via scraper.py / daily_refresh.py)
- Selects the freshest unique links
- Writes data/daily_pool.json

Optional:
- --run-scraper will run scraper.py first (requires network access)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_IN = ROOT / "data" / "scraped_articles.csv"
DEFAULT_OUT = ROOT / "data" / "daily_pool.json"


def _utc_day_str(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).date().isoformat()


def _json_write_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _pick_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    # Keep compatibility with various scrape outputs.
    headline_col = "Headline" if "Headline" in df.columns else "title"
    summary_col = "Summary" if "Summary" in df.columns else "description"
    link_col = "Link" if "Link" in df.columns else "link"
    published_col = "Published" if "Published" in df.columns else "published"
    return headline_col, summary_col, link_col, published_col


def build_daily_pool(
    *,
    in_csv: Path,
    out_json: Path,
    k: int,
    top_n: int,
) -> dict[str, object]:
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_csv}")

    df = pd.read_csv(in_csv)
    headline_col, summary_col, link_col, published_col = _pick_columns(df)

    # Minimal hygiene.
    df[headline_col] = df[headline_col].fillna("").astype(str)
    if summary_col in df.columns:
        df[summary_col] = df[summary_col].fillna("").astype(str)
    df[link_col] = df[link_col].fillna("").astype(str)
    if published_col in df.columns:
        df[published_col] = df[published_col].fillna("").astype(str)
    else:
        df[published_col] = ""

    df = df[df[link_col].str.strip() != ""]
    df = df[df[headline_col].str.strip() != ""]
    df = df.drop_duplicates(subset=[link_col]).reset_index(drop=True)

    # Try to sort by publish time descending; if parse fails, preserve current order.
    published_ts = pd.to_datetime(df[published_col], errors="coerce", utc=True)
    df["_published_ts"] = published_ts
    if df["_published_ts"].notna().any():
        df = df.sort_values(by="_published_ts", ascending=False, kind="mergesort").reset_index(drop=True)

    # Constrain to the freshest window, then take top K.
    top_n = max(k, int(top_n))
    if len(df) > top_n:
        df = df.iloc[:top_n].reset_index(drop=True)

    df = df.iloc[: max(0, int(k))].reset_index(drop=True)

    now = datetime.now(timezone.utc)
    items: list[dict[str, object]] = []
    for _, row in df.iterrows():
        items.append(
            {
                "title": str(row.get(headline_col, "") or "").strip(),
                "summary": str(row.get(summary_col, "") or "").strip() if summary_col in df.columns else "",
                "link": str(row.get(link_col, "") or "").strip(),
                "published": str(row.get(published_col, "") or "").strip(),
            }
        )

    payload: dict[str, object] = {
        "version": 1,
        "generated_at_utc": now.isoformat(),
        "day_utc": _utc_day_str(now),
        "size": len(items),
        "items": items,
    }
    _json_write_atomic(out_json, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default=str(DEFAULT_IN), help="Input CSV (default: data/scraped_articles.csv)")
    ap.add_argument("--out", dest="out_json", default=str(DEFAULT_OUT), help="Output JSON (default: data/daily_pool.json)")
    ap.add_argument("--k", type=int, default=150, help="Pool size")
    ap.add_argument("--top-n", type=int, default=800, help="Freshness window before truncating to K")
    ap.add_argument("--run-scraper", action="store_true", help="Run scraper.py first (requires network)")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_json = Path(args.out_json)

    if args.run_scraper:
        py = sys.executable
        env = os.environ.copy()
        # Keep scraper defaults (it can be heavy); the pool we build is always exactly K.
        cmd = [py, str(ROOT / "scraper.py")]
        print(f"▶ {' '.join(cmd)}")
        subprocess.check_call(cmd, env=env)

    payload = build_daily_pool(in_csv=in_csv, out_json=out_json, k=args.k, top_n=args.top_n)
    print(f"✅ Wrote daily pool: {out_json} items={payload.get('size')}")


if __name__ == "__main__":
    main()
