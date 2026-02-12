"""Analyze interaction logs and produce a dynamic scraper wishlist.

This creates `data/scraper_wishlist.json`, which `scraper.py` can ingest
to expand queries and prioritize domains users are rewarding.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DEFAULT_LOG_PATH = Path("data/usage_interactions.csv")
DEFAULT_INVENTORY_PATH = Path("data/scraped_articles.csv")
DEFAULT_WISHLIST_PATH = Path("data/scraper_wishlist.json")

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#'-]+")
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "among",
    "and",
    "are",
    "around",
    "because",
    "been",
    "before",
    "being",
    "best",
    "between",
    "from",
    "have",
    "into",
    "just",
    "more",
    "most",
    "much",
    "news",
    "over",
    "says",
    "saying",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "today",
    "under",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
    "like",
    "news",
    "update",
    "updates",
    "announced",
    "inside",
    "movement",
    "frenzy",
    "patch",
    "notes",
}

# Domain -> feed map for "winner multiplier" promotion.
KNOWN_DOMAIN_FEEDS = {
    "theverge.com": "https://www.theverge.com/rss/index.xml",
    "techcrunch.com": "https://techcrunch.com/feed/",
    "wired.com": "https://www.wired.com/feed/rss",
    "arstechnica.com": "https://feeds.arstechnica.com/arstechnica/index",
    "engadget.com": "https://www.engadget.com/rss.xml",
    "zdnet.com": "https://www.zdnet.com/news/rss.xml",
    "cnet.com": "https://www.cnet.com/rss/news/",
    "technologyreview.com": "https://www.technologyreview.com/feed/",
    "venturebeat.com": "https://venturebeat.com/category/ai/feed/",
    "androidauthority.com": "https://www.androidauthority.com/feed/",
    "9to5mac.com": "https://9to5mac.com/feed/",
    "macrumors.com": "https://www.macrumors.com/macrumors.xml",
    "ign.com": "https://www.ign.com/rss",
    "polygon.com": "https://www.polygon.com/rss/index.xml",
    "kotaku.com": "https://kotaku.com/rss",
    "eurogamer.net": "https://www.eurogamer.net/feed",
    "pcgamer.com": "https://www.pcgamer.com/rss/",
    "nintendoeverything.com": "https://nintendoeverything.com/feed/",
    "gamespot.com": "https://www.gamespot.com/feeds/mashup/",
    "destructoid.com": "https://www.destructoid.com/feed/",
    "rockpapershotgun.com": "https://www.rockpapershotgun.com/feed",
    "lifehacker.com": "https://lifehacker.com/rss",
    "atlasobscura.com": "https://www.atlasobscura.com/feeds/latest",
    "sciencealert.com": "https://www.sciencealert.com/feed",
    "bbc.com": "https://feeds.bbci.co.uk/news/rss.xml",
    "bbc.co.uk": "https://feeds.bbci.co.uk/news/rss.xml",
    "nytimes.com": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "cnn.com": "http://rss.cnn.com/rss/cnn_topstories.rss",
    "aljazeera.com": "https://www.aljazeera.com/xml/rss/all.xml",
    "goodnewsnetwork.org": "https://www.goodnewsnetwork.org/feed/",
}


def normalize_domain(value: str) -> str:
    domain = str(value or "").strip().lower()
    domain = domain.replace("https://", "").replace("http://", "")
    domain = domain.replace("www.", "")
    if "/" in domain:
        domain = domain.split("/", 1)[0]
    return domain


def extract_terms(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall(str(text).lower()):
        tok = raw.strip("-'")
        if len(tok) < 4:
            continue
        if tok in STOPWORDS:
            continue
        if tok.isdigit():
            continue
        tokens.append(tok)
    return tokens


def terms_from_headline(headline: str) -> set[str]:
    toks = extract_terms(headline)
    terms: set[str] = set(toks)
    for i in range(len(toks) - 1):
        a, b = toks[i], toks[i + 1]
        if a in STOPWORDS or b in STOPWORDS:
            continue
        terms.add(f"{a} {b}")
    return terms


def load_interactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    if "reward" in df.columns:
        df["reward"] = pd.to_numeric(df["reward"], errors="coerce").fillna(0.0)
    if "regret" in df.columns:
        df["regret"] = pd.to_numeric(df["regret"], errors="coerce")
    if "topic_name" not in df.columns:
        df["topic_name"] = ""
    if "headline" not in df.columns:
        df["headline"] = ""
    if "user_type" not in df.columns:
        df["user_type"] = "unknown"
    df["topic_name"] = df["topic_name"].map(normalize_domain)
    return df


def load_inventory(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Headline" not in df.columns:
        return pd.DataFrame()
    return df


def build_inventory_term_counts(inv: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    if inv.empty:
        return counts
    for headline in inv["Headline"].fillna("").astype(str).tolist():
        counts.update(terms_from_headline(headline))
    return counts


def pick_domains(df: pd.DataFrame, max_domains: int, min_impressions: int) -> list[str]:
    if df.empty:
        return []
    grouped = (
        df.groupby("topic_name", dropna=False)
        .agg(interactions=("reward", "size"), avg_reward=("reward", "mean"))
        .reset_index()
    )
    grouped = grouped[grouped["topic_name"] != ""]
    grouped = grouped[grouped["interactions"] >= min_impressions]
    if grouped.empty:
        return []
    grouped["score"] = grouped["avg_reward"] * grouped["interactions"].pow(0.5)
    grouped = grouped.sort_values("score", ascending=False)
    return grouped["topic_name"].head(max_domains).tolist()


def pick_queries(
    df: pd.DataFrame,
    inv_term_counts: Counter[str],
    max_queries: int,
    positive_threshold: float,
) -> list[str]:
    if df.empty:
        return []

    hit_df = df[df["reward"] >= positive_threshold].copy()
    if hit_df.empty:
        hit_df = df[df["reward"] > 0.0].copy()
    if hit_df.empty:
        return []

    # Boost terms that appear in high-reward interactions for unhappy cohorts.
    user_stats = (
        df.groupby("user_type")
        .agg(interactions=("reward", "size"), avg_reward=("reward", "mean"))
        .reset_index()
    )
    starving_users = set(
        user_stats[
            (user_stats["interactions"] >= 20) & (user_stats["avg_reward"] < 2.5)
        ]["user_type"].tolist()
    )

    term_hits: Counter[str] = Counter()
    term_user_bonus: Counter[str] = Counter()
    for row in hit_df.itertuples(index=False):
        terms = terms_from_headline(getattr(row, "headline", ""))
        term_hits.update(terms)
        if getattr(row, "user_type", "") in starving_users:
            term_user_bonus.update(terms)

    scored: list[tuple[str, float]] = []
    for term, hits in term_hits.items():
        if hits < 2:
            continue
        parts = term.split()
        if any(p in STOPWORDS for p in parts):
            continue
        inv_count = inv_term_counts.get(term, 0)
        scarcity = 1.0 / (1.0 + (inv_count ** 0.5))
        bonus = 1.0 + min(2.0, term_user_bonus.get(term, 0) / 5.0)
        score = float(hits) * scarcity * bonus
        scored.append((term, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    terms = [t for t, _ in scored[: max_queries * 3]]

    queries: list[str] = []
    seen = set()
    for term in terms:
        q = term if " " in term else term
        if q in seen:
            continue
        seen.add(q)
        queries.append(q)
        if len(queries) >= max_queries:
            break
    return queries


def summarize_users(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    if df.empty:
        return {}
    out: dict[str, dict[str, float | int]] = {}
    grouped = (
        df.groupby("user_type")
        .agg(
            interactions=("reward", "size"),
            avg_reward=("reward", "mean"),
            avg_regret=("regret", "mean"),
        )
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        out[str(row.user_type)] = {
            "interactions": int(row.interactions),
            "avg_reward": float(row.avg_reward),
            "avg_regret": float(getattr(row, "avg_regret", float("nan")))
            if pd.notna(getattr(row, "avg_regret", float("nan")))
            else 0.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--inventory-path", default=str(DEFAULT_INVENTORY_PATH))
    parser.add_argument("--wishlist-path", default=str(DEFAULT_WISHLIST_PATH))
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--positive-threshold", type=float, default=7.0)
    parser.add_argument("--max-queries", type=int, default=24)
    parser.add_argument("--max-domains", type=int, default=12)
    parser.add_argument("--min-domain-impressions", type=int, default=6)
    args = parser.parse_args()

    interactions_path = Path(args.interactions_path)
    inventory_path = Path(args.inventory_path)
    wishlist_path = Path(args.wishlist_path)

    interactions = load_interactions(interactions_path)
    inventory = load_inventory(inventory_path)

    if interactions.empty:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "lookback_days": int(args.lookback_days),
            "queries": [],
            "preferred_domains": [],
            "extra_feed_urls": [],
            "user_summary": {},
            "notes": "No interaction rows yet.",
        }
        wishlist_path.parent.mkdir(parents=True, exist_ok=True)
        wishlist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"No interactions found. Wrote empty wishlist: {wishlist_path}")
        return

    if "timestamp_utc" in interactions.columns and interactions["timestamp_utc"].notna().any():
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(args.lookback_days)))
        recent = interactions[interactions["timestamp_utc"] >= cutoff].copy()
        if recent.empty:
            recent = interactions.copy()
    else:
        recent = interactions.copy()

    inv_term_counts = build_inventory_term_counts(inventory)
    preferred_domains = pick_domains(
        recent,
        max_domains=int(args.max_domains),
        min_impressions=int(args.min_domain_impressions),
    )
    queries = pick_queries(
        recent,
        inv_term_counts=inv_term_counts,
        max_queries=int(args.max_queries),
        positive_threshold=float(args.positive_threshold),
    )

    # "Winner Multiplier": promote feeds from top rewarded domains when known.
    extra_feed_urls = [
        KNOWN_DOMAIN_FEEDS[d] for d in preferred_domains if d in KNOWN_DOMAIN_FEEDS
    ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lookback_days": int(args.lookback_days),
        "interaction_rows_total": int(len(interactions)),
        "interaction_rows_analyzed": int(len(recent)),
        "queries": queries,
        "preferred_domains": preferred_domains,
        "extra_feed_urls": list(dict.fromkeys(extra_feed_urls)),
        "user_summary": summarize_users(recent),
    }

    wishlist_path.parent.mkdir(parents=True, exist_ok=True)
    wishlist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Analyzed rows: {len(recent)}")
    print(f"Top domains: {preferred_domains[:5]}")
    print(f"Queries generated: {len(queries)}")
    print(f"Wishlist saved: {wishlist_path}")


if __name__ == "__main__":
    main()
