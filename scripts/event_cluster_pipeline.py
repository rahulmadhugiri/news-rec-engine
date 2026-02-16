#!/usr/bin/env python3
"""
Event-first clustering pipeline.

Method 1: LLM event naming (cheap/fast model) -> group by event label (exact/fuzzy) -> filter General.
Method 2: Entity graph (no LLM) -> connect by 2+ rare headline entities -> connected components.

Stdlib-only. LLM mode requires OPENAI_API_KEY and network access.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import html as _html
import json
import os
import random
import re
import urllib.parse
import urllib.request
import urllib.error
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import blake2b
from typing import Any, Dict, List, Optional, Sequence, Tuple


TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")
TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "how",
    "why",
    "will",
    "with",
    "you",
    "your",
    # boilerplate
    "rss",
    "oc",
    "news",
    "source",
    "font",
    "href",
    "http",
    "https",
    "www",
    "com",
}


def _stable_id(s: str) -> str:
    return blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).hexdigest()


def _clean_text(s: str) -> str:
    s = _html.unescape(s or "")
    s = TAG_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canonicalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    try:
        p = urllib.parse.urlsplit(u)
        return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, "", ""))
    except Exception:
        return u


def tokenize(text: str) -> List[str]:
    t = _clean_text(text).lower()
    toks = TOKEN_RE.findall(t)
    out: List[str] = []
    for w in toks:
        if w in STOPWORDS:
            continue
        if w.isdigit():
            continue
        out.append(w)
    return out


def _parse_dt(s: str) -> Optional[_dt.datetime]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = _dt.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.UTC)
        return dt.astimezone(_dt.UTC)
    except Exception:
        return None


CAT_TECH_RE = re.compile(r"\b(openai|chatgpt|sora|ai|llm|copilot|ios|iphone|android|google|microsoft|nvidia|tesla|spacex|software|chip|compute)\b", re.I)
CAT_BUS_RE = re.compile(r"\b(stock|etf|earnings|inflation|rates|fed|market|crypto|bitcoin|funding|deal|ipo)\b", re.I)
CAT_POL_RE = re.compile(r"\b(election|parliament|congress|senate|president|prime minister|ukraine|gaza|israel|china|russia|nato)\b", re.I)
CAT_ENT_RE = re.compile(r"\b(netflix|disney|hulu|movie|film|tv|trailer|music|album|celebrity|tiktok|youtube)\b", re.I)
CAT_SPO_RE = re.compile(r"\b(nfl|nba|mlb|nhl|olympics|soccer|football|tennis|golf|f1)\b", re.I)
CAT_LIF_RE = re.compile(r"\b(diet|fitness|workout|sleep|recipe|travel|fashion|home|garden|relationships)\b", re.I)
CAT_SCI_RE = re.compile(r"\b(study|research|scientists|clinical|cancer|flu|health|medicine|biology|brain)\b", re.I)


def classify_category(headline: str, summary: str) -> str:
    t = f"{headline} {summary}".lower()
    if CAT_TECH_RE.search(t):
        return "tech"
    if CAT_BUS_RE.search(t):
        return "business"
    if CAT_POL_RE.search(t):
        return "politics"
    if CAT_ENT_RE.search(t):
        return "entertainment"
    if CAT_SPO_RE.search(t):
        return "sports"
    if CAT_LIF_RE.search(t):
        return "lifestyle"
    if CAT_SCI_RE.search(t):
        return "science_health"
    return "other"


@dataclass(frozen=True)
class Item:
    idx: int
    item_id: str
    headline: str
    summary: str
    raw_headline: str
    link: str
    published_utc: str
    topic_name: str


def load_items(csv_path: str) -> List[Item]:
    items: List[Item] = []
    seen_url: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for idx, row in enumerate(r):
            def g(k: str) -> str:
                v = row.get(k)
                return v.strip() if isinstance(v, str) else ""

            raw_h = g("Headline")
            raw_s = g("Summary")
            if "<a " in raw_h.lower():
                continue
            h = _clean_text(raw_h)
            s = _clean_text(raw_s)
            link = _canonicalize_url(g("Link"))
            if link:
                if link in seen_url:
                    continue
                seen_url.add(link)
            if not h and not s:
                continue

            items.append(
                Item(
                    idx=idx,
                    item_id=g("Item_ID") or str(idx),
                    headline=h,
                    summary=s,
                    raw_headline=raw_h,
                    link=link,
                    published_utc=g("Published_UTC"),
                    topic_name=g("Topic_Name"),
                )
            )
    return items


def select_portion(items_all: Sequence[Item], *, n: int, overfetch: int, seed: int, sort_mode: str) -> List[Item]:
    scored: List[Tuple[float, int]] = []
    for i, it in enumerate(items_all):
        dt = _parse_dt(it.published_utc)
        ts = dt.timestamp() if dt else float(i)
        scored.append((ts, i))
    if sort_mode == "random":
        rng = random.Random(seed)
        rng.shuffle(scored)
    else:
        scored.sort(reverse=True)
    pool = [items_all[i] for _, i in scored[: max(1, int(overfetch))]]
    return pool[: max(1, int(n))]


EVENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "event": {"type": "string"},
                },
                "required": ["id", "event"],
            },
        }
    },
    "required": ["items"],
}


def _openai_responses_extract_text(resp: Dict[str, Any]) -> str:
    if isinstance(resp.get("output_text"), str) and resp.get("output_text"):
        return resp["output_text"]
    out = resp.get("output") or []
    parts: List[str] = []
    for o in out:
        for c in (o.get("content") or []):
            if isinstance(c, dict) and c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                parts.append(c["text"])
    return "\\n".join(parts).strip()


def _json_load_loose(s: str) -> Any:
    s = (s or "").strip()
    if not s:
        raise ValueError("empty response")
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is not None and start > 0:
        s = s[start:]
    return json.loads(s)


def normalize_event_label(label: str) -> str:
    s = _clean_text(label or "")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "General"
    if s.lower() in ("general", "generic", "misc", "miscellaneous", "other"):
        return "General"
    # Allow words, whitespace, ampersand, slash, and hyphen. Put '-' last to avoid range parsing.
    s = re.sub(r"[^\w\s&/\-]+", "", s).strip()
    words = []
    for w in s.split():
        if w.isupper() and len(w) <= 6:
            words.append(w)
        else:
            words.append(w[:1].upper() + w[1:].lower())
    s = " ".join(words)
    if len(s) > 60:
        s = s[:60].rsplit(" ", 1)[0].strip()
    return s or "General"


def openai_event_batch(batch: Sequence[Tuple[str, str]], *, api_key: str, model: str, timeout_s: int) -> Dict[str, str]:
    sys_prompt = (
        "You label news/events. Return ONLY JSON that matches the provided schema.\\n"
        "Task: for each headline, output a normalized specific event name in 3-5 words.\\n"
        "Rules:\\n"
        "- If generic/evergreen/listicle/market-size report/not a single event: output 'General'.\\n"
        "- Normalize synonyms and naming variants to the same event string.\\n"
        "- Do NOT output publisher/site names as events.\\n"
        "- No extra commentary.\\n"
    )
    user_lines = ["Headlines:"]
    for _id, h in batch:
        user_lines.append(f"- id={_id} | {h}")
    user_prompt = "\\n".join(user_lines)

    payload = {
        "model": model,
        "instructions": sys_prompt,
        "input": user_prompt,
        "text": {"format": {"type": "json_schema", "name": "event_labels", "schema": EVENT_SCHEMA, "strict": True}},
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    # Transient DNS failures happen on some networks; retry a few times.
    last_err: Optional[BaseException] = None
    body = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            last_err = None
            break
        except urllib.error.HTTPError as e:
            # Include the server's JSON error message; it usually explains *why* auth failed.
            detail = ""
            try:
                raw = e.read().decode("utf-8", errors="replace")
                detail = raw.strip()
            except Exception:
                detail = ""
            msg = f"OpenAI API request failed: HTTP {getattr(e, 'code', '?')} {getattr(e, 'reason', '')}".strip()
            if detail:
                msg += f"\nResponse body:\n{detail}"
            raise RuntimeError(msg) from e
        except urllib.error.URLError as e:
            last_err = e
            if attempt < 2:
                time.sleep(0.6 * (2**attempt))
                continue
            raise RuntimeError(
                "OpenAI API request failed: network/DNS error.\n"
                f"Details: {e}\n"
                "Checks:\n"
                "- Are you online (VPN, captive portal, corporate DNS/proxy)?\n"
                "- Can you resolve api.openai.com? Try: `nslookup api.openai.com` or `dig api.openai.com`\n"
                "- Can you reach the host? Try: `curl -I https://api.openai.com/v1/responses`\n"
                "- If you need a proxy, set HTTPS_PROXY/HTTP_PROXY in your environment.\n"
            ) from e
    obj = json.loads(body)
    txt = _openai_responses_extract_text(obj)
    parsed = _json_load_loose(txt)
    items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        raise ValueError("LLM returned invalid JSON (missing items[])")
    out: Dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        _id = str(it.get("id", "")).strip()
        ev = normalize_event_label(str(it.get("event", "")))
        if _id:
            out[_id] = ev
    return out


def load_cache_jsonl(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = str(obj.get("key", "")).strip()
                v = str(obj.get("event", "")).strip()
                if k and v:
                    out[k] = v
            except Exception:
                continue
    return out


def append_cache_jsonl(path: str, rows: Sequence[Tuple[str, str]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for k, v in rows:
            f.write(json.dumps({"key": k, "event": v}, ensure_ascii=True) + "\\n")


def event_key_for_item(it: Item) -> str:
    return _stable_id(f"{it.link}|{it.headline}")


def fuzzy_merge_labels(labels: List[str], *, threshold: float) -> Dict[str, str]:
    labs = [l for l in labels if l and l != "General"]
    labs = sorted(set(labs), key=lambda s: (-len(s), s))
    canon: List[str] = []
    mapping: Dict[str, str] = {"General": "General"}
    for l in labs:
        best = None
        best_score = 0.0
        for c in canon:
            s = SequenceMatcher(None, l.lower(), c.lower()).ratio()
            if s > best_score:
                best_score = s
                best = c
        if best is not None and best_score >= float(threshold):
            mapping[l] = best
        else:
            canon.append(l)
            mapping[l] = l
    return mapping


def _load_dotenv(path: str = ".env") -> None:
    # Minimal .env loader (stdlib-only).
    #
    # We intentionally override obviously-bad existing env values (e.g. "...") because it's
    # easy to accidentally export a placeholder during debugging.
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if not k:
                    continue
                # Strip simple quotes.
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                existing = os.environ.get(k, "")
                # Override only when existing looks like a placeholder/bad value.
                if existing and existing not in ("...", "''", '""') and len(existing) >= 20:
                    continue
                os.environ[k] = v
    except Exception:
        return


ENTITY_BAD_TOKENS = {
    "times",
    "news",
    "post",
    "daily",
    "journal",
    "tribune",
    "herald",
    "telegraph",
    "chronicle",
    "gazette",
    "guardian",
    "insider",
    "report",
    "reports",
    "review",
    "guide",
    "explained",
    "analysis",
    "latest",
    "top",
    "best",
    "more",
    "know",
    "about",
    "everything",
    "need",
    "watch",
    "listen",
    "video",
    "videos",
    "photo",
    "photos",
    "image",
    "images",
    "bbc",
    "cbc",
    "cnn",
    "wsj",
    "ft",
    "nyt",
    "reuters",
    "ap",
    "afp",
}

ENT_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|[A-Z]{2,}|iPhone|iPad|MacBook|OpenAI|ChatGPT|Sora|YouTube|TikTok)\b")


def extract_headline_entities(raw_headline: str) -> set[str]:
    txt = TAG_RE.sub(" ", raw_headline or "")
    found: set[str] = set()
    for m in ENT_RE.findall(txt):
        w = re.sub(r"[^A-Za-z0-9 \\-]+", "", m).strip()
        if not w:
            continue
        wl = w.lower()
        parts = [p for p in wl.split() if p]
        # Keep only informative tokens; don't discard the whole entity just because it contains "and/the".
        kept = [p for p in parts if (p not in STOPWORDS and p not in ENTITY_BAD_TOKENS and len(p) >= 3)]
        if not kept:
            continue
        # Normalize to a compact key; helps matching across punctuation variants.
        key = " ".join(kept)
        if len(key) < 3:
            continue
        found.add(key)
    return found


def entity_graph_components(
    items: Sequence[Item],
    *,
    min_shared_rare_ents: int,
    entity_max_df: int,
    entity_bucket_max: int,
    min_cluster_size: int,
) -> List[List[int]]:
    n = len(items)
    ent_sets: List[set[str]] = [extract_headline_entities(it.raw_headline) for it in items]
    df: Dict[str, int] = {}
    for s in ent_sets:
        for e in s:
            df[e] = df.get(e, 0) + 1

    ent_to_docs: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(ent_sets):
        for e in s:
            if df.get(e, 0) <= int(entity_max_df):
                ent_to_docs[e].append(i)

    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for e, docs in ent_to_docs.items():
        if len(docs) < 2:
            continue
        if len(docs) > int(entity_bucket_max):
            continue
        docs = sorted(docs)
        for a in range(len(docs) - 1):
            da = docs[a]
            for b in range(a + 1, len(docs)):
                db = docs[b]
                pair_counts[(da, db)] += 1

    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for (a, b), c in pair_counts.items():
        if c >= int(min_shared_rare_ents):
            union(a, b)

    comps: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    out = [c for c in comps.values() if len(c) >= int(min_cluster_size)]
    out.sort(key=len, reverse=True)
    return out


def _domain(it: Item) -> str:
    if it.topic_name:
        return it.topic_name
    try:
        p = urllib.parse.urlsplit(it.link or "")
        return p.netloc or "unknown"
    except Exception:
        return "unknown"


def build_cluster_objects(items: Sequence[Item], groups: Sequence[Sequence[int]], *, label_by_item: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    now = _dt.datetime.now(tz=_dt.UTC)
    for cid, idxs in enumerate(groups):
        mem = [items[i] for i in idxs]
        mem_sorted = sorted(mem, key=lambda it: (_parse_dt(it.published_utc) or now), reverse=True)
        rep = mem_sorted[0]
        cat = Counter([classify_category(it.headline, it.summary) for it in mem]).most_common(1)[0][0]
        doms = [_domain(it) for it in mem]
        ents: List[str] = []
        for it in mem:
            ents.extend(sorted(extract_headline_entities(it.raw_headline)))
        ent_counts = Counter(ents)
        story_id = _stable_id(f"{rep.link}|{rep.headline}|{cid}")
        label = ""
        if label_by_item is not None:
            label = label_by_item.get(idxs[0], "")
        out.append(
            {
                "story_id": story_id,
                "cluster_id": cid,
                "event_label": label,
                "topic_category": cat,
                "size": len(idxs),
                "source_diversity": len(set(doms)),
                "age_days": float(((now - (_parse_dt(rep.published_utc) or now)).total_seconds()) / 86400.0),
                "entities": [{"entity": e, "count": int(c)} for (e, c) in ent_counts.most_common(15)],
                "neutral_core_summary": {"headline": rep.headline, "summary": (rep.summary or "")[:300]},
                "key_facts": [it.headline for it in mem_sorted[:7] if it.headline],
                "supporting_articles": [
                    {
                        "headline": it.headline,
                        "summary": (it.summary or "")[:220],
                        "link": it.link,
                        "published_utc": it.published_utc,
                        "source": _domain(it),
                    }
                    for it in mem_sorted[:10]
                ],
            }
        )
    return out


def write_report(path: str, *, meta: Dict[str, Any], clusters: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Event Clusters")
    lines.append("")
    for k in sorted(meta.keys()):
        lines.append(f"- {k}: {meta[k]}")
    lines.append("")
    for i, c in enumerate(clusters, 1):
        lines.append(f"## {i}. {c.get('neutral_core_summary', {}).get('headline', '(no headline)')}")
        if c.get("event_label"):
            lines.append(f"- event: `{c.get('event_label')}`")
        lines.append(f"- story_id: `{c.get('story_id')}`")
        lines.append(f"- category: `{c.get('topic_category')}`")
        lines.append(f"- size: {c.get('size')} src_div: {c.get('source_diversity')} age_days: {float(c.get('age_days', 0.0)):.2f}")
        ents = [e.get("entity") for e in (c.get("entities") or [])[:10]]
        if ents:
            lines.append(f"- entities: {', '.join(ents)}")
        facts = c.get("key_facts") or []
        if facts:
            lines.append("- headlines:")
            for f in facts[:5]:
                lines.append(f"  - {f}")
        lines.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/scraped_articles_backup.csv")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--overfetch", type=int, default=1200)
    ap.add_argument("--sort", choices=["recency", "random"], default="recency")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--method", choices=["llm_events", "entity_graph"], default="entity_graph")
    ap.add_argument("--min-cluster-size", type=int, default=3)

    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--batch-size", type=int, default=80)
    ap.add_argument("--timeout-s", type=int, default=60)
    ap.add_argument("--cache", default="data/event_labels_cache.jsonl")
    ap.add_argument("--fuzzy-merge", type=float, default=0.92)
    ap.add_argument("--drop-general", action="store_true")

    ap.add_argument("--min-shared-rare-ents", type=int, default=2)
    ap.add_argument("--entity-max-df", type=int, default=30)
    ap.add_argument("--entity-bucket-max", type=int, default=40)

    ap.add_argument("--out", default="data/event_clusters.json")
    ap.add_argument("--out-report", default="data/event_clusters_report.md")
    args = ap.parse_args(argv)

    # Allow users to set OPENAI_API_KEY in a local .env file without extra deps.
    _load_dotenv(".env")

    items_all = load_items(args.input)
    portion = select_portion(items_all, n=int(args.n), overfetch=int(args.overfetch), seed=int(args.seed), sort_mode=args.sort)

    if args.method == "entity_graph":
        comps = entity_graph_components(
            portion,
            min_shared_rare_ents=int(args.min_shared_rare_ents),
            entity_max_df=int(args.entity_max_df),
            entity_bucket_max=int(args.entity_bucket_max),
            min_cluster_size=int(args.min_cluster_size),
        )
        clusters = build_cluster_objects(portion, comps)
        meta = {
            "generated_at_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(),
            "method": "entity_graph",
            "n_items": len(portion),
            "n_clusters": len(clusters),
            "min_cluster_size": int(args.min_cluster_size),
            "min_shared_rare_ents": int(args.min_shared_rare_ents),
            "entity_max_df": int(args.entity_max_df),
            "entity_bucket_max": int(args.entity_bucket_max),
        }
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is not set. Put `OPENAI_API_KEY=...` in .env or export it in your shell.")
        # Basic sanity check (don't print the key).
        if len(api_key) < 20 or not api_key.startswith("sk-"):
            raise SystemExit(
                "OPENAI_API_KEY looks invalid (too short or not starting with `sk-`). "
                "Double-check your .env file (no extra quotes/typos) and that you're using an OpenAI API key."
            )
        cache = load_cache_jsonl(args.cache)
        labels: Dict[str, str] = {}
        needed: List[Item] = []
        for it in portion:
            k = event_key_for_item(it)
            v = cache.get(k)
            if v:
                labels[it.item_id] = normalize_event_label(v)
            else:
                needed.append(it)

        batch_size = max(1, int(args.batch_size))
        new_cache_rows: List[Tuple[str, str]] = []
        for i in range(0, len(needed), batch_size):
            batch_items = needed[i : i + batch_size]
            batch = [(it.item_id, it.headline) for it in batch_items]
            out = openai_event_batch(batch, api_key=api_key, model=args.model, timeout_s=int(args.timeout_s))
            for it in batch_items:
                ev = normalize_event_label(out.get(it.item_id, "General"))
                labels[it.item_id] = ev
                new_cache_rows.append((event_key_for_item(it), ev))
        append_cache_jsonl(args.cache, new_cache_rows)

        mapping = fuzzy_merge_labels(list(labels.values()), threshold=float(args.fuzzy_merge))
        for k, v in list(labels.items()):
            labels[k] = mapping.get(v, v)

        groups: Dict[str, List[int]] = defaultdict(list)
        for i, it in enumerate(portion):
            ev = labels.get(it.item_id, "General")
            if args.drop_general and ev == "General":
                continue
            groups[ev].append(i)

        kept = [(ev, idxs) for ev, idxs in groups.items() if len(idxs) >= int(args.min_cluster_size)]
        kept.sort(key=lambda t: len(t[1]), reverse=True)
        comps = [idxs for _ev, idxs in kept]

        label_by_item: Dict[int, str] = {}
        for ev, idxs in kept:
            for i in idxs:
                label_by_item[i] = ev

        clusters = build_cluster_objects(portion, comps, label_by_item=label_by_item)
        meta = {
            "generated_at_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(),
            "method": "llm_events",
            "model": args.model,
            "n_items": len(portion),
            "n_clusters": len(clusters),
            "min_cluster_size": int(args.min_cluster_size),
            "batch_size": batch_size,
            "cache_path": args.cache,
            "fuzzy_merge": float(args.fuzzy_merge),
            "drop_general": bool(args.drop_general),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "clusters": clusters}, f, ensure_ascii=True, indent=2)
    write_report(args.out_report, meta=meta, clusters=clusters[:80])
    print(f"loaded_all={len(items_all)} portion={len(portion)} clusters={len(clusters)}")
    print(f"wrote {args.out}")
    print(f"wrote {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
