#!/usr/bin/env python3
"""
Story clustering pipeline (stdlib-only).

Portion -> dedupe -> embed -> cluster (spherical k-means) -> refine/split blobs -> prune outliers
-> merge near-duplicates -> rank clusters and emit core objects + report.

Defaults are tuned for "single-event nucleus" clustering:
- embeddings use title-only by default and boost entity tokens
- min_cluster_size=3
- min_cohesion=0.60
- gated refine requires 2+ shared entities (3 for politics) or 2+ rare headline ngrams
- outlier prune uses title TF-IDF average similarity
- centroid merge merges near-duplicate clusters when similarity > 0.85 and entity overlap exists
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import html as _html
import json
import math
import os
import random
import re
import urllib.parse
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Dict, List, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")

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
    # RSS / web boilerplate
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

# Source-y / boilerplate tokens that often show up as capitalized phrases but are not event entities.
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
    "open",
    "share",
    "home",
    "market",
    "markets",
    "forecast",
    "insights",
    "frontiers",
    "reach",
    "billion",
    "usd",
    "world",
    "largest",
    "biggest",
    "new",
    "first",
    "last",
    "rise",
    "morning",
    "magazine",
    # media acronyms
    "bbc",
    "cbc",
    "cnn",
    "wsj",
    "ft",
    "nyt",
    "reuters",
    "ap",
    "afp",
    "tech",
    "ai",
}


def _stable_u64(s: str) -> int:
    h = blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


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


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def l2_norm(a: Sequence[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        s += a[i] * a[i]
    return math.sqrt(s)


def normalize_vec(v: array) -> array:
    n = l2_norm(v)
    if n <= 1e-12:
        return v
    inv = 1.0 / n
    for i in range(len(v)):
        v[i] *= inv
    return v


def _ri_components(token: str, dim: int, *, n_nonzero: int, salt: int) -> List[Tuple[int, float]]:
    comps: List[Tuple[int, float]] = []
    used: set[int] = set()
    for i in range(n_nonzero):
        u = _stable_u64(f"ri:{salt}:{token}:{i}")
        j = int(u % dim)
        if j in used:
            u2 = _stable_u64(f"ri:{salt}:{token}:{i}:x")
            j = int(u2 % dim)
        used.add(j)
        sign = 1.0 if (u >> 63) == 0 else -1.0
        comps.append((j, sign))
    return comps


def build_df_from_tokens(token_lists: Sequence[Sequence[str]]) -> Dict[str, int]:
    df: Dict[str, int] = {}
    for toks in token_lists:
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    return df


def idf_map(df: Dict[str, int], n_docs: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for w, d in df.items():
        out[w] = math.log((n_docs + 1) / (d + 1)) + 1.0
    return out


def embed_token_lists(token_lists: Sequence[Sequence[str]], idf: Dict[str, float], *, dim: int, salt: int) -> List[array]:
    vecs: List[array] = []
    for toks in token_lists:
        tf: Dict[str, int] = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1

        v = array("f", [0.0] * dim)
        for w, c in tf.items():
            w_idf = idf.get(w)
            if w_idf is None:
                continue
            w_tf = 1.0 + math.log(1.0 + c)
            weight = w_tf * w_idf
            for j, sign in _ri_components(w, dim, n_nonzero=10, salt=salt):
                v[j] += float(weight * sign)
        normalize_vec(v)
        vecs.append(v)
    return vecs


def spherical_kmeans(vecs: Sequence[Sequence[float]], *, k: int, iters: int, seed: int) -> Tuple[List[int], List[array]]:
    n = len(vecs)
    if n <= 0:
        return ([], [])
    k = max(1, min(int(k), n))
    dim = len(vecs[0])
    rng = random.Random(seed)

    # k-means++ init using cosine distance (1 - dot).
    cents: List[array] = []
    first = rng.randrange(n)
    cents.append(array("f", [float(x) for x in vecs[first]]))
    normalize_vec(cents[0])

    best_d = [1.0 - dot(vecs[i], cents[0]) for i in range(n)]
    for _ in range(1, k):
        wsum = 0.0
        weights = []
        for d in best_d:
            w = max(0.0, d) ** 2
            weights.append(w)
            wsum += w
        if wsum <= 1e-12:
            pick = rng.randrange(n)
            c = array("f", [float(x) for x in vecs[pick]])
            normalize_vec(c)
            cents.append(c)
            continue
        r = rng.random() * wsum
        acc = 0.0
        pick = 0
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                pick = i
                break
        c = array("f", [float(x) for x in vecs[pick]])
        normalize_vec(c)
        cents.append(c)
        for i in range(n):
            d = 1.0 - dot(vecs[i], cents[-1])
            if d < best_d[i]:
                best_d[i] = d

    assign = [-1] * n
    for _it in range(max(1, int(iters))):
        changed = 0
        for i in range(n):
            v = vecs[i]
            best = -1e9
            best_k = 0
            for ci in range(k):
                s = dot(v, cents[ci])
                if s > best:
                    best = s
                    best_k = ci
            if assign[i] != best_k:
                changed += 1
                assign[i] = best_k

        sums = [array("f", [0.0] * dim) for _ in range(k)]
        counts = [0] * k
        for i, ci in enumerate(assign):
            counts[ci] += 1
            v = vecs[i]
            sv = sums[ci]
            for j in range(dim):
                sv[j] += float(v[j])
        for ci in range(k):
            if counts[ci] <= 0:
                pick = rng.randrange(n)
                cents[ci] = array("f", [float(x) for x in vecs[pick]])
                normalize_vec(cents[ci])
            else:
                normalize_vec(sums[ci])
                cents[ci] = sums[ci]

        if changed <= max(1, int(0.01 * n)):
            break

    return assign, cents


def clusters_from_assign(assign: Sequence[int]) -> List[List[int]]:
    if not assign:
        return []
    k = max(assign) + 1
    members: List[List[int]] = [[] for _ in range(k)]
    for i, ci in enumerate(assign):
        if 0 <= ci < k:
            members[ci].append(i)
    return members


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    d = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            d += va * vb
    na = math.sqrt(sum(x * x for x in a.values()))
    nb = math.sqrt(sum(x * x for x in b.values()))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return d / (na * nb)


def title_tfidf_vectors(titles: Sequence[str], *, max_tokens: int = 60) -> List[Dict[str, float]]:
    docs = [tokenize(t)[:max_tokens] for t in titles]
    n = len(docs) or 1
    df: Dict[str, int] = {}
    for toks in docs:
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    idf: Dict[str, float] = {}
    for w, d in df.items():
        idf[w] = math.log((n + 1) / (d + 1)) + 1.0
    vecs: List[Dict[str, float]] = []
    for toks in docs:
        tf: Dict[str, int] = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        v: Dict[str, float] = {}
        for w, c in tf.items():
            v[w] = (1.0 + math.log(1.0 + c)) * idf.get(w, 0.0)
        vecs.append(v)
    return vecs


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


ENT_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|[A-Z]{2,}|iPhone|iPad|MacBook|OpenAI|ChatGPT|Sora|YouTube|TikTok)\b")


def extract_entities(raw_headline: str, raw_summary: str) -> List[str]:
    txt = f"{raw_headline or ''} {raw_summary or ''}"
    txt = TAG_RE.sub(" ", txt)
    out: List[str] = []
    bad = set(STOPWORDS)
    bad.update(ENTITY_BAD_TOKENS)
    bad.update(
        {
            "trend",
            "trends",
            "size",
            "deep",
            "dive",
            "today",
            "tonight",
            "tomorrow",
            "yesterday",
        }
    )
    for m in ENT_RE.findall(txt):
        w = re.sub(r"[^A-Za-z0-9 \-]+", "", m).strip()
        if not w:
            continue
        wl = w.lower()
        if wl.endswith(".com") or wl.endswith(".net") or wl.endswith(".org"):
            continue
        parts = [p for p in wl.split() if p]
        if not parts:
            continue
        if any(p in bad for p in parts):
            continue
        if len(wl) < 3:
            continue
        out.append(wl)
    return out


@dataclass(frozen=True)
class Item:
    idx: int
    item_id: str
    headline: str
    summary: str
    raw_headline: str
    raw_summary: str
    link: str
    published_utc: str
    topic_name: str
    topic_cluster_id: str
    entity_key: str
    event_key: str

    @property
    def text(self) -> str:
        return " ".join(p for p in (self.headline, self.summary) if p)


def load_items(csv_path: str, *, dedupe_url: bool = True) -> List[Item]:
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
            if dedupe_url and link:
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
                    raw_summary=raw_s,
                    link=link,
                    published_utc=g("Published_UTC"),
                    topic_name=g("Topic_Name"),
                    topic_cluster_id=g("Topic_Cluster_ID"),
                    entity_key=g("Entity_Key"),
                    event_key=g("Event_Key"),
                )
            )
    return items


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


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa.intersection(sb))
    union = len(sa) + len(sb) - inter
    return float(inter) / float(union) if union else 0.0


def select_portion(
    items_all: Sequence[Item],
    *,
    overfetch: int,
    target_n: int,
    seed: int,
    sort_mode: str,
    dedupe_event_key: bool,
    title_jaccard_threshold: float,
) -> List[Item]:
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
    initial = [items_all[i] for _, i in scored[: max(1, int(overfetch))]]

    picked: List[Item] = []
    picked_keys: set[Tuple[str, str]] = set()
    seen_event: set[str] = set()
    seen_head_sig: Dict[Tuple[str, ...], int] = {}

    def head_sig(h: str) -> Tuple[str, ...]:
        toks = tokenize(h)
        toks = [t for t in toks if len(t) >= 3]
        toks.sort()
        return tuple(toks[:10])

    def try_add(it: Item) -> bool:
        if len(picked) >= int(target_n):
            return False
        key = (it.link or "", it.item_id or "")
        if key in picked_keys:
            return False

        if dedupe_event_key:
            ek = (it.event_key or "").strip().lower()
            if ek and ek != "unknown_event":
                if ek in seen_event:
                    return False

        sig = head_sig(it.headline or "")
        if sig and sig in seen_head_sig:
            prev = picked[seen_head_sig[sig]]
            if _jaccard(tokenize(it.headline or ""), tokenize(prev.headline or "")) >= float(title_jaccard_threshold):
                return False

        if dedupe_event_key:
            ek = (it.event_key or "").strip().lower()
            if ek and ek != "unknown_event":
                seen_event.add(ek)
        if sig:
            seen_head_sig.setdefault(sig, len(picked))
        picked.append(it)
        picked_keys.add(key)
        return True

    for it in initial:
        if len(picked) >= int(target_n):
            break
        try_add(it)
    return picked


def cluster_cohesion(idxs: Sequence[int], vecs: Sequence[Sequence[float]]) -> float:
    if not idxs:
        return 0.0
    dim = len(vecs[0])
    cent = array("f", [0.0] * dim)
    for i in idxs:
        v = vecs[i]
        for j in range(dim):
            cent[j] += float(v[j])
    normalize_vec(cent)
    return sum(dot(vecs[i], cent) for i in idxs) / float(len(idxs))


def prune_cluster_by_title_sim(
    idxs: Sequence[int],
    *,
    title_tfidf: Sequence[Dict[str, float]],
    min_avg_sim: float,
    min_pair_sim: float,
    min_cluster_size: int,
) -> List[int]:
    cur = list(idxs)
    while len(cur) >= int(min_cluster_size):
        if len(cur) <= 2:
            a, b = cur[0], cur[-1]
            if _cosine_sparse(title_tfidf[a], title_tfidf[b]) >= float(min_pair_sim):
                return cur
            return []

        worst = cur[0]
        worst_avg = 1e9
        all_ok = True
        for ai in cur:
            acc = 0.0
            n = 0
            for bj in cur:
                if bj == ai:
                    continue
                acc += _cosine_sparse(title_tfidf[ai], title_tfidf[bj])
                n += 1
            avg = acc / float(max(1, n))
            if avg < float(min_avg_sim):
                all_ok = False
            if avg < worst_avg:
                worst_avg = avg
                worst = ai

        if all_ok:
            if float(min_pair_sim) > 0.0:
                min_s = 1.0
                for i in range(len(cur) - 1):
                    ai = cur[i]
                    for j in range(i + 1, len(cur)):
                        bj = cur[j]
                        s = _cosine_sparse(title_tfidf[ai], title_tfidf[bj])
                        if s < min_s:
                            min_s = s
                if min_s < float(min_pair_sim):
                    cur = [x for x in cur if x != worst]
                    continue
            return cur

        cur = [x for x in cur if x != worst]
    return []


def refine_clusters_gated(
    clusters: Sequence[Sequence[int]],
    *,
    vecs: Sequence[Sequence[float]],
    ent_sets: Sequence[set[str]],
    ng_sets: Sequence[set[str]],
    gate_sim: float,
    gate_min_shared_ents: int,
    gate_min_shared_ngrams: int,
    gate_min_shared_ents_policy: int,
    cats: Sequence[str],
    entity_df: Dict[str, int],
    entity_max_df: int,
    entity_bucket_max: int,
    ngram_bucket_max: int,
    min_cluster_size: int,
) -> List[List[int]]:
    def good_ent_count(a: set[str], b: set[str], *, max_df: int) -> int:
        if not a or not b:
            return 0
        if len(a) > len(b):
            a, b = b, a
        c = 0
        for x in a:
            if x in b and entity_df.get(x, 0) <= int(max_df):
                c += 1
        return c

    def inter_count(a: set[str], b: set[str]) -> int:
        if not a or not b:
            return 0
        if len(a) > len(b):
            a, b = b, a
        c = 0
        for x in a:
            if x in b:
                c += 1
        return c

    refined: List[List[int]] = []

    for idxs in clusters:
        if len(idxs) < max(2, int(min_cluster_size)):
            continue
        m = len(idxs)
        parent = list(range(m))
        size = [1] * m

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

        ent_to_locals: Dict[str, List[int]] = defaultdict(list)
        ng_to_locals: Dict[str, List[int]] = defaultdict(list)
        for li, gi in enumerate(idxs):
            for e in ent_sets[gi]:
                if entity_df.get(e, 0) <= int(entity_max_df):
                    ent_to_locals[e].append(li)
            for g in ng_sets[gi]:
                ng_to_locals[g].append(li)

        for e, locals_ in ent_to_locals.items():
            if len(locals_) < 2 or len(locals_) > int(entity_bucket_max):
                continue
            for a_i in range(len(locals_) - 1):
                la = locals_[a_i]
                ga = idxs[la]
                va = vecs[ga]
                ea = ent_sets[ga]
                for b_i in range(a_i + 1, len(locals_)):
                    lb = locals_[b_i]
                    gb = idxs[lb]
                    if dot(va, vecs[gb]) < float(gate_sim):
                        continue
                    req = int(gate_min_shared_ents)
                    if (cats[ga] == "politics" or cats[gb] == "politics") and int(gate_min_shared_ents_policy) > req:
                        req = int(gate_min_shared_ents_policy)
                    if good_ent_count(ea, ent_sets[gb], max_df=entity_max_df) < req:
                        continue
                    union(la, lb)

        for g, locals_ in ng_to_locals.items():
            if len(locals_) < 2 or len(locals_) > int(ngram_bucket_max):
                continue
            for a_i in range(len(locals_) - 1):
                la = locals_[a_i]
                ga = idxs[la]
                va = vecs[ga]
                nga = ng_sets[ga]
                for b_i in range(a_i + 1, len(locals_)):
                    lb = locals_[b_i]
                    gb = idxs[lb]
                    if dot(va, vecs[gb]) < float(gate_sim):
                        continue
                    if inter_count(nga, ng_sets[gb]) < int(gate_min_shared_ngrams):
                        continue
                    union(la, lb)

        comps: Dict[int, List[int]] = defaultdict(list)
        for li, gi in enumerate(idxs):
            comps[find(li)].append(gi)
        for comp in comps.values():
            if len(comp) >= int(min_cluster_size):
                refined.append(comp)

    return refined


def merge_clusters_by_centroid(
    clusters: Sequence[Sequence[int]],
    *,
    vecs: Sequence[Sequence[float]],
    ent_sets: Sequence[set[str]],
    merge_sim: float,
    merge_min_shared_ents: int,
) -> List[List[int]]:
    cs = [list(c) for c in clusters if c]
    if len(cs) <= 1:
        return cs

    dim = len(vecs[0])
    cents: List[array] = []
    c_ents: List[set[str]] = []
    for c in cs:
        cent = array("f", [0.0] * dim)
        for i in c:
            v = vecs[i]
            for j in range(dim):
                cent[j] += float(v[j])
        normalize_vec(cent)
        cents.append(cent)
        es: set[str] = set()
        for i in c:
            es.update(ent_sets[i])
        c_ents.append(es)

    n = len(cs)
    parent = list(range(n))
    size = [len(cs[i]) for i in range(n)]

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

    def shared_ents(a: set[str], b: set[str]) -> int:
        if not a or not b:
            return 0
        if len(a) > len(b):
            a, b = b, a
        c = 0
        for x in a:
            if x in b:
                c += 1
        return c

    for i in range(n - 1):
        for j in range(i + 1, n):
            if dot(cents[i], cents[j]) < float(merge_sim):
                continue
            if int(merge_min_shared_ents) > 0 and shared_ents(c_ents[i], c_ents[j]) < int(merge_min_shared_ents):
                continue
            union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    out: List[List[int]] = []
    for g in groups.values():
        merged: List[int] = []
        seen: set[int] = set()
        for ci in g:
            for x in cs[ci]:
                if x in seen:
                    continue
                seen.add(x)
                merged.append(x)
        out.append(merged)
    return out


def _domain(it: Item) -> str:
    if it.topic_name:
        return it.topic_name
    try:
        p = urllib.parse.urlsplit(it.link or "")
        return p.netloc or "unknown"
    except Exception:
        return "unknown"


def build_cluster_cores(
    items: Sequence[Item],
    vecs: Sequence[Sequence[float]],
    clusters: Sequence[Sequence[int]],
    *,
    min_cluster_size: int,
    min_cohesion: float,
    now_utc: _dt.datetime,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cid, idxs in enumerate(clusters):
        if len(idxs) < int(min_cluster_size):
            continue

        dim = len(vecs[0])
        cent = array("f", [0.0] * dim)
        for i in idxs:
            v = vecs[i]
            for j in range(dim):
                cent[j] += float(v[j])
        normalize_vec(cent)
        coh = sum(dot(vecs[i], cent) for i in idxs) / float(len(idxs))
        if coh < float(min_cohesion):
            continue

        dts = [_parse_dt(items[i].published_utc) for i in idxs]
        dts2 = [dt for dt in dts if dt is not None]
        newest = max(dts2) if dts2 else None
        age_days = ((now_utc - newest).total_seconds() / 86400.0) if newest else 9999.0

        cats = [classify_category(items[i].headline, items[i].summary) for i in idxs]
        cat = Counter(cats).most_common(1)[0][0] if cats else "other"

        ent_all: List[str] = []
        for i in idxs:
            ent_all.extend(extract_entities(items[i].raw_headline, ""))  # headline-only
        ent_counts = Counter(ent_all)

        doms = [_domain(items[i]) for i in idxs]
        src_div = len(set(doms))

        sims = [(dot(vecs[i], cent), i) for i in idxs]
        sims.sort(reverse=True)
        rep = items[sims[0][1]]
        rep_head = (rep.headline or "").strip()
        rep_sum = (rep.summary or "").strip()
        rep_sum = rep_sum[:300] + ("..." if len(rep_sum) > 300 else "")

        story_id = blake2b(f"{rep_head}|{rep.link}".encode("utf-8", errors="ignore"), digest_size=8).hexdigest()

        key_facts = []
        for _s, i in sims[:7]:
            h = (items[i].headline or "").strip()
            if h:
                key_facts.append(h)

        out.append(
            {
                "story_id": story_id,
                "cluster_id": cid,
                "topic_category": cat,
                "size": len(idxs),
                "cohesion": float(coh),
                "source_diversity": int(src_div),
                "age_days": float(age_days),
                "entities": [{"entity": e, "count": int(c)} for (e, c) in ent_counts.most_common(15)],
                "neutral_core_summary": {"headline": rep_head, "summary": rep_sum},
                "key_facts": key_facts[:7],
                "confidence_score": float(coh),
                "supporting_articles": [
                    {
                        "headline": (items[i].headline or "").strip(),
                        "summary": (items[i].summary or "").strip()[:220],
                        "link": (items[i].link or "").strip(),
                        "published_utc": (items[i].published_utc or "").strip(),
                        "source": _domain(items[i]),
                    }
                    for (_s, i) in sims[:10]
                ],
            }
        )
    return out


def _parse_quotas(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    s = (s or "").strip()
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad quota segment: {part!r} (expected key=value)")
        k, v = part.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def rank_and_select(cores: Sequence[Dict[str, Any]], *, quotas: Dict[str, int]) -> List[Dict[str, Any]]:
    def score(c: Dict[str, Any]) -> float:
        size = float(c.get("size", 0))
        div = float(c.get("source_diversity", 0))
        coh = float(c.get("cohesion", 0.0))
        age = float(c.get("age_days", 9999.0))
        rec = math.exp(-max(0.0, age) / 5.0)
        return 1.3 * size + 1.4 * div + 7.5 * max(0.0, coh) + 8.0 * rec

    def bucket(c: Dict[str, Any]) -> str:
        cat = (c.get("topic_category") or "other").strip()
        head = ((c.get("neutral_core_summary") or {}).get("headline") or "").lower()
        ents = " ".join([e.get("entity", "") for e in (c.get("entities") or [])[:10]]).lower()
        blob = head + " " + ents
        if cat == "tech" and re.search(r"\b(openai|chatgpt|sora|llm|ai|model|gpu|chip)\b", blob):
            return "ai"
        if cat == "business":
            return "markets"
        if cat == "politics":
            return "policy"
        if cat == "tech":
            return "tech_industry"
        return cat

    by_bucket: Dict[str, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
    for c in cores:
        b = bucket(c)
        c["bucket"] = b
        by_bucket[b].append((score(c), c))
    for b in by_bucket:
        by_bucket[b].sort(key=lambda t: t[0], reverse=True)

    chosen: List[Dict[str, Any]] = []
    for b, q in quotas.items():
        if q <= 0:
            continue
        for _s, c in by_bucket.get(b, [])[:q]:
            chosen.append(c)

    wanted = sum(max(0, int(v)) for v in quotas.values())
    if len(chosen) < wanted:
        seen = {c.get("story_id") for c in chosen if c.get("story_id")}
        rest: List[Tuple[float, Dict[str, Any]]] = []
        for c in cores:
            sid = c.get("story_id")
            if sid and sid in seen:
                continue
            rest.append((score(c), c))
        rest.sort(key=lambda t: t[0], reverse=True)
        for _s, c in rest:
            if len(chosen) >= wanted:
                break
            chosen.append(c)

    chosen.sort(key=lambda c: c.get("confidence_score", 0.0), reverse=True)
    return chosen


def write_report(path: str, *, selected: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Story Clusters")
    lines.append("")
    for k in sorted(meta.keys()):
        lines.append(f"- {k}: {meta[k]}")
    lines.append("")
    for i, c in enumerate(selected, 1):
        lines.append(f"## {i}. {c.get('neutral_core_summary', {}).get('headline', '(no headline)')}")
        lines.append(f"- story_id: `{c.get('story_id')}`")
        lines.append(f"- bucket: `{c.get('bucket', c.get('topic_category'))}` (category: `{c.get('topic_category')}`)")
        lines.append(f"- size: {c.get('size')}, cohesion: {float(c.get('cohesion', 0.0)):.3f}, src_div: {c.get('source_diversity')}, age_days: {float(c.get('age_days', 0.0)):.2f}")
        ents = [e.get("entity") for e in (c.get("entities") or [])[:10]]
        if ents:
            lines.append(f"- entities: {', '.join(ents)}")
        facts = c.get("key_facts") or []
        if facts:
            lines.append("- key_facts:")
            for f in facts[:5]:
                lines.append(f"  - {f}")
        lines.append("- supporting_articles:")
        for a in (c.get("supporting_articles") or [])[:5]:
            lines.append(f"  - {a.get('headline')} ({a.get('source')})")
        lines.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/scraped_articles_backup.csv")
    ap.add_argument("--target-n", type=int, default=900)
    ap.add_argument("--overfetch", type=int, default=9000)
    ap.add_argument("--sort", choices=["recency", "random"], default="recency")
    ap.add_argument("--dedupe-event-key", action="store_true")
    ap.add_argument("--title-jaccard", type=float, default=0.92)

    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--k", type=int, default=280)
    ap.add_argument("--iters", type=int, default=18)
    ap.add_argument("--min-cluster-size", type=int, default=3)
    ap.add_argument("--min-cohesion", type=float, default=0.60)

    ap.add_argument("--gate-sim", type=float, default=0.58)
    ap.add_argument("--gate-min-shared-ents", type=int, default=2)
    ap.add_argument("--gate-min-shared-ents-policy", type=int, default=3)
    ap.add_argument("--gate-min-shared-ngrams", type=int, default=2)
    ap.add_argument("--entity-max-df", type=int, default=40)
    ap.add_argument("--entity-bucket-max", type=int, default=35)
    ap.add_argument("--ngram-idf", type=float, default=5.2)
    ap.add_argument("--ngram-bucket-max", type=int, default=25)

    ap.add_argument("--min-title-avg-sim", type=float, default=0.10)
    ap.add_argument("--min-title-pair-sim", type=float, default=0.00)

    ap.add_argument("--merge-sim", type=float, default=0.85)
    ap.add_argument("--merge-min-shared-ents", type=int, default=1)

    ap.add_argument("--embed-mode", choices=["title", "title_summary"], default="title")
    ap.add_argument("--entity-weight", type=float, default=2.2)
    ap.add_argument("--no-entity-boost", action="store_true")
    ap.add_argument("--entity-scope", choices=["headline", "headline_summary"], default="headline")

    ap.add_argument(
        "--quotas",
        default="ai=20,markets=8,tech_industry=8,policy=10,science_health=6,entertainment=6,sports=4,lifestyle=6,other=0",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data/story_clusters.json")
    ap.add_argument("--out-report", default="data/story_clusters_report.md")
    args = ap.parse_args(argv)

    now = _dt.datetime.now(tz=_dt.UTC)
    items_all = load_items(args.input, dedupe_url=True)
    portion = select_portion(
        items_all,
        overfetch=int(args.overfetch),
        target_n=int(args.target_n),
        seed=int(args.seed),
        sort_mode=args.sort,
        dedupe_event_key=bool(args.dedupe_event_key),
        title_jaccard_threshold=float(args.title_jaccard),
    )

    def key_entity(it: Item) -> set[str]:
        ek = (it.entity_key or "").strip().lower()
        if not ek or ek in ("unknown_entity", "unknown"):
            return set()
        ek = re.sub(r"[^a-z0-9_\\-]+", "_", ek)
        ek = re.sub(r"_+", "_", ek).strip("_")
        return {ek} if len(ek) >= 3 else set()

    def embed_tokens_for_item(it: Item) -> List[str]:
        base_text = it.headline if args.embed_mode == "title" else it.text
        toks = tokenize(base_text)[:220]
        if bool(args.no_entity_boost):
            return toks
        if args.entity_scope == "headline":
            ents = set(extract_entities(it.raw_headline, ""))
        else:
            ents = set(extract_entities(it.raw_headline, it.raw_summary))
        ents |= key_entity(it)
        boost = max(1, int(round(float(args.entity_weight))))
        for e in list(ents)[:24]:
            toks.extend([f"ent_{e}"] * boost)
        return toks

    token_lists = [embed_tokens_for_item(it) for it in portion]
    idf = idf_map(build_df_from_tokens(token_lists), n_docs=max(1, len(token_lists)))
    vecs = embed_token_lists(token_lists, idf, dim=int(args.dim), salt=int(args.seed))
    title_tfidf = title_tfidf_vectors([it.headline for it in portion], max_tokens=60)

    assign, _ = spherical_kmeans(vecs, k=int(args.k), iters=int(args.iters), seed=int(args.seed))
    clusters = clusters_from_assign(assign)

    cats = [classify_category(it.headline, it.summary) for it in portion]
    if args.entity_scope == "headline":
        ent_sets = [set(extract_entities(it.raw_headline, "")).union(key_entity(it)) for it in portion]
    else:
        ent_sets = [set(extract_entities(it.raw_headline, it.raw_summary)).union(key_entity(it)) for it in portion]
    entity_df: Dict[str, int] = {}
    for s in ent_sets:
        for e in s:
            entity_df[e] = entity_df.get(e, 0) + 1

    def headline_ngrams(it: Item) -> set[str]:
        toks = [t for t in tokenize(it.headline or "") if len(t) >= 3]
        ngs: set[str] = set()
        for i in range(len(toks) - 1):
            ngs.add(toks[i] + "_" + toks[i + 1])
        for i in range(len(toks) - 2):
            ngs.add(toks[i] + "_" + toks[i + 1] + "_" + toks[i + 2])
        return ngs

    ng_all = [headline_ngrams(it) for it in portion]
    ng_df: Dict[str, int] = {}
    for s in ng_all:
        for g in s:
            ng_df[g] = ng_df.get(g, 0) + 1
    ng_idf = idf_map(ng_df, n_docs=max(1, len(portion)))
    ng_sets = [{g for g in ng_all[i] if ng_idf.get(g, 0.0) >= float(args.ngram_idf)} for i in range(len(portion))]

    clusters_pre_refine = len([c for c in clusters if len(c) >= int(args.min_cluster_size)])
    kept: List[List[int]] = []
    to_refine: List[List[int]] = []
    for c in clusters:
        if len(c) < int(args.min_cluster_size):
            continue
        coh = cluster_cohesion(c, vecs)
        if coh >= float(args.min_cohesion):
            kept.append(list(c))
        else:
            to_refine.append(list(c))

    refined: List[List[int]] = []
    if to_refine:
        refined = refine_clusters_gated(
            to_refine,
            vecs=vecs,
            ent_sets=ent_sets,
            ng_sets=ng_sets,
            gate_sim=float(args.gate_sim),
            gate_min_shared_ents=int(args.gate_min_shared_ents),
            gate_min_shared_ngrams=int(args.gate_min_shared_ngrams),
            gate_min_shared_ents_policy=int(args.gate_min_shared_ents_policy),
            cats=cats,
            entity_df=entity_df,
            entity_max_df=int(args.entity_max_df),
            entity_bucket_max=int(args.entity_bucket_max),
            ngram_bucket_max=int(args.ngram_bucket_max),
            min_cluster_size=int(args.min_cluster_size),
        )
    clusters = kept + refined

    clusters_before_title_prune = len(clusters)
    pruned: List[List[int]] = []
    for c in clusters:
        cc = prune_cluster_by_title_sim(
            c,
            title_tfidf=title_tfidf,
            min_avg_sim=float(args.min_title_avg_sim),
            min_pair_sim=float(args.min_title_pair_sim),
            min_cluster_size=int(args.min_cluster_size),
        )
        if cc and len(cc) >= int(args.min_cluster_size):
            pruned.append(cc)
    clusters = pruned

    clusters_before_merge = len(clusters)
    clusters = merge_clusters_by_centroid(
        clusters,
        vecs=vecs,
        ent_sets=ent_sets,
        merge_sim=float(args.merge_sim),
        merge_min_shared_ents=int(args.merge_min_shared_ents),
    )

    cores = build_cluster_cores(
        items=portion,
        vecs=vecs,
        clusters=clusters,
        min_cluster_size=int(args.min_cluster_size),
        min_cohesion=float(args.min_cohesion),
        now_utc=now,
    )
    quotas = _parse_quotas(args.quotas)
    selected = rank_and_select(cores, quotas=quotas)

    meta = {
        "generated_at_utc": now.isoformat(),
        "input": args.input,
        "loaded_all": len(items_all),
        "portion_n": len(portion),
        "overfetch": int(args.overfetch),
        "dim": int(args.dim),
        "k": int(args.k),
        "min_cluster_size": int(args.min_cluster_size),
        "min_cohesion": float(args.min_cohesion),
        "gate_sim": float(args.gate_sim),
        "gate_min_shared_ents": int(args.gate_min_shared_ents),
        "gate_min_shared_ents_policy": int(args.gate_min_shared_ents_policy),
        "gate_min_shared_ngrams": int(args.gate_min_shared_ngrams),
        "entity_max_df": int(args.entity_max_df),
        "entity_bucket_max": int(args.entity_bucket_max),
        "ngram_idf": float(args.ngram_idf),
        "ngram_bucket_max": int(args.ngram_bucket_max),
        "clusters_pre_refine": int(clusters_pre_refine),
        "clusters_before_title_prune": int(clusters_before_title_prune),
        "min_title_avg_sim": float(args.min_title_avg_sim),
        "min_title_pair_sim": float(args.min_title_pair_sim),
        "clusters_before_merge": int(clusters_before_merge),
        "merge_sim": float(args.merge_sim),
        "merge_min_shared_ents": int(args.merge_min_shared_ents),
        "embed_mode": args.embed_mode,
        "entity_weight": float(args.entity_weight),
        "entity_boost": (not bool(args.no_entity_boost)),
        "entity_scope": args.entity_scope,
        "cores_total": len(cores),
        "selected_total": len(selected),
        "quotas": quotas,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "clusters": cores, "selected": selected}, f, ensure_ascii=True, indent=2)
    write_report(args.out_report, selected=selected, meta=meta)
    print(f"loaded_all={len(items_all)} portion={len(portion)} cores={len(cores)} selected={len(selected)}")
    print(f"wrote {args.out}")
    print(f"wrote {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

