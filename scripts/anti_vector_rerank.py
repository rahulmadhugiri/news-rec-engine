#!/usr/bin/env python3
"""
Anti-vector reranking prototype.

This implements the practical version of the "poison vector" idea:

- You cannot hand an embedding vector to an LLM and expect it to "move away" in latent space.
- But you *can*:
  1) generate multiple candidate summaries with an LLM
  2) embed each candidate (sentence-level)
  3) pick the candidate whose embeddings are *far* from the "poison" (skipped) anchor and
     *close* to the "positive" (listened) anchor.

Usage (example):

  python3 scripts/anti_vector_rerank.py \\
    --facts "SpaceX launched Starship... landed..." \\
    --positive "The Fed held rates steady today. Powell said inflation is cooling." \\
    --negative "This follows a 0.5% variance in the CPI for January." \\
    --model gpt-4o-mini \\
    --embed-model text-embedding-3-small \\
    --n-candidates 8

This script is stdlib-only. It requires OPENAI_API_KEY and network access.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Dict, List, Optional, Sequence, Tuple


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WS_RE = re.compile(r"\s+")


def _load_dotenv(path: str = ".env") -> None:
    """
    Minimal .env loader (no third-party deps).
    - Supports KEY=VALUE lines
    - Ignores comments and empty lines
    - Does not override already-set env vars
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if not k:
                    continue
                cur = os.environ.get(k)
                # If something like "..." is exported in the shell, prefer the .env value.
                if cur and len(cur.strip()) >= 20 and cur.strip() not in {"...", "changeme", "YOUR_KEY"}:
                    continue
                os.environ[k] = v
    except FileNotFoundError:
        return


def _stable_id(s: str) -> str:
    return blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).hexdigest()


def _norm_text(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip())


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # Defensive: allow empty vectors (shouldn't happen).
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"cosine dim mismatch: {len(a)} != {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    den = math.sqrt(na) * math.sqrt(nb)
    if den <= 0:
        return 0.0
    return float(dot / den)


def _mean(vs: List[List[float]]) -> List[float]:
    if not vs:
        return []
    d = len(vs[0])
    out = [0.0] * d
    for v in vs:
        if len(v) != d:
            raise ValueError("mean dim mismatch")
        for i, x in enumerate(v):
            out[i] += x
    inv = 1.0 / float(len(vs))
    for i in range(d):
        out[i] *= inv
    return out


def _http_post_json(url: str, api_key: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"HTTP {e.code} from {url}: {body[:500]}") from e


def openai_embeddings(texts: List[str], api_key: str, model: str, timeout_s: int) -> List[List[float]]:
    # Batch embeddings in one request.
    payload: Dict[str, Any] = {
        "model": model,
        "input": texts,
    }
    out = _http_post_json("https://api.openai.com/v1/embeddings", api_key=api_key, payload=payload, timeout_s=timeout_s)
    data = out.get("data") or []
    # embeddings API returns entries aligned to inputs
    embs: List[List[float]] = []
    for it in data:
        v = it.get("embedding")
        if not isinstance(v, list):
            v = []
        embs.append([float(x) for x in v])
    if len(embs) != len(texts):
        raise RuntimeError(f"embeddings count mismatch: got {len(embs)} for {len(texts)} inputs")
    return embs


EVENTLESS_SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "string"},
            },
        }
    },
    "required": ["candidates"],
}


def openai_generate_candidates(
    *,
    facts: str,
    model: str,
    api_key: str,
    n_candidates: int,
    temperature: float,
    timeout_s: int,
) -> List[List[str]]:
    instructions = (
        "You are writing a 4-sentence spoken-news summary.\n"
        "Return JSON only, matching the provided JSON schema.\n"
        "Rules:\n"
        "- Each candidate must be exactly 4 sentences.\n"
        "- Keep sentences concrete and easy to follow.\n"
        "- Avoid excessive numbers/percentages unless crucial.\n"
        "- No disclaimers, no markdown.\n"
    )
    user_input = (
        "Facts for the story (write a 4-sentence summary):\n"
        f"{facts.strip()}\n"
        f"\nGenerate {n_candidates} distinct candidates."
    )

    payload: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "temperature": temperature,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "candidates",
                "schema": EVENTLESS_SUMMARY_SCHEMA,
                "strict": True,
            }
        },
    }

    out = _http_post_json("https://api.openai.com/v1/responses", api_key=api_key, payload=payload, timeout_s=timeout_s)

    # Responses API: "output_text" is a convenience string; since we requested JSON, parse it.
    raw = out.get("output_text", "")
    if not raw:
        # Fallback: try to find any text content.
        raw = json.dumps(out)
    try:
        parsed = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Could not parse JSON candidates from output_text: {raw[:500]}") from e

    cands = parsed.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise RuntimeError(f"JSON did not contain candidates: {raw[:500]}")

    out_cands: List[List[str]] = []
    for cand in cands:
        if not isinstance(cand, list):
            continue
        sents = [_norm_text(str(x)) for x in cand][:4]
        if len(sents) != 4:
            continue
        out_cands.append(sents)

    if not out_cands:
        raise RuntimeError("No valid candidates (expected list of 4-sentence arrays).")
    return out_cands


def _split_sentences(s: str) -> List[str]:
    s = _norm_text(s)
    if not s:
        return []
    parts = SENT_SPLIT_RE.split(s)
    out: List[str] = []
    for p in parts:
        p = _norm_text(p)
        if not p:
            continue
        out.append(p)
    return out


@dataclass
class RankedCandidate:
    idx: int
    sentences: List[str]
    positive_mean: float
    negative_max: float
    score: float


def rank_candidates(
    *,
    candidates: List[List[str]],
    pos_anchor: str,
    neg_anchor: str,
    api_key: str,
    embed_model: str,
    alpha: float,
    timeout_s: int,
) -> List[RankedCandidate]:
    pos_sents = _split_sentences(pos_anchor)
    neg_sents = _split_sentences(neg_anchor)
    if not pos_sents:
        raise SystemExit("positive anchor produced 0 sentences; pass 1-4 sentences of 'listened' content.")
    if not neg_sents:
        raise SystemExit("negative anchor produced 0 sentences; pass the skipped/bail-out sentence(s).")

    # Build embedding batch:
    # [pos..., neg..., cand0_s1..s4, cand1_s1..s4, ...]
    texts: List[str] = []
    texts.extend(pos_sents)
    texts.extend(neg_sents)
    offset_pos = 0
    offset_neg = len(pos_sents)
    offset_cands = offset_neg + len(neg_sents)

    for cand in candidates:
        texts.extend(cand)

    embs = openai_embeddings(texts, api_key=api_key, model=embed_model, timeout_s=timeout_s)

    pos_vec = _mean(embs[offset_pos : offset_pos + len(pos_sents)])
    neg_vec = _mean(embs[offset_neg : offset_neg + len(neg_sents)])

    ranked: List[RankedCandidate] = []
    base = offset_cands
    for i, cand in enumerate(candidates):
        v_sents = embs[base + i * 4 : base + i * 4 + 4]
        pos_scores = [_cosine(v, pos_vec) for v in v_sents]
        neg_scores = [_cosine(v, neg_vec) for v in v_sents]
        pos_mean = float(sum(pos_scores) / 4.0)
        neg_max = float(max(neg_scores) if neg_scores else 0.0)
        score = pos_mean - alpha * neg_max
        ranked.append(
            RankedCandidate(
                idx=i,
                sentences=cand,
                positive_mean=pos_mean,
                negative_max=neg_max,
                score=score,
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="", help="Raw story facts to summarize.")
    ap.add_argument("--facts-file", default="", help="Path to a text file containing facts.")
    ap.add_argument("--positive", default="", help="Positive/listened anchor text (1-4 sentences).")
    ap.add_argument("--positive-file", default="", help="Path to a text file for positive anchor.")
    ap.add_argument("--negative", default="", help="Negative/skipped 'poison' anchor text (1+ sentences).")
    ap.add_argument("--negative-file", default="", help="Path to a text file for negative anchor.")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=1.0, help="Penalty weight for similarity to negative anchor.")
    ap.add_argument("--timeout-s", type=int, default=60)
    ap.add_argument("--show-all", action="store_true", help="Print all candidates with scores.")
    args = ap.parse_args(argv)

    _load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Put `OPENAI_API_KEY=...` in .env or export it in your shell.")
    if len(api_key) < 20 or not api_key.startswith("sk-"):
        raise SystemExit("OPENAI_API_KEY looks invalid. Double-check your .env file (no extra quotes/typos).")

    def read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    facts = args.facts
    if args.facts_file:
        facts = read_file(args.facts_file)
    facts = _norm_text(facts)
    if not facts:
        raise SystemExit("No facts provided. Use --facts or --facts-file.")

    pos = args.positive
    if args.positive_file:
        pos = read_file(args.positive_file)
    pos = _norm_text(pos)
    if not pos:
        raise SystemExit("No positive anchor provided. Use --positive or --positive-file.")

    neg = args.negative
    if args.negative_file:
        neg = read_file(args.negative_file)
    neg = _norm_text(neg)
    if not neg:
        raise SystemExit("No negative anchor provided. Use --negative or --negative-file.")

    t0 = time.time()
    candidates = openai_generate_candidates(
        facts=facts,
        model=args.model,
        api_key=api_key,
        n_candidates=max(1, args.n_candidates),
        temperature=float(args.temperature),
        timeout_s=int(args.timeout_s),
    )
    ranked = rank_candidates(
        candidates=candidates,
        pos_anchor=pos,
        neg_anchor=neg,
        api_key=api_key,
        embed_model=args.embed_model,
        alpha=float(args.alpha),
        timeout_s=int(args.timeout_s),
    )
    dt = time.time() - t0

    best = ranked[0]
    print(f"best_candidate={best.idx} score={best.score:.3f} pos_mean={best.positive_mean:.3f} neg_max={best.negative_max:.3f} in {dt:.1f}s")
    print()
    for s in best.sentences:
        print(s)

    if args.show_all:
        print("\n---\n")
        for r in ranked:
            sid = _stable_id(" ".join(r.sentences))
            print(f"cand={r.idx} id={sid} score={r.score:.3f} pos_mean={r.positive_mean:.3f} neg_max={r.negative_max:.3f}")
            for s in r.sentences:
                print(f"  - {s}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
