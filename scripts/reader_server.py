#!/usr/bin/env python3
"""
Local sentence-by-sentence reader for collecting skip/bail-out data.

What you get:
- Shows a 4-sentence summary, one sentence at a time.
- Buttons: Next sentence, Skip article.
- Logs exactly where you skipped (which sentence index) + what you read.
- Optional: use the anti-vector reranker to generate the 4-sentence summary for each article.

Run:
  python3 scripts/reader_server.py --port 8001
Then open:
  http://localhost:8001

Outputs:
  data/reader_events.jsonl
  data/reader_state.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import random
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    # When running `python3 scripts/reader_server.py`, sys.path[0] is `.../scripts`,
    # so add repo root to allow `import scripts.*`.
    sys.path.insert(0, ROOT)

# Reuse the anti-vector generation/reranking helpers.
from scripts.anti_vector_rerank import _load_dotenv, openai_generate_candidates, rank_candidates

DATA_DIR = os.path.join(ROOT, "data")
EVENTS_PATH = os.path.join(DATA_DIR, "reader_events.jsonl")
STATE_PATH = os.path.join(DATA_DIR, "reader_state.json")


WS_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(s: str) -> str:
    # Minimal cleaning (RSS/HTML noise).
    try:
        import html as _html

        s = _html.unescape(s or "")
    except Exception:
        s = s or ""
    s = TAG_RE.sub(" ", s)
    return _norm(s)


def _now_utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _norm(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip())


def _json_write_atomic(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception:
        return default


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _split_into_4_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(_norm(text)) if p.strip()]
    if len(parts) >= 4:
        return parts[:4]
    # If too short, pad by splitting on commas / semicolons, else repeat last.
    if len(parts) == 1:
        more = [p.strip() for p in re.split(r"[;,]\s+", parts[0]) if p.strip()]
        parts = more[:4] if len(more) >= 2 else parts
    while len(parts) < 4:
        parts.append(parts[-1] if parts else "")
    return parts[:4]


def load_articles(csv_path: str, limit: int = 0) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            headline = _clean_text(row.get("Headline", "") or "")
            summary = _clean_text(row.get("Summary", "") or "")
            link = _norm(row.get("Link", "") or "")
            if link:
                try:
                    p = urllib.parse.urlsplit(link)
                    link = urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, "", ""))
                except Exception:
                    pass
            item_id = _norm(row.get("Item_ID", "") or "")
            if not headline and not summary:
                continue
            out.append({"item_id": item_id, "headline": headline, "summary": summary, "link": link})
            if limit and len(out) >= limit:
                break
    return out


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Reader Test</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: rgba(255,255,255,0.06);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --accent: #6df0c2;
      --danger: #ff6b6b;
      --btn: rgba(255,255,255,0.10);
      --btnHover: rgba(255,255,255,0.16);
    }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: radial-gradient(1200px 800px at 30% 20%, rgba(109,240,194,0.12), transparent 60%),
                  radial-gradient(900px 700px at 70% 80%, rgba(255,107,107,0.10), transparent 55%),
                  var(--bg);
      color: var(--text);
      height: 100vh;
      overflow: hidden;
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
      height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 14px;
    }
    .topbar {
      display: flex;
      gap: 10px;
      align-items: baseline;
      justify-content: space-between;
    }
    .title { font-weight: 650; letter-spacing: 0.2px; }
    .meta { color: var(--muted); font-size: 13px; }
    .card {
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 16px;
      padding: 18px 18px;
      backdrop-filter: blur(10px);
      box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    }
    .headline {
      font-size: 18px;
      line-height: 1.25;
      margin: 0 0 6px 0;
    }
    .headline a { color: var(--accent); text-decoration: none; }
    .headline a:hover { text-decoration: underline; }
    .sentenceBox {
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 10px;
      min-height: 280px;
    }
    .labelRow { display: flex; justify-content: space-between; color: var(--muted); font-size: 13px; }
    .sentence {
      font-size: 28px;
      line-height: 1.22;
      letter-spacing: -0.2px;
      margin: 0;
      align-self: center;
    }
    .controls {
      display: flex;
      gap: 12px;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
    }
    .btnRow { display: flex; gap: 10px; }
    button {
      appearance: none;
      border: 1px solid rgba(255,255,255,0.14);
      background: var(--btn);
      color: var(--text);
      padding: 12px 14px;
      border-radius: 12px;
      font-weight: 650;
      cursor: pointer;
      transition: background 120ms ease, transform 120ms ease;
    }
    button:hover { background: var(--btnHover); transform: translateY(-1px); }
    button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .danger { border-color: rgba(255,107,107,0.35); }
    .danger:hover { background: rgba(255,107,107,0.18); }
    .small { font-size: 13px; padding: 10px 12px; font-weight: 600; color: var(--muted); }
    .pill { font-size: 12px; padding: 6px 10px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.06); color: var(--muted); }
    .status { display: flex; gap: 8px; align-items: center; }
    .hint { color: var(--muted); font-size: 12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="title">Reader Test</div>
      <div class="meta">Reads are logged; Skip marks the bail-out sentence.</div>
    </div>

    <div class="card">
      <div class="headline"><a id="headlineLink" href="#" target="_blank" rel="noreferrer">Loading...</a></div>
      <div class="meta" id="submeta"></div>
    </div>

    <div class="card sentenceBox">
      <div class="labelRow">
        <div id="sentLabel">Sentence 1 of 4</div>
        <div class="status">
          <span class="pill" id="modePill">mode: ...</span>
          <span class="pill" id="sessionPill">session: ...</span>
        </div>
      </div>
      <p class="sentence" id="sentenceText">Loading...</p>
      <div class="hint">Keys: Space = next sentence, S = skip</div>
    </div>

    <div class="controls">
      <div class="btnRow">
        <button id="nextBtn">Next sentence</button>
        <button id="skipBtn" class="danger">Skip article</button>
      </div>
      <div class="btnRow">
        <button id="restartBtn" class="small">New session</button>
      </div>
    </div>
  </div>

<script>
  const api = {
    next: () => fetch('/api/next', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ session_id: getSessionId() }) }).then(r => r.json()),
    event: (payload) => fetch('/api/event', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) }).then(r => r.json()),
  };

  function uuid() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  function getSessionId() {
    let sid = localStorage.getItem('reader_session_id');
    if (!sid) {
      sid = uuid();
      localStorage.setItem('reader_session_id', sid);
    }
    return sid;
  }

  function resetSession() {
    localStorage.removeItem('reader_session_id');
    location.reload();
  }

  let current = null; // { article_id, headline, link, sentences, mode }
  let sentIdx = 0;

  const elHeadlineLink = document.getElementById('headlineLink');
  const elSubmeta = document.getElementById('submeta');
  const elSentLabel = document.getElementById('sentLabel');
  const elSentenceText = document.getElementById('sentenceText');
  const elModePill = document.getElementById('modePill');
  const elSessionPill = document.getElementById('sessionPill');
  const nextBtn = document.getElementById('nextBtn');
  const skipBtn = document.getElementById('skipBtn');
  const restartBtn = document.getElementById('restartBtn');

  elSessionPill.textContent = 'session: ' + getSessionId().slice(0, 8);

  function render() {
    if (!current) return;
    elHeadlineLink.textContent = current.headline || '(no headline)';
    elHeadlineLink.href = current.link || '#';
    elSubmeta.textContent = 'article_id: ' + current.article_id + ' | ' + (current.source_hint || '');
    elModePill.textContent = 'mode: ' + current.mode;
    elSentLabel.textContent = `Sentence ${sentIdx + 1} of 4`;
    elSentenceText.textContent = current.sentences[sentIdx] || '';
    nextBtn.textContent = (sentIdx === 3) ? 'Next article' : 'Next sentence';
  }

  async function loadNextArticle() {
    sentIdx = 0;
    current = await api.next();
    render();
    await api.event({
      session_id: getSessionId(),
      ts: new Date().toISOString(),
      action: 'start_article',
      article_id: current.article_id,
    });
  }

  async function logRead(idx) {
    await api.event({
      session_id: getSessionId(),
      ts: new Date().toISOString(),
      action: 'read_sentence',
      article_id: current.article_id,
      sentence_index: idx,
      sentence: current.sentences[idx] || ''
    });
  }

  async function onNext() {
    if (!current) return;
    await logRead(sentIdx);
    if (sentIdx < 3) {
      sentIdx += 1;
      render();
      return;
    }
    await api.event({
      session_id: getSessionId(),
      ts: new Date().toISOString(),
      action: 'complete_article',
      article_id: current.article_id,
    });
    await loadNextArticle();
  }

  async function onSkip() {
    if (!current) return;
    await api.event({
      session_id: getSessionId(),
      ts: new Date().toISOString(),
      action: 'skip_article',
      article_id: current.article_id,
      sentence_index: sentIdx,
      sentence: current.sentences[sentIdx] || ''
    });
    await loadNextArticle();
  }

  nextBtn.addEventListener('click', onNext);
  skipBtn.addEventListener('click', onSkip);
  restartBtn.addEventListener('click', resetSession);

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); onNext(); }
    if (e.key === 's' || e.key === 'S') { e.preventDefault(); onSkip(); }
  });

  loadNextArticle();
</script>
</body>
</html>
"""


class App:
    def __init__(self, articles: List[Dict[str, str]], use_openai: bool, model: str, embed_model: str):
        self.articles = articles
        self.use_openai = use_openai
        self.model = model
        self.embed_model = embed_model
        self.lock = threading.Lock()
        self.state = _load_json(STATE_PATH, default={"sessions": {}})

    def _save_state(self) -> None:
        _json_write_atomic(STATE_PATH, self.state)

    def _get_session(self, session_id: str) -> Dict[str, Any]:
        sess = self.state.setdefault("sessions", {}).setdefault(session_id, {})
        sess.setdefault("created_utc", _now_utc_iso())
        sess.setdefault("seen_article_ids", [])
        sess.setdefault("pos_anchor", "")
        sess.setdefault("neg_anchor", "")
        sess.setdefault("last_article", {})
        return sess

    def _pick_article(self, sess: Dict[str, Any]) -> Dict[str, str]:
        seen = set(sess.get("seen_article_ids") or [])
        # Deterministic-ish per session: random seeded once.
        seed = sess.get("seed")
        if seed is None:
            seed = random.randint(1, 10_000_000)
            sess["seed"] = seed
        rng = random.Random(int(seed) + len(seen))

        # Try a few random picks; fall back to first unseen.
        for _ in range(50):
            a = rng.choice(self.articles)
            aid = a.get("item_id") or ""
            if aid and aid in seen:
                continue
            return a
        for a in self.articles:
            aid = a.get("item_id") or ""
            if aid and aid not in seen:
                return a
        return rng.choice(self.articles)

    def _make_4_sentences(self, *, facts: str, pos_anchor: str, neg_anchor: str) -> Tuple[List[str], str]:
        """
        Returns (sentences, mode).
        mode: "anti_vector" or "simple_split"
        """
        facts = _norm(facts)
        if not facts:
            return _split_into_4_sentences("No content available."), "simple_split"

        if not self.use_openai:
            return _split_into_4_sentences(facts), "simple_split"

        # If we don't have anchors yet, fall back to simple generation (no rerank).
        if not _norm(pos_anchor) or not _norm(neg_anchor):
            # Seed with defaults so rerank has something, but keep it light.
            pos_anchor = _norm(pos_anchor) or "Clear, direct explanation with concrete nouns."
            neg_anchor = _norm(neg_anchor) or "Overly dense jargon and lots of percentages."

        candidates = openai_generate_candidates(
            facts=facts,
            model=self.model,
            api_key=os.environ["OPENAI_API_KEY"],
            n_candidates=6,
            temperature=0.85,
            timeout_s=60,
        )
        ranked = rank_candidates(
            candidates=candidates,
            pos_anchor=pos_anchor,
            neg_anchor=neg_anchor,
            api_key=os.environ["OPENAI_API_KEY"],
            embed_model=self.embed_model,
            alpha=1.2,
            timeout_s=60,
        )
        return ranked[0].sentences, "anti_vector"

    def next_article(self, session_id: str) -> Dict[str, Any]:
        with self.lock:
            sess = self._get_session(session_id)
            art = self._pick_article(sess)
            aid = art.get("item_id") or ""
            headline = art.get("headline") or ""
            summary = art.get("summary") or ""
            link = art.get("link") or ""

            facts = headline
            if summary and summary not in headline:
                facts = f"{headline} {summary}".strip()

            sents, mode = self._make_4_sentences(
                facts=facts,
                pos_anchor=str(sess.get("pos_anchor") or ""),
                neg_anchor=str(sess.get("neg_anchor") or ""),
            )

            payload = {
                "article_id": aid or f"noid-{random.randint(1,9999999)}",
                "headline": headline or "(no headline)",
                "link": link or "#",
                "sentences": sents,
                "mode": mode,
                "source_hint": "from scraped_articles.csv",
            }

            sess["last_article"] = payload
            if payload["article_id"] not in sess["seen_article_ids"]:
                sess["seen_article_ids"].append(payload["article_id"])
                sess["seen_article_ids"] = sess["seen_article_ids"][-2000:]

            self._save_state()
            return payload

    def on_event(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(ev.get("session_id") or "").strip()
        if not session_id:
            return {"ok": False, "error": "missing session_id"}
        with self.lock:
            sess = self._get_session(session_id)
            sess["last_seen_utc"] = _now_utc_iso()
            action = str(ev.get("action") or "")

            # Update anchors based on skip behavior.
            if action == "skip_article":
                last = sess.get("last_article") or {}
                sents = last.get("sentences") or []
                idx = int(ev.get("sentence_index") or 0)
                idx = max(0, min(3, idx))
                pos = " ".join([str(x) for x in sents[:idx]]).strip()
                neg = str(sents[idx] if idx < len(sents) else "")[:600].strip()
                if pos:
                    sess["pos_anchor"] = pos
                if neg:
                    sess["neg_anchor"] = neg
            elif action == "complete_article":
                last = sess.get("last_article") or {}
                sents = last.get("sentences") or []
                pos = " ".join([str(x) for x in sents]).strip()
                if pos:
                    sess["pos_anchor"] = pos
                # Do not overwrite neg_anchor on a full read.

            self._save_state()
            _append_jsonl(EVENTS_PATH, ev)
            return {"ok": True}


def _read_body_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    try:
        n = int(handler.headers.get("Content-Length", "0") or "0")
    except Exception:
        n = 0
    raw = handler.rfile.read(n) if n > 0 else b"{}"
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return {}


class Handler(BaseHTTPRequestHandler):
    app: App

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path == "/api/next":
            body = _read_body_json(self)
            sid = str(body.get("session_id") or "").strip()
            if not sid:
                self._send(400, b'{"error":"missing session_id"}', "application/json; charset=utf-8")
                return
            try:
                payload = self.app.next_article(sid)
                self._send(200, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
            except Exception as e:
                err = {"error": str(e)}
                self._send(500, json.dumps(err).encode("utf-8"), "application/json; charset=utf-8")
            return

        if self.path == "/api/event":
            body = _read_body_json(self)
            try:
                out = self.app.on_event(body)
                code = 200 if out.get("ok") else 400
                self._send(code, json.dumps(out).encode("utf-8"), "application/json; charset=utf-8")
            except Exception as e:
                err = {"ok": False, "error": str(e)}
                self._send(500, json.dumps(err).encode("utf-8"), "application/json; charset=utf-8")
            return

        self._send(404, b"not found", "text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep server quiet; comment this out if you want access logs.
        return


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--csv", default="")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of articles loaded.")
    ap.add_argument("--use-openai", action="store_true", help="Generate/rerank summaries using OpenAI.")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    args = ap.parse_args(argv)

    _load_dotenv()
    if args.use_openai:
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is not set. Put `OPENAI_API_KEY=...` in .env or export it in your shell.")
        if len(api_key) < 20 or not api_key.startswith("sk-"):
            raise SystemExit("OPENAI_API_KEY looks invalid. Double-check your .env file (no extra quotes/typos).")
        os.environ["OPENAI_API_KEY"] = api_key

    csv_path = args.csv.strip()
    if not csv_path:
        p1 = os.path.join(DATA_DIR, "scraped_articles.csv")
        p2 = os.path.join(DATA_DIR, "scraped_articles_backup.csv")
        csv_path = p1 if os.path.exists(p1) else p2

    articles = load_articles(csv_path, limit=int(args.limit))
    if not articles:
        raise SystemExit(f"Loaded 0 articles from {csv_path}")

    app = App(articles=articles, use_openai=bool(args.use_openai), model=args.model, embed_model=args.embed_model)
    Handler.app = app

    httpd = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(f"server http://{args.host}:{args.port} (articles={len(articles)} mode={'openai' if args.use_openai else 'simple'})")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
