"""
Neural News MVP utilities (Stages 3 + 4).

Design goals:
- stdlib-only (no new deps)
- strict output shape (exactly 4 sentences) using OpenAI Responses JSON schema
- simple, resilient article text fetcher with RSS-summary fallback handled by caller
"""

from __future__ import annotations

import html as _html
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Sequence


WS_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\\1>")
P_RE = re.compile(r"(?is)<p[^>]*>(.*?)</p>")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


SENTENCE_SCRIPT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentences": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "string"},
        }
    },
    "required": ["sentences"],
}


def load_dotenv(path: str = ".env") -> None:
    """
    Minimal .env loader (no third-party deps).
    - Supports KEY=VALUE lines
    - Ignores comments and empty lines
    - Overrides env vars only when the current value is empty or looks like a placeholder
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
                cur = (os.environ.get(k) or "").strip()
                # Prefer explicit exported env vars, except when they're clearly placeholders.
                if cur and len(cur) >= 20 and cur not in {"...", "changeme", "YOUR_KEY"}:
                    continue
                os.environ[k] = v
    except FileNotFoundError:
        return


def _norm_text(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip())


def _strip_tags(s: str) -> str:
    return TAG_RE.sub(" ", s or "")


def fetch_article_text(url: str, *, timeout_s: int = 8, max_chars: int = 8000) -> str:
    """
    Best-effort article text fetch.
    - Pulls <p> blocks
    - Strips tags/script/style
    - Returns normalized text (may be empty)
    """
    url = (url or "").strip()
    if not url:
        return ""

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read(2_000_000)  # hard cap to avoid huge pages
            charset = None
            ct = resp.headers.get("Content-Type", "")
            if "charset=" in ct:
                charset = ct.split("charset=", 1)[1].split(";", 1)[0].strip() or None
    except Exception:
        return ""

    try:
        html = raw.decode(charset or "utf-8", errors="replace")
    except Exception:
        html = raw.decode("utf-8", errors="replace")

    html = SCRIPT_STYLE_RE.sub(" ", html)
    paras = P_RE.findall(html)
    cleaned: List[str] = []
    for p in paras:
        p = _strip_tags(p)
        p = _html.unescape(p)
        p = _norm_text(p)
        if not p:
            continue
        cleaned.append(p)
        if sum(len(x) for x in cleaned) >= max_chars:
            break

    text = _norm_text(" ".join(cleaned))
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def build_system_prompt(positives: Sequence[str], negatives: Sequence[str]) -> str:
    pos = [str(s).strip() for s in positives if str(s).strip()]
    neg = [str(s).strip() for s in negatives if str(s).strip()]
    pos_context = "\n".join(f"- {s}" for s in pos[-5:]) if pos else "None yet."
    neg_context = "\n".join(f"- {s}" for s in neg[-5:]) if neg else "None yet."

    # Keep this prompt intentionally "tight" and close to the user's spec.
    return f"""
Generate a 15–25 second spoken news segment (40–65 words, exactly 4 sentences).
Audience: 22–28 commuters who skip if bored in 3 seconds.
Voice: Texting a smart friend, not a news anchor. No big words.

STRICT BEHAVIORAL CONSTRAINTS:

REPLICATE THESE (LIKE):
{pos_context}

AVOID THESE (POISON):
{neg_context}

RULES:
- The Hook: Open with the most interesting fact, create a gap they need filled.
- The Details: Include 2+ specific details from the article (names, numbers, quotes, locations).
- The 'So What': End with stakes or implication.
- Banned phrases: "Stay tuned", "Sources say", "Interestingly", "Remains to be seen", "Hope remains", "Authorities are investigating".

OUTPUT:
- Script only.
- Exactly 4 sentences.
- No invented details.
""".strip()


def _http_post_json(url: str, api_key: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"HTTP {e.code} from {url}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error calling {url}: {e}") from e
    except socket.timeout as e:
        raise RuntimeError(f"Network timeout calling {url}: {e}") from e
    except TimeoutError as e:
        raise RuntimeError(f"Network timeout calling {url}: {e}") from e


def _is_timeout_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "timed out" in msg or "timeout" in msg


def _openai_responses_with_retry(
    *,
    api_key: str,
    payload: Dict[str, Any],
    timeout_s: int,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        # Gradually increase timeout each attempt.
        attempt_timeout = int(timeout_s + (attempt - 1) * 15)
        try:
            return _http_post_json(
                "https://api.openai.com/v1/responses",
                api_key=api_key,
                payload=payload,
                timeout_s=attempt_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_attempts or not _is_timeout_error(exc):
                raise
            # Small linear backoff between retries.
            time.sleep(0.6 * attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OpenAI call failed without an explicit exception")


def _responses_extract_output_text(resp: Dict[str, Any]) -> str:
    """
    Extract a plaintext payload from a Responses API response.

    The API commonly provides a convenience `output_text`, but it may be absent.
    In that case, walk `output[*].content[*]` and concatenate any text fragments.
    """
    raw = resp.get("output_text")
    if isinstance(raw, str) and raw.strip():
        return raw

    out_items = resp.get("output", [])
    if not isinstance(out_items, list):
        out_items = []

    parts: List[str] = []
    for it in out_items:
        if not isinstance(it, dict):
            continue
        content = it.get("content", [])
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            ctype = str(c.get("type", "") or "")
            # Common shapes:
            # - {"type":"output_text","text":"..."}
            # - {"type":"text","text":"..."}
            # - {"type":"json","json":{...}}
            if "text" in c and isinstance(c.get("text"), str) and c.get("text", "").strip():
                parts.append(str(c["text"]))
                continue
            if ctype == "json" and isinstance(c.get("json"), dict):
                parts.append(json.dumps(c["json"]))

    return "\n".join(p.strip() for p in parts if isinstance(p, str) and p.strip()).strip()


def _normalize_to_4_sentences(text: str) -> List[str]:
    """
    Best-effort splitter for plain-text fallback.
    Returns up to 4 non-empty sentences.
    """
    text = _norm_text(text)
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p and p.strip()]
    out: List[str] = []
    for p in parts:
        if p[-1] not in ".!?":
            p = p + "."
        out.append(p)
        if len(out) >= 4:
            break
    return out


def openai_generate_4_sentences(
    *,
    article_text: str,
    positives: Sequence[str],
    negatives: Sequence[str],
    api_key: str,
    model: str,
    temperature: float = 0.8,
    timeout_s: int = 25,
) -> List[str]:
    """
    Returns exactly 4 sentences (strings). Raises on failure.

    Uses OpenAI Responses API with strict JSON schema to avoid post-hoc sentence splitting.
    """
    article_text = _norm_text(article_text)
    if not article_text:
        raise ValueError("article_text is empty")

    instructions = build_system_prompt(positives=positives, negatives=negatives)
    user_input = f"Article Content:\n{article_text}"

    payload: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "temperature": float(temperature),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "script",
                "schema": SENTENCE_SCRIPT_SCHEMA,
                "strict": True,
            }
        },
    }

    t0 = time.time()
    out = _openai_responses_with_retry(
        api_key=api_key,
        payload=payload,
        timeout_s=timeout_s,
        max_attempts=3,
    )

    raw = _responses_extract_output_text(out)
    if not raw:
        err = out.get("error")
        if isinstance(err, dict) and err.get("message"):
            raise RuntimeError(f"OpenAI error: {err.get('message')}")
        raise RuntimeError(
            "OpenAI response missing text output "
            f"(status={out.get('status')!r} keys={list(out.keys())})"
        )
    try:
        parsed = json.loads(raw)
    except Exception as e:
        # Fallback: sometimes model returns plain text despite schema ask.
        fallback = _normalize_to_4_sentences(raw)
        if len(fallback) == 4:
            return fallback
        raise RuntimeError(f"Could not parse JSON from OpenAI output_text: {raw[:500]}") from e

    sents = parsed.get("sentences")
    if not isinstance(sents, list) or len(sents) != 4:
        raise RuntimeError(f"OpenAI JSON did not include 4 sentences: {raw[:500]}")

    cleaned = [_norm_text(str(s)) for s in sents]
    non_empty = [s for s in cleaned if s]
    if len(non_empty) == 4:
        cleaned = non_empty
    else:
        # Fallback #1: split whatever text we got.
        fallback_source = " ".join(non_empty) if non_empty else raw
        fallback = _normalize_to_4_sentences(fallback_source)
        if len(fallback) == 4:
            return fallback

        # Fallback #2: one retry without JSON schema, plain text only.
        retry_payload: Dict[str, Any] = {
            "model": model,
            "instructions": (
                instructions
                + "\n\nReturn plain text only: exactly 4 sentences (no JSON, no markdown)."
            ),
            "input": user_input,
            "temperature": max(0.2, min(0.6, float(temperature))),
        }
        retry_out = _openai_responses_with_retry(
            api_key=api_key,
            payload=retry_payload,
            timeout_s=timeout_s,
            max_attempts=2,
        )
        retry_raw = _responses_extract_output_text(retry_out)
        retry_sents = _normalize_to_4_sentences(retry_raw)
        if len(retry_sents) == 4:
            return retry_sents
        raise RuntimeError(
            "OpenAI returned empty/malformed sentences after fallback "
            f"(raw={raw[:220]!r}, retry={retry_raw[:220]!r})"
        )

    # Ensure sentence terminal punctuation for TTS cadence.
    out_sents: List[str] = []
    for s in cleaned:
        if s[-1] not in ".!?":
            s = s + "."
        out_sents.append(s)

    _ = t0  # keep hook for optional timing, but avoid prints in server
    return out_sents


def elevenlabs_tts_mp3(
    *,
    text: str,
    voice_id: str,
    api_key: str,
    model_id: str = "eleven_turbo_v2_5",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    timeout_s: int = 30,
) -> bytes:
    text = _norm_text(text)
    if not text:
        raise ValueError("TTS text is empty")
    voice_id = (voice_id or "").strip()
    if not voice_id:
        raise ValueError("Missing ElevenLabs voice_id")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": float(stability),
            "similarity_boost": float(similarity_boost),
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"ElevenLabs HTTP {e.code}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error calling ElevenLabs: {e}") from e
