from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from model import (
    ITEM_COL_ITEM_ID,
    ITEM_COL_POPULARITY_BUCKET,
    ITEM_COL_RECENCY_BUCKET,
    ITEM_COL_TOPIC_CLUSTER_ID,
    ITEM_COL_TOPIC_ID,
    ITEM_COL_VIBE_ID,
    NUM_ITEM_FEATURES,
    GenerativeTwoTower,
)

app = FastAPI()

ROOT = Path(__file__).resolve().parent


def _resolve_path_from_env(env_key: str, default_path: Path) -> Path:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return default_path
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p)


DATA_PATH = _resolve_path_from_env("DATA_PATH", ROOT / "data" / "scraped_articles.csv")
INDEX_PATH = _resolve_path_from_env("INDEX_PATH", ROOT / "index.html")
WARM_START_PATH = _resolve_path_from_env(
    "WARM_START_PATH",
    ROOT / "outputs" / "train_real_warm_start.pt",
)
USAGE_LOG_PATH = _resolve_path_from_env(
    "USAGE_LOG_PATH",
    ROOT / "data" / "usage_interactions.csv",
)
SESSION_STATE_PATH = _resolve_path_from_env(
    "SESSION_STATE_PATH",
    ROOT / "data" / "live_session_state.json",
)

DEVICE = torch.device("cpu")
EVENT_SEQ_LEN = 30
EVENT_VOCAB_SIZE = 4096
LIVE_LR = 0.003
CANDIDATE_POOL = 160
FEED_K = 20
DEDUP_CANDIDATE_LIMIT = 50
DEDUP_SIM_THRESHOLD = 0.35
RECENT_SEEN_WINDOW = 60
RECALL_SIZE = int(os.getenv("RECALL_SIZE", "260"))
PRE_RANK_SIZE = int(os.getenv("PRE_RANK_SIZE", "96"))
RANK_POOL_SIZE = int(os.getenv("RANK_POOL_SIZE", "40"))
RECALL_SCAN_LIMIT = int(os.getenv("RECALL_SCAN_LIMIT", "5000"))
MMR_DIVERSITY_LAMBDA = float(os.getenv("MMR_DIVERSITY_LAMBDA", "0.18"))
EXPLORATION_EPS_START = float(os.getenv("EXPLORATION_EPS_START", "0.18"))
EXPLORATION_EPS_MIN = float(os.getenv("EXPLORATION_EPS_MIN", "0.04"))
EXPLORATION_DECAY_STEPS = float(os.getenv("EXPLORATION_DECAY_STEPS", "220.0"))
MAX_PERSISTED_SESSIONS = int(os.getenv("MAX_PERSISTED_SESSIONS", "3000"))
SESSION_COOKIE = "vibe_session_id"
SKIP_BASELINE_MIN = 12
SKIP_STD_FLOOR = 0.35
SKIP_HISTORY_MAX = 240
ONLINE_UPDATE_MODE = os.getenv("ONLINE_UPDATE_MODE", "private").strip().lower()
PRIVATE_USER_LR = float(os.getenv("PRIVATE_USER_LR", "0.03"))
PRIVATE_USER_MAX_NORM = float(os.getenv("PRIVATE_USER_MAX_NORM", "4.0"))
PRIVATE_USER_GRAD_CLIP = float(os.getenv("PRIVATE_USER_GRAD_CLIP", "5.0"))

if ONLINE_UPDATE_MODE not in {"private", "global", "none"}:
    print(
        f"⚠️ Invalid ONLINE_UPDATE_MODE={ONLINE_UPDATE_MODE!r}; "
        "falling back to 'private'."
    )
    ONLINE_UPDATE_MODE = "private"

SCORE_WEIGHTS = {
    "finish": 1.2,
    "like": 1.0,
    "share": 1.2,
    "rewatch": 1.1,
    "fast_skip": 0.8,
    "watch_time": 1.1,
}

state_lock = threading.Lock()

USAGE_LOG_COLUMNS = [
    "timestamp_utc",
    "source",
    "run_id",
    "session_id",
    "step",
    "user_step",
    "user_type",
    "item_id",
    "topic_id",
    "topic_name",
    "vibe_id",
    "vibe_style",
    "cluster_id",
    "headline",
    "reward",
    "regret",
    "epsilon",
    "decision",
    "pred_score",
    "listen_ms",
    "total_ms",
    "liked",
    "shared",
    "rewinded",
    "fast_skip",
    "finished",
    "interest",
]


def append_usage_log_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=USAGE_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in USAGE_LOG_COLUMNS})


def load_persisted_session_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_sessions = payload.get("sessions", payload)
            if isinstance(raw_sessions, dict):
                out = {}
                for sid, state in raw_sessions.items():
                    if isinstance(sid, str) and isinstance(state, dict):
                        out[sid] = state
                return out
    except Exception as exc:
        print(f"⚠️ Failed to load persisted session state from {path}: {exc}")
    return {}


def save_persisted_session_map(path: Path, session_map: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sessions": session_map,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp_path.replace(path)


def extract_domain(url: str) -> str:
    clean = str(url).replace("https://", "").replace("http://", "")
    return clean.split("/")[0].replace("www.", "").strip().lower() or "unknown"


def entity_key(headline: str) -> str:
    text = str(headline).lower()
    if any(k in text for k in ("openai", "chatgpt", "gpt")):
        return "openai"
    if any(k in text for k in ("google", "android", "youtube", "alphabet")):
        return "google"
    if any(k in text for k in ("apple", "iphone", "mac", "airpods")):
        return "apple"
    if any(k in text for k in ("meta", "facebook", "instagram", "whatsapp")):
        return "meta"
    if any(k in text for k in ("microsoft", "windows", "azure")):
        return "microsoft"
    if any(k in text for k in ("amazon", "aws", "alexa")):
        return "amazon"
    return "other"


def event_key(headline: str) -> str:
    text = str(headline).lower()
    if any(k in text for k in ("ai", "model", "agent", "llm")):
        return "ai_release"
    if any(k in text for k in ("raises", "series", "funding", "acquire")):
        return "funding"
    if any(k in text for k in ("earnings", "revenue", "profit")):
        return "earnings"
    if any(k in text for k in ("law", "policy", "regulation", "sanction")):
        return "regulation"
    if any(k in text for k in ("hack", "breach", "cyber", "security")):
        return "security"
    if any(k in text for k in ("launch", "announces", "unveils", "introduces")):
        return "product_launch"
    return "other"


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Topic_Name" not in out.columns:
        out["Topic_Name"] = out["Link"].apply(extract_domain)
    if "Topic_ID" not in out.columns:
        out["Topic_ID"], _ = pd.factorize(out["Topic_Name"])

    style_col = "Suggested_Style" if "Suggested_Style" in out.columns else "Vibe_Style"
    if style_col not in out.columns:
        out["Vibe_Style"] = "Standard Reporting"
    else:
        out["Vibe_Style"] = out[style_col].fillna("Standard Reporting").astype(str)
    if "Vibe_ID" not in out.columns:
        out["Vibe_ID"], _ = pd.factorize(out["Vibe_Style"])

    if "Topic_Cluster_ID" not in out.columns:
        out["Topic_Cluster_ID"] = (
            out["Headline"].fillna("").astype(str).apply(lambda s: abs(hash(s.split(" ")[0])) % 32)
        )

    if "Entity_ID" not in out.columns:
        out["Entity_Key"] = out["Headline"].fillna("").apply(entity_key)
        out["Entity_ID"], _ = pd.factorize(out["Entity_Key"])

    if "Event_ID" not in out.columns:
        out["Event_Key"] = out["Headline"].fillna("").apply(event_key)
        out["Event_ID"], _ = pd.factorize(out["Event_Key"])

    if "Recency_Bucket" not in out.columns:
        out["Recency_Bucket"] = 2
    if "Popularity_Bucket" not in out.columns:
        freq = out["Topic_Name"].value_counts()
        out["_freq"] = out["Topic_Name"].map(freq).astype(float)
        out["Popularity_Bucket"] = pd.qcut(
            out["_freq"].rank(method="first"),
            q=4,
            labels=False,
            duplicates="drop",
        ).astype(int)
        out = out.drop(columns=["_freq"])

    out["Item_ID"] = range(len(out))
    out["id"] = out["Item_ID"].astype(int)

    for col in [
        "Item_ID",
        "Topic_ID",
        "Vibe_ID",
        "Topic_Cluster_ID",
        "Entity_ID",
        "Event_ID",
        "Recency_Bucket",
        "Popularity_Bucket",
    ]:
        out[col] = out[col].fillna(0).astype(int).clip(lower=0)

    return out


def stable_event_token(raw: str) -> int:
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return 1 + (int(digest[:8], 16) % (EVENT_VOCAB_SIZE - 1))


def build_event_token(item_id: int, topic_id: int, vibe_id: int, dwell: float, step: int) -> int:
    ratio = max(0.0, min(1.0, dwell / 10.0))
    if ratio < 0.2:
        dwell_bucket = 0
    elif ratio < 0.5:
        dwell_bucket = 1
    elif ratio < 0.8:
        dwell_bucket = 2
    else:
        dwell_bucket = 3
    raw = f"i{item_id}|t{topic_id}|v{vibe_id}|dw{dwell_bucket}|sx{min(9, step//10)}"
    return stable_event_token(raw)


def reward_from_watch_time(dwell_seconds: float) -> float:
    dwell = max(0.0, min(10.0, dwell_seconds))
    return -10.0 + (dwell / 10.0) * 20.0


def adaptive_skip_reward(dwell_seconds: float, skip_history: list[float]) -> float:
    dwell = max(0.0, min(10.0, dwell_seconds))
    watch_ratio = dwell / 10.0
    absolute = -8.0 + (watch_ratio * 16.0)
    if len(skip_history) < SKIP_BASELINE_MIN:
        return max(-10.0, min(10.0, absolute))

    mean_skip = sum(skip_history) / len(skip_history)
    variance = sum((x - mean_skip) ** 2 for x in skip_history) / len(skip_history)
    std_skip = max(SKIP_STD_FLOOR, variance ** 0.5)

    # Positive z-score means this skip is later than your normal skip window.
    z = (dwell - mean_skip) / std_skip
    relative = math.tanh(z / 1.35) * 9.0
    reward = 0.75 * relative + 0.25 * absolute
    return max(-10.0, min(10.0, reward))


def reward_from_interaction(reason: str, dwell_seconds: float, skip_history: list[float]) -> float:
    if reason == "auto":
        return 10.0
    if reason == "skip":
        return adaptive_skip_reward(dwell_seconds, skip_history)
    return reward_from_watch_time(dwell_seconds)


def reward_to_norm(reward: float) -> float:
    return max(0.05, min(0.95, (reward + 10.0) / 20.0))


def get_content_fingerprint(title: str, description: str) -> set[str]:
    """Combine title + description into a normalized token set for dedup."""
    full_text = f"{title} {description}".lower()
    return set(full_text.split())


def calculate_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    if union == 0:
        return 0.0
    return inter / union


if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

raw_inventory = pd.read_csv(DATA_PATH)
inventory = ensure_feature_columns(raw_inventory).reset_index(drop=True)

headline_col = "Headline" if "Headline" in inventory.columns else "title"
link_col = "Link" if "Link" in inventory.columns else "link"
summary_col = "Summary" if "Summary" in inventory.columns else "description"
domain_col = "Topic_Name"

item_features = torch.tensor(
    inventory[
        [
            "Item_ID",
            "Topic_ID",
            "Vibe_ID",
            "Topic_Cluster_ID",
            "Entity_ID",
            "Event_ID",
            "Recency_Bucket",
            "Popularity_Bucket",
        ]
    ].values,
    dtype=torch.long,
    device=DEVICE,
)

item_fingerprints: list[set[str]] = []
for i in range(len(inventory)):
    row = inventory.iloc[i]
    title = str(row[headline_col])
    description = str(row[summary_col]) if summary_col in inventory.columns else ""
    item_fingerprints.append(get_content_fingerprint(title=title, description=description))

if item_features.size(1) != NUM_ITEM_FEATURES:
    raise ValueError(f"Expected {NUM_ITEM_FEATURES} feature columns, got {item_features.size(1)}")

warm_ckpt = None
warm_meta: dict[str, int] = {}
if WARM_START_PATH.exists():
    loaded = torch.load(WARM_START_PATH, map_location=DEVICE)
    if isinstance(loaded, dict):
        warm_ckpt = loaded
        meta_obj = loaded.get("meta", {})
        if isinstance(meta_obj, dict):
            warm_meta = {k: int(v) for k, v in meta_obj.items() if isinstance(v, int | float)}

def dim_with_warm(data_dim: int, meta_key: str) -> int:
    warm_dim = int(warm_meta.get(meta_key, data_dim))
    return max(data_dim, warm_dim, 1)

model = GenerativeTwoTower(
    num_items=dim_with_warm(int(inventory["Item_ID"].max()) + 1, "num_items"),
    num_topics=dim_with_warm(int(inventory["Topic_ID"].max()) + 1, "num_topics"),
    num_vibes=dim_with_warm(int(inventory["Vibe_ID"].max()) + 1, "num_vibes"),
    num_topic_clusters=dim_with_warm(
        int(inventory["Topic_Cluster_ID"].max()) + 1, "num_topic_clusters"
    ),
    num_entities=dim_with_warm(int(inventory["Entity_ID"].max()) + 1, "num_entities"),
    num_events=dim_with_warm(int(inventory["Event_ID"].max()) + 1, "num_events"),
    num_recency_buckets=dim_with_warm(
        int(inventory["Recency_Bucket"].max()) + 1, "num_recency_buckets"
    ),
    num_popularity_buckets=dim_with_warm(
        int(inventory["Popularity_Bucket"].max()) + 1, "num_popularity_buckets"
    ),
    event_vocab_size=dim_with_warm(EVENT_VOCAB_SIZE, "event_vocab_size"),
).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LIVE_LR) if ONLINE_UPDATE_MODE == "global" else None
if ONLINE_UPDATE_MODE != "global":
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
bce = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()

if warm_ckpt is not None and "model_state_dict" in warm_ckpt:
    source_state = warm_ckpt["model_state_dict"]
    target_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, tensor in source_state.items():
        if key in target_state and hasattr(tensor, "shape") and tensor.shape == target_state[key].shape:
            compatible[key] = tensor
        else:
            skipped.append(key)
    model.load_state_dict(compatible, strict=False)
    print(
        f"♻️ Loaded warm-start model from {WARM_START_PATH} "
        f"(loaded={len(compatible)} tensors, skipped={len(skipped)})"
    )

if ONLINE_UPDATE_MODE == "global":
    print("⚙️ Online update mode: GLOBAL (shared model weights updated per interaction)")
elif ONLINE_UPDATE_MODE == "private":
    print("⚙️ Online update mode: PRIVATE (per-session user state adaptation only)")
else:
    print("⚙️ Online update mode: NONE (no online gradient updates)")


def forward_heads_from_user_state(
    user_state: torch.Tensor,
    item_batch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Run item tower + interaction heads using an externally supplied user state.
    This enables private per-session adaptation without mutating shared weights.
    """
    if user_state.dim() == 1:
        user_state = user_state.unsqueeze(0)
    item_vec = model.encode_item(item_batch)
    if user_state.size(0) == 1 and item_vec.size(0) > 1:
        user_state = user_state.expand(item_vec.size(0), -1)
    elif user_state.size(0) != item_vec.size(0):
        raise ValueError(
            f"user_state batch={user_state.size(0)} must match item batch={item_vec.size(0)}"
        )

    interaction = torch.cat(
        [
            user_state,
            item_vec,
            user_state * item_vec,
            torch.abs(user_state - item_vec),
        ],
        dim=1,
    )
    hidden = model.interaction_tower(interaction)
    return {
        "finish_logit": model.finish_head(hidden),
        "like_logit": model.like_head(hidden),
        "share_logit": model.share_head(hidden),
        "rewatch_logit": model.rewatch_head(hidden),
        "fast_skip_logit": model.fast_skip_head(hidden),
        "watch_time_raw": model.watch_time_head(hidden),
    }


class SessionState:
    def __init__(self) -> None:
        self.event_seq = deque([0] * EVENT_SEQ_LEN, maxlen=EVENT_SEQ_LEN)
        self.recent_seen = deque(maxlen=RECENT_SEEN_WINDOW)
        self.skip_dwells = deque(maxlen=SKIP_HISTORY_MAX)
        self.step = 0
        self.impressions = 0
        self.avg_reward = 0.0
        self.last_batch: list[int] = []
        self.user_delta = torch.zeros(model.user_hidden_dim, dtype=torch.float32, device=DEVICE)
        self.updated_at = time.time()

    def update_reward(self, reward: float) -> None:
        self.impressions += 1
        self.avg_reward += (reward - self.avg_reward) / max(1, self.impressions)
        self.updated_at = time.time()

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        state = cls()

        seq = data.get("event_seq", [])
        if isinstance(seq, list):
            cleaned = [int(x) for x in seq if isinstance(x, int | float)]
            cleaned = cleaned[-EVENT_SEQ_LEN:]
            if len(cleaned) < EVENT_SEQ_LEN:
                cleaned = ([0] * (EVENT_SEQ_LEN - len(cleaned))) + cleaned
            state.event_seq = deque(cleaned, maxlen=EVENT_SEQ_LEN)

        seen = data.get("recent_seen", [])
        if isinstance(seen, list):
            cleaned_seen = [int(x) for x in seen if isinstance(x, int | float)]
            state.recent_seen = deque(cleaned_seen[-RECENT_SEEN_WINDOW:], maxlen=RECENT_SEEN_WINDOW)

        skips = data.get("skip_dwells", [])
        if isinstance(skips, list):
            cleaned_skips = [float(max(0.0, min(10.0, float(x)))) for x in skips if isinstance(x, int | float)]
            state.skip_dwells = deque(cleaned_skips[-SKIP_HISTORY_MAX:], maxlen=SKIP_HISTORY_MAX)

        if isinstance(data.get("step"), int | float):
            state.step = max(0, int(data["step"]))
        if isinstance(data.get("impressions"), int | float):
            state.impressions = max(0, int(data["impressions"]))
        if isinstance(data.get("avg_reward"), int | float):
            state.avg_reward = float(data["avg_reward"])
        if isinstance(data.get("updated_at"), int | float):
            state.updated_at = float(data["updated_at"])

        raw_delta = data.get("user_delta", [])
        if isinstance(raw_delta, list):
            delta_vals = [float(x) for x in raw_delta if isinstance(x, int | float)]
            delta = torch.tensor(delta_vals, dtype=torch.float32, device=DEVICE)
            if delta.numel() < model.user_hidden_dim:
                pad = torch.zeros(model.user_hidden_dim - delta.numel(), dtype=torch.float32, device=DEVICE)
                delta = torch.cat([delta, pad], dim=0)
            elif delta.numel() > model.user_hidden_dim:
                delta = delta[: model.user_hidden_dim]
            state.user_delta = delta

        return state

    def to_dict(self) -> dict[str, object]:
        return {
            "event_seq": list(self.event_seq),
            "recent_seen": list(self.recent_seen),
            "skip_dwells": list(self.skip_dwells),
            "step": int(self.step),
            "impressions": int(self.impressions),
            "avg_reward": float(self.avg_reward),
            "user_delta": [float(x) for x in self.user_delta.detach().cpu().tolist()],
            "updated_at": float(self.updated_at),
        }


sessions: dict[str, SessionState] = {}
persisted_session_map = load_persisted_session_map(SESSION_STATE_PATH)


def prune_persisted_sessions() -> None:
    if len(persisted_session_map) <= MAX_PERSISTED_SESSIONS:
        return
    ordered = sorted(
        persisted_session_map.items(),
        key=lambda kv: float(kv[1].get("updated_at", 0.0)),
        reverse=True,
    )
    trimmed = dict(ordered[:MAX_PERSISTED_SESSIONS])
    persisted_session_map.clear()
    persisted_session_map.update(trimmed)


def persist_session_state(session_id: str, session: SessionState) -> None:
    persisted_session_map[session_id] = session.to_dict()
    prune_persisted_sessions()
    save_persisted_session_map(SESSION_STATE_PATH, persisted_session_map)


def get_session_id(request: Request) -> str:
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        return sid
    return str(uuid.uuid4())


def get_or_create_session(session_id: str) -> SessionState:
    if session_id not in sessions:
        raw_state = persisted_session_map.get(session_id)
        if isinstance(raw_state, dict):
            sessions[session_id] = SessionState.from_dict(raw_state)
        else:
            sessions[session_id] = SessionState()
    return sessions[session_id]


def exploration_epsilon(step: int) -> float:
    decay = math.exp(-max(0, step) / max(1.0, EXPLORATION_DECAY_STEPS))
    return max(EXPLORATION_EPS_MIN, EXPLORATION_EPS_START * decay)


def select_recall_candidates(
    available_indices: list[int],
    adapted_user_state: torch.Tensor,
) -> list[int]:
    if not available_indices:
        return []

    scan_indices = available_indices
    if len(scan_indices) > RECALL_SCAN_LIMIT:
        scan_indices = random.sample(scan_indices, RECALL_SCAN_LIMIT)
    recall_target = min(RECALL_SIZE, len(scan_indices))
    if recall_target <= 0:
        return []

    scan_tensor = item_features[scan_indices]
    with torch.no_grad():
        item_vectors = model.encode_item(scan_tensor)
        nn_scores = torch.mv(item_vectors, adapted_user_state)
        nn_scores = nn_scores + 0.01 * torch.randn_like(nn_scores)

    n_random = min(len(scan_indices), max(1, int(recall_target * 0.35)))
    n_nn = min(len(scan_indices), max(1, int(recall_target * 0.35)))
    n_pop = min(len(scan_indices), max(1, int(recall_target * 0.15)))
    n_fresh = min(len(scan_indices), max(1, recall_target - n_random - n_nn - n_pop))

    random_part = random.sample(scan_indices, n_random)

    _, nn_local = torch.topk(nn_scores, k=n_nn)
    nn_part = [scan_indices[i] for i in nn_local.tolist()]

    pop_scores = scan_tensor[:, ITEM_COL_POPULARITY_BUCKET].float()
    _, pop_local = torch.topk(pop_scores, k=n_pop)
    pop_part = [scan_indices[i] for i in pop_local.tolist()]

    fresh_scores = -scan_tensor[:, ITEM_COL_RECENCY_BUCKET].float()
    _, fresh_local = torch.topk(fresh_scores, k=n_fresh)
    fresh_part = [scan_indices[i] for i in fresh_local.tolist()]

    merged: list[int] = []
    seen = set()
    for idx in random_part + nn_part + pop_part + fresh_part:
        if idx not in seen:
            merged.append(idx)
            seen.add(idx)
        if len(merged) >= recall_target:
            break

    if len(merged) < recall_target:
        leftovers = [i for i in scan_indices if i not in seen]
        merged.extend(leftovers[: recall_target - len(merged)])
    return merged


def pre_rank_candidates(
    session: SessionState,
    recall_indices: list[int],
    adapted_user_state: torch.Tensor,
) -> list[tuple[int, float]]:
    if not recall_indices:
        return []

    recall_features = item_features[recall_indices]
    with torch.no_grad():
        heads = forward_heads_from_user_state(adapted_user_state, recall_features)
        base_scores = model.score_from_heads(heads, weights=SCORE_WEIGHTS).squeeze(1).cpu().tolist()

    recent_ids = list(session.recent_seen)
    topic_counts = Counter(int(item_features[i, ITEM_COL_TOPIC_ID].item()) for i in recent_ids)
    cluster_counts = Counter(int(item_features[i, ITEM_COL_TOPIC_CLUSTER_ID].item()) for i in recent_ids)

    diversity_tax_topic_id = None
    if len(recent_ids) >= 5:
        last_five_topic_ids = [int(item_features[i, ITEM_COL_TOPIC_ID].item()) for i in recent_ids[-5:]]
        if len(set(last_five_topic_ids)) == 1:
            diversity_tax_topic_id = last_five_topic_ids[0]

    adjusted: list[tuple[int, float]] = []
    for local_i, idx in enumerate(recall_indices):
        feat = item_features[idx]
        topic_id = int(feat[ITEM_COL_TOPIC_ID].item())
        cluster_id = int(feat[ITEM_COL_TOPIC_CLUSTER_ID].item())
        recency_bucket = int(feat[ITEM_COL_RECENCY_BUCKET].item())
        base = float(base_scores[local_i])

        novelty_bonus = 0.10 if topic_counts[topic_id] == 0 else 0.0
        cluster_bonus = 0.06 if cluster_counts[cluster_id] == 0 else 0.0
        freshness_bonus = 0.08 * max(0, 4 - recency_bucket) / 4.0
        fatigue_penalty = 0.08 * topic_counts[topic_id]
        adjusted_score = base + novelty_bonus + cluster_bonus + freshness_bonus - fatigue_penalty

        if diversity_tax_topic_id is not None and topic_id == diversity_tax_topic_id:
            adjusted_score *= 0.35

        adjusted.append((idx, adjusted_score))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted[: min(PRE_RANK_SIZE, len(adjusted))]


def final_rank_candidates(
    session: SessionState,
    pre_ranked: list[tuple[int, float]],
) -> list[int]:
    if not pre_ranked:
        return []

    rank_pool = pre_ranked[: min(RANK_POOL_SIZE, len(pre_ranked))]
    selected: list[int] = []
    selected_fps: list[set[str]] = []

    while len(selected) < min(FEED_K, len(rank_pool)):
        best_idx = None
        best_value = -float("inf")
        for idx, score in rank_pool:
            if idx in selected:
                continue
            fp = item_fingerprints[idx]
            diversity_penalty = 0.0
            if selected_fps:
                diversity_penalty = max(calculate_similarity(fp, seen_fp) for seen_fp in selected_fps)
            mmr_score = (1.0 - MMR_DIVERSITY_LAMBDA) * float(score) - MMR_DIVERSITY_LAMBDA * diversity_penalty
            if mmr_score > best_value:
                best_value = mmr_score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_fps.append(item_fingerprints[best_idx])

    eps = exploration_epsilon(session.step)
    if selected and random.random() < eps:
        explore_candidates = [idx for idx, _ in rank_pool[: max(8, FEED_K * 2)] if idx not in selected]
        if explore_candidates:
            replace_pos = random.randrange(len(selected))
            selected[replace_pos] = random.choice(explore_candidates)

    # Enforce strict near-duplicate suppression after final rank.
    deduped: list[int] = []
    deduped_fps: list[set[str]] = []
    for idx in selected:
        fp = item_fingerprints[idx]
        if any(calculate_similarity(fp, seen_fp) > DEDUP_SIM_THRESHOLD for seen_fp in deduped_fps):
            continue
        deduped.append(idx)
        deduped_fps.append(fp)

    if len(deduped) < FEED_K:
        for idx, _ in rank_pool:
            if idx in deduped:
                continue
            fp = item_fingerprints[idx]
            if any(calculate_similarity(fp, seen_fp) > DEDUP_SIM_THRESHOLD for seen_fp in deduped_fps):
                continue
            deduped.append(idx)
            deduped_fps.append(fp)
            if len(deduped) >= FEED_K:
                break
    return deduped[:FEED_K]


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request) -> HTMLResponse:
    sid = get_session_id(request)
    response = HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))
    response.set_cookie(key=SESSION_COOKIE, value=sid, httponly=True, samesite="lax")
    return response


@app.get("/feed")
async def get_feed(request: Request) -> dict[str, list[dict[str, object]]]:
    sid = get_session_id(request)
    with state_lock:
        session = get_or_create_session(sid)
        recent_seen_set = set(session.recent_seen)
        available = [i for i in range(len(inventory)) if i not in recent_seen_set]
        if len(available) < FEED_K:
            available = list(range(len(inventory)))
        if not available:
            return {"items": []}

        event_seq = torch.tensor(list(session.event_seq), dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            base_user_state = model.encode_user_state(event_seq.unsqueeze(0)).squeeze(0)
            adapted_user_state = base_user_state + session.user_delta

        # Multi-stage cascade: recall -> pre-rank -> final rank.
        recall_indices = select_recall_candidates(
            available_indices=available,
            adapted_user_state=adapted_user_state,
        )
        pre_ranked = pre_rank_candidates(
            session=session,
            recall_indices=recall_indices,
            adapted_user_state=adapted_user_state,
        )
        chosen = final_rank_candidates(session=session, pre_ranked=pre_ranked)
        if len(chosen) < FEED_K:
            fallback = [idx for idx in recall_indices if idx not in chosen]
            chosen.extend(fallback[: FEED_K - len(chosen)])

        session.last_batch = chosen

        items = []
        for idx in chosen:
            row = inventory.iloc[idx]
            items.append(
                {
                    "id": int(row["id"]),
                    "title": str(row[headline_col]),
                    "description": (
                        str(row[summary_col]) if summary_col in inventory.columns else ""
                    ),
                    "domain": str(row[domain_col]),
                    "link": str(row[link_col]),
                }
            )
        return {"items": items}


@app.post("/interact")
async def interact(data: Request) -> dict[str, object]:
    json_data = await data.json()
    sid = get_session_id(data)
    item_id = int(json_data.get("item_id"))
    reason = str(json_data.get("reason", "auto")).lower()
    dwell = max(0.0, min(10.0, float(json_data.get("dwell_time", 0.0))))

    with state_lock:
        session = get_or_create_session(sid)
        if item_id < 0 or item_id >= len(inventory):
            return {"status": "error", "message": "invalid item_id"}

        feat = item_features[item_id].unsqueeze(0)
        topic_id = int(feat[0, ITEM_COL_TOPIC_ID].item())
        vibe_id = int(feat[0, ITEM_COL_VIBE_ID].item())
        reward = reward_from_interaction(reason, dwell, list(session.skip_dwells))
        reward_norm = reward_to_norm(reward)
        watch_ratio = dwell / 10.0

        event_seq = torch.tensor(list(session.event_seq), dtype=torch.long, device=DEVICE)
        target_watch = torch.tensor([[watch_ratio]], dtype=torch.float32, device=DEVICE)
        target_finish = torch.tensor([[1.0 if watch_ratio >= 0.95 else 0.0]], device=DEVICE)
        target_fast_skip = torch.tensor([[1.0 if watch_ratio <= 0.20 else 0.0]], device=DEVICE)
        target_like = torch.tensor([[1.0 if watch_ratio >= 0.85 else 0.0]], device=DEVICE)
        target_share = torch.tensor([[1.0 if watch_ratio >= 0.95 else 0.0]], device=DEVICE)
        target_rewatch = torch.tensor([[1.0 if watch_ratio >= 0.90 else 0.0]], device=DEVICE)

        if ONLINE_UPDATE_MODE == "global":
            base_user_state = model.encode_user_state(event_seq.unsqueeze(0)).squeeze(0)
            adapted_user_state = base_user_state + session.user_delta
            heads = forward_heads_from_user_state(adapted_user_state, feat)
            pred_score = model.score_from_heads(heads, weights=SCORE_WEIGHTS)
            loss = (
                0.8 * bce(heads["finish_logit"], target_finish)
                + 0.6 * bce(heads["fast_skip_logit"], target_fast_skip)
                + 0.5 * bce(heads["like_logit"], target_like)
                + 0.4 * bce(heads["share_logit"], target_share)
                + 0.4 * bce(heads["rewatch_logit"], target_rewatch)
                + 1.5 * mse(torch.sigmoid(heads["watch_time_raw"]), target_watch)
                + 1.0 * mse(torch.sigmoid(pred_score), torch.tensor([[reward_norm]], device=DEVICE))
            )
            if optimizer is None:
                raise RuntimeError("Global mode requires optimizer")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        elif ONLINE_UPDATE_MODE == "private":
            with torch.no_grad():
                base_user_state = model.encode_user_state(event_seq.unsqueeze(0)).squeeze(0)

            session_delta = session.user_delta.detach().clone().requires_grad_(True)
            adapted_user_state = base_user_state + session_delta
            heads = forward_heads_from_user_state(adapted_user_state, feat)
            pred_score = model.score_from_heads(heads, weights=SCORE_WEIGHTS)
            loss = (
                0.8 * bce(heads["finish_logit"], target_finish)
                + 0.6 * bce(heads["fast_skip_logit"], target_fast_skip)
                + 0.5 * bce(heads["like_logit"], target_like)
                + 0.4 * bce(heads["share_logit"], target_share)
                + 0.4 * bce(heads["rewatch_logit"], target_rewatch)
                + 1.5 * mse(torch.sigmoid(heads["watch_time_raw"]), target_watch)
                + 1.0 * mse(torch.sigmoid(pred_score), torch.tensor([[reward_norm]], device=DEVICE))
            )
            loss.backward()
            grad = session_delta.grad
            if grad is not None:
                grad_norm = float(torch.linalg.vector_norm(grad).item())
                if grad_norm > PRIVATE_USER_GRAD_CLIP:
                    grad = grad * (PRIVATE_USER_GRAD_CLIP / (grad_norm + 1e-8))
                updated = (session.user_delta - PRIVATE_USER_LR * grad).detach()
                delta_norm = float(torch.linalg.vector_norm(updated).item())
                if delta_norm > PRIVATE_USER_MAX_NORM:
                    updated = updated * (PRIVATE_USER_MAX_NORM / (delta_norm + 1e-8))
                session.user_delta = updated
        else:
            with torch.no_grad():
                base_user_state = model.encode_user_state(event_seq.unsqueeze(0)).squeeze(0)
                adapted_user_state = base_user_state + session.user_delta
                heads = forward_heads_from_user_state(adapted_user_state, feat)
                pred_score = model.score_from_heads(heads, weights=SCORE_WEIGHTS)

        session.step += 1
        session.recent_seen.append(item_id)
        if reason == "skip":
            session.skip_dwells.append(dwell)
        session.update_reward(reward)
        token = build_event_token(
            item_id=int(feat[0, ITEM_COL_ITEM_ID].item()),
            topic_id=topic_id,
            vibe_id=vibe_id,
            dwell=dwell,
            step=session.step,
        )
        session.event_seq.append(token)

        row = inventory.iloc[item_id]
        append_usage_log_row(
            USAGE_LOG_PATH,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source": "live_app",
                "run_id": "",
                "session_id": sid,
                "step": session.step,
                "user_step": session.step,
                "user_type": "unknown_live",
                "item_id": int(feat[0, ITEM_COL_ITEM_ID].item()),
                "topic_id": topic_id,
                "topic_name": str(row[domain_col]),
                "vibe_id": vibe_id,
                "vibe_style": str(inventory.iloc[item_id]["Vibe_Style"])
                if "Vibe_Style" in inventory.columns
                else "",
                "cluster_id": int(feat[0, ITEM_COL_TOPIC_CLUSTER_ID].item()),
                "headline": str(row[headline_col]),
                "reward": float(reward),
                "regret": "",
                "epsilon": "",
                "decision": reason,
                "pred_score": float(torch.sigmoid(pred_score).item()),
                "listen_ms": float(dwell * 1000.0),
                "total_ms": 10000.0,
                "liked": bool(watch_ratio >= 0.85),
                "shared": bool(watch_ratio >= 0.95),
                "rewinded": bool(watch_ratio >= 0.90),
                "fast_skip": bool(watch_ratio <= 0.20),
                "finished": bool(watch_ratio >= 0.95),
                "interest": float(watch_ratio),
            },
        )
        persist_session_state(sid, session)

        skip_avg = sum(session.skip_dwells) / len(session.skip_dwells) if session.skip_dwells else 0.0
        print(
            f"🔥 LIVE SIGNAL | Session: {sid[:8]} | Reason: {reason.upper()} | "
            f"Dwell: {dwell:.2f}s | Watch: {watch_ratio:.2f} | Reward: {reward:.2f} | "
            f"AvgReward: {session.avg_reward:.2f} | SkipAvg: {skip_avg:.2f}s | "
            f"Mode: {ONLINE_UPDATE_MODE.upper()} | Item: {item_id}"
        )

        return {
            "status": "learned",
            "reward": reward,
            "watch_ratio": watch_ratio,
            "avg_reward": session.avg_reward,
            "step": session.step,
            "update_mode": ONLINE_UPDATE_MODE,
        }
