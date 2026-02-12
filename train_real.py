"""Train v2 recommender with session-sequence user state and multi-head objectives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans

from model import (
    ITEM_COL_EVENT_ID,
    ITEM_COL_ITEM_ID,
    ITEM_COL_POPULARITY_BUCKET,
    ITEM_COL_RECENCY_BUCKET,
    ITEM_COL_TOPIC_CLUSTER_ID,
    ITEM_COL_TOPIC_ID,
    ITEM_COL_VIBE_ID,
    NUM_ITEM_FEATURES,
    GenerativeTwoTower,
)
from user_bot import (
    AIDoomBot,
    ContrarianChaosBot,
    GamerBot,
    InvestorBot,
    LowMoodAuditBot,
    SarahBot,
    UserBot,
)

# --- CONFIGURATION ---
DATA_PATH = "data/scraped_articles.csv"
USAGE_LOG_PATH = Path("data/usage_interactions.csv")
STEPS = 150
LEARNING_RATE = 0.001
WARM_START_PATH = Path("outputs/train_real_warm_start.pt")
LOAD_WARM_START = False

EVENT_SEQ_LEN = 30
EVENT_VOCAB_SIZE = 4096
RECALL_SIZE = 2000
PRE_RANK_SIZE = 200
RANK_POOL_SIZE = 8
RANK_POSITIVE_MIN = 2
POSITIVE_SIGNAL_THRESHOLD = 1.15
EPS_START = 0.25
EPS_END = 0.10
PANIC_EPS = 0.35
GLOBAL_RECOVERY_EPS = 0.80
GLOBAL_RECOVERY_ROLLING_THRESHOLD = 0.0
GLOBAL_RECOVERY_STREAK_STEPS = 20
GRAD_CLIP_NORM = 5.0
REGRET_EVERY = 10
FAST_SKIP_PENALTY = 6.0
BAD_DOMAIN_REWARD_THRESHOLD = -3.0
BAD_DOMAIN_MIN_IMPRESSIONS = 4
HARD_SUPPRESS_MULTIPLIER = 0.10
RECENT_SEEN_COOLDOWN = 180
REPLAY_BUFFER_SIZE = 512
REPLAY_POS_THRESHOLD = 4.5
REPLAY_UPDATES_PER_STEP = 0
HARD_NEGATIVE_MINING = True
HARD_NEG_POOL_SIZE = 64
HARD_NEG_CONFIDENCE_MIN = 0.60
HARD_NEGATIVE_WEIGHT = 0.5
HARD_NEGATIVE_TARGET_NORM = 0.05
SAVE_ONLY_IF_GOOD = True
SAVE_REWARD_THRESHOLD = 0.0
SCORE_WEIGHTS = {
    "finish": 1.2,
    "like": 1.0,
    "share": 1.2,
    "rewatch": 1.1,
    "fast_skip": 0.8,
    "watch_time": 1.1,
}
LOW_SIGNAL_DOMAINS = {
    "news.google.com",
    "news.un.org",
}
TECH_BOOST_DOMAINS = {
    "theverge.com",
    "techcrunch.com",
    "wired.com",
    "arstechnica.com",
    "engadget.com",
    "technologyreview.com",
    "venturebeat.com",
    "androidauthority.com",
    "macrumors.com",
    "9to5mac.com",
    "zdnet.com",
    "cnet.com",
}

DEVICE = torch.device("cpu")
WORD_RE = re.compile(r"[a-z0-9]+")

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


def extract_domain(url: str) -> str:
    clean = str(url).replace("https://", "").replace("http://", "")
    domain = clean.split("/")[0].replace("www.", "").strip().lower()
    return domain or "unknown"


def entity_key(headline: str) -> str:
    text = str(headline).lower()
    tokens = set(WORD_RE.findall(text))

    def has_phrase(phrase: str) -> bool:
        p = phrase.strip().lower()
        if not p:
            return False
        if " " in p:
            return p in text
        return p in tokens

    # Guard against false positives like "prime minister" leaking into unrelated entities.
    if "prime minister" in text:
        pass

    mapping = [
        ("geopolitics_conflict", ["ukraine", "russia", "nato", "china", "taiwan", "border", "missile"]),
        ("structural_governance", ["starmer", "labour", "tory", "trump", "senate", "congress", "ftc", "court"]),
        ("macro_economics", ["inflation", "tariff", "recession", "jobs", "fed", "interest rate", "debt"]),
        ("systemic_health", ["alzheimer", "cancer", "cholera", "formula", "toxic", "forever chemicals"]),
        ("frontier_lifestyle", ["luxury", "billionaire", "zuckerberg", "mansion", "exclusive", "fashion"]),
        ("openai", ["openai", "chatgpt", "sam altman", "sora", "gpt-5", "o1-preview"]),
        ("google", ["google", "alphabet", "youtube", "gemini", "deepmind", "waymo", "pixel 10"]),
        ("apple", ["apple", "iphone", "macbook", "vision pro", "m4 max", "ios 18", "tim cook"]),
        ("meta", ["meta", "facebook", "instagram", "threads", "llama", "zuckerberg", "quest 4"]),
        ("microsoft", ["microsoft", "windows", "azure", "copilot", "nadella", "xbox"]),
        ("amazon", ["amazon", "aws", "blue origin", "bezos", "anthropic", "kindle"]),
        ("tesla_musk", ["tesla", "spacex", "elon musk", "xai", "grok", "starship", "neuralink"]),
        ("nvidia_chips", ["nvidia", "amd", "tsmc", "intel", "h100", "blackwell", "semiconductor"]),
        ("finance_markets", ["sp500", "nasdaq", "ipo", "bitcoin"]),
    ]
    for key, keywords in mapping:
        if any(has_phrase(k) for k in keywords):
            return key
    return "general_tech_other"


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
    """Ensure all model-required feature columns exist."""
    out = df.copy()

    if "Topic_Name" not in out.columns:
        out["Topic_Name"] = out["Link"].apply(extract_domain)
    if "Topic_ID" not in out.columns:
        out["Topic_ID"], _ = pd.factorize(out["Topic_Name"])

    style_col = "Suggested_Style" if "Suggested_Style" in out.columns else "Vibe_Style"
    if style_col not in out.columns:
        raise ValueError("CSV must contain 'Suggested_Style' or 'Vibe_Style'.")
    out["Vibe_Style"] = out[style_col].fillna("Standard Reporting").astype(str)
    if "Vibe_ID" not in out.columns:
        out["Vibe_ID"], _ = pd.factorize(out["Vibe_Style"])

    if "Topic_Cluster_ID" not in out.columns:
        # Cheap fallback cluster ID from hashed headline token.
        out["Topic_Cluster_ID"] = (
            out["Headline"].fillna("").astype(str).apply(lambda s: abs(hash(s.split(" ")[0])) % 32)
        )

    if "Entity_ID" not in out.columns:
        out["Entity_Key"] = out["Headline"].apply(entity_key)
        out["Entity_ID"], _ = pd.factorize(out["Entity_Key"])

    if "Event_ID" not in out.columns:
        out["Event_Key"] = out["Headline"].apply(event_key)
        out["Event_ID"], _ = pd.factorize(out["Event_Key"])

    if "Recency_Bucket" not in out.columns:
        if "Published_UTC" in out.columns:
            now = pd.Timestamp.now(tz="UTC")
            published = pd.to_datetime(out["Published_UTC"], errors="coerce", utc=True).fillna(now)
            hours = (now - published).dt.total_seconds() / 3600.0
            out["Recency_Bucket"] = pd.cut(
                hours,
                bins=[-1, 1, 6, 24, 72, float("inf")],
                labels=[0, 1, 2, 3, 4],
            ).astype(int)
        else:
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

    if "Item_ID" not in out.columns:
        out["Item_ID"] = range(len(out))
    out["Item_ID"], _ = pd.factorize(out["Item_ID"])

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


def calculate_addiction_reward(session: dict[str, float | bool]) -> float:
    completion = float(session["listen_ms"]) / float(session["total_ms"])
    points = completion * 8.0
    if bool(session["rewinded"]):
        points += 10.0
    if bool(session["shared"]):
        points += 6.0
    if float(session["listen_ms"]) < 2000.0:
        points -= FAST_SKIP_PENALTY
    return max(-10.0, min(10.0, points))


def calibrate_reward(raw_reward: float) -> float:
    """Map raw reward in [-10, 10] to [0.05, 0.95] for stable training."""
    norm = (raw_reward + 10.0) / 20.0
    return max(0.05, min(0.95, norm))


def epsilon_for_step(step: int, total_steps: int) -> float:
    t = min(1.0, max(0.0, step / max(1, total_steps)))
    return EPS_START + (EPS_END - EPS_START) * t


def stable_event_token(raw: str) -> int:
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return 1 + (int(digest[:8], 16) % (EVENT_VOCAB_SIZE - 1))


def build_event_token(
    item_id: int,
    topic_id: int,
    vibe_id: int,
    session: dict[str, float | bool],
    step: int,
    user_tag: str = "u0",
) -> int:
    listen_ratio = float(session["listen_ms"]) / float(session["total_ms"])
    if listen_ratio < 0.2:
        dwell_bucket = 0
    elif listen_ratio < 0.5:
        dwell_bucket = 1
    elif listen_ratio < 0.8:
        dwell_bucket = 2
    else:
        dwell_bucket = 3
    session_bucket = min(9, step // 10)
    time_of_day_bucket = step % 4
    raw = (
        f"i{item_id}|t{topic_id}|v{vibe_id}|dw{dwell_bucket}|"
        f"fs{int(bool(session['fast_skip']))}|rw{int(bool(session['rewinded']))}|"
        f"sh{int(bool(session['shared']))}|lk{int(bool(session['liked']))}|"
        f"cm{int(bool(session['commented']))}|tod{time_of_day_bucket}|"
        f"sx{session_bucket}|uid{user_tag}|dev0|geo0"
    )
    return stable_event_token(raw)


def build_targets(session: dict[str, float | bool]) -> dict[str, torch.Tensor]:
    completion = float(session["listen_ms"]) / float(session["total_ms"])
    return {
        "finish": torch.tensor([[float(bool(session["finished"]))]], device=DEVICE),
        "like": torch.tensor([[float(bool(session["liked"]))]], device=DEVICE),
        "share": torch.tensor([[float(bool(session["shared"]))]], device=DEVICE),
        "rewatch": torch.tensor([[float(bool(session["rewinded"]))]], device=DEVICE),
        "fast_skip": torch.tensor([[float(bool(session["fast_skip"]))]], device=DEVICE),
        "watch_time": torch.tensor([[completion]], device=DEVICE),
    }


def compute_total_loss(
    heads: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    reward_norm: float,
    bce: nn.BCEWithLogitsLoss,
    mse: nn.MSELoss,
    score_weights: dict[str, float],
):
    pred_score_tensor = model_score_from_heads(heads, score_weights)
    pred_score_norm = torch.sigmoid(pred_score_tensor)

    loss_finish = bce(heads["finish_logit"], targets["finish"])
    loss_like = bce(heads["like_logit"], targets["like"])
    loss_share = bce(heads["share_logit"], targets["share"])
    loss_rewatch = bce(heads["rewatch_logit"], targets["rewatch"])
    loss_fast_skip = bce(heads["fast_skip_logit"], targets["fast_skip"])
    pred_watch = torch.sigmoid(heads["watch_time_raw"])
    loss_watch = mse(pred_watch, targets["watch_time"])
    loss_rank = mse(
        pred_score_norm,
        torch.tensor([[reward_norm]], dtype=torch.float32, device=DEVICE),
    )

    loss = (
        1.0 * loss_finish
        + 0.8 * loss_like
        + 1.2 * loss_share
        + 1.2 * loss_rewatch
        + 1.0 * loss_fast_skip
        + 2.0 * loss_watch
        + 1.5 * loss_rank
    )
    return loss, pred_score_tensor


def model_score_from_heads(
    heads: dict[str, torch.Tensor], score_weights: dict[str, float]
) -> torch.Tensor:
    p_finish = torch.sigmoid(heads["finish_logit"])
    p_like = torch.sigmoid(heads["like_logit"])
    p_share = torch.sigmoid(heads["share_logit"])
    p_rewatch = torch.sigmoid(heads["rewatch_logit"])
    p_fast_skip = torch.sigmoid(heads["fast_skip_logit"])
    expected_watch = torch.sigmoid(heads["watch_time_raw"])
    return (
        score_weights["finish"] * p_finish
        + score_weights["like"] * p_like
        + score_weights["share"] * p_share
        + score_weights["rewatch"] * p_rewatch
        - score_weights["fast_skip"] * p_fast_skip
        + score_weights["watch_time"] * expected_watch
    )


def mine_hard_negative(
    bot: UserBot,
    step: int,
    candidate_indices: list[int],
    candidate_scores: np.ndarray,
    winner_idx: int,
    all_item_features: torch.Tensor,
    headlines: list[str],
    topic_map: dict[int, str],
    vibe_map: dict[int, str],
) -> dict[str, float | int] | None:
    """
    Find a hard negative: high-scoring candidate that this user would dislike.

    This emulates "confidently wrong" mining by scanning top-scored candidates
    and selecting the most confident one with negative reward.
    """
    if not candidate_indices:
        return None

    ranked = sorted(
        zip(candidate_indices, candidate_scores.tolist()),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    top_scan = ranked[: min(HARD_NEG_POOL_SIZE, len(ranked))]

    best_idx = None
    best_conf = -1.0
    best_reward = 0.0
    for idx, score in top_scan:
        if idx == winner_idx:
            continue
        feat = all_item_features[idx]
        topic_id = int(feat[ITEM_COL_TOPIC_ID].item())
        vibe_id = int(feat[ITEM_COL_VIBE_ID].item())
        cluster_id = int(feat[ITEM_COL_TOPIC_CLUSTER_ID].item())
        sim = bot.simulate_session(
            topic_id=topic_id,
            vibe_id=vibe_id,
            topic_map=topic_map,
            vibe_map=vibe_map,
            headline=headlines[idx],
            cluster_id=cluster_id,
            session_index=step,
            update_state=False,
        )
        sim_reward = float(calculate_addiction_reward(sim))
        if sim_reward >= 0.0:
            continue

        confidence = float(1.0 / (1.0 + np.exp(-float(score))))
        if confidence > best_conf:
            best_conf = confidence
            best_idx = int(idx)
            best_reward = sim_reward

    if best_idx is None or best_conf < HARD_NEG_CONFIDENCE_MIN:
        return None
    return {"idx": best_idx, "confidence": best_conf, "reward": best_reward}


def select_recall_candidates(
    model: GenerativeTwoTower,
    event_seq: torch.Tensor,
    all_item_features: torch.Tensor,
    available_indices: list[int],
) -> list[int]:
    if not available_indices:
        return []
    recall_target = min(RECALL_SIZE, len(available_indices))
    available_tensor = all_item_features[available_indices]

    with torch.no_grad():
        user_state = model.encode_user_state(event_seq.unsqueeze(0)).squeeze(0)
        item_vectors = model.encode_item(available_tensor)
        nn_scores = torch.mv(item_vectors, user_state)
        nn_scores = nn_scores + 0.03 * torch.randn_like(nn_scores)

    n_random = min(len(available_indices), max(1, int(recall_target * 0.35)))
    n_nn = min(len(available_indices), max(1, int(recall_target * 0.35)))
    n_pop = min(len(available_indices), max(1, int(recall_target * 0.15)))
    n_fresh = min(len(available_indices), max(1, recall_target - n_random - n_nn - n_pop))

    random_part = random.sample(available_indices, n_random)

    _, nn_local = torch.topk(nn_scores, k=n_nn)
    nn_part = [available_indices[i] for i in nn_local.tolist()]

    pop_scores = available_tensor[:, ITEM_COL_POPULARITY_BUCKET]
    _, pop_local = torch.topk(pop_scores, k=n_pop)
    pop_part = [available_indices[i] for i in pop_local.tolist()]

    fresh_scores = -available_tensor[:, ITEM_COL_RECENCY_BUCKET]
    _, fresh_local = torch.topk(fresh_scores, k=n_fresh)
    fresh_part = [available_indices[i] for i in fresh_local.tolist()]

    merged = []
    seen = set()
    for idx in random_part + nn_part + pop_part + fresh_part:
        if idx not in seen:
            merged.append(idx)
            seen.add(idx)
        if len(merged) >= recall_target:
            break

    if len(merged) < recall_target:
        leftovers = [i for i in available_indices if i not in seen]
        merged.extend(leftovers[: recall_target - len(merged)])
    return merged


def pre_rank_candidates(
    model: GenerativeTwoTower,
    event_seq: torch.Tensor,
    all_item_features: torch.Tensor,
    recall_indices: list[int],
    recent_topic_ids: deque[int],
    recent_cluster_ids: deque[int],
    recent_domain_ids: deque[int],
    bad_domain_ids: set[int],
    topic_map: dict[int, str],
    score_weights: dict[str, float],
) -> list[int]:
    if not recall_indices:
        return []
    recall_features = all_item_features[recall_indices]
    repeated_seq = event_seq.unsqueeze(0).repeat(len(recall_indices), 1)

    with torch.no_grad():
        heads = model.forward(repeated_seq, recall_features)
        base_scores = model.score_from_heads(heads, weights=score_weights).squeeze(1).cpu().numpy()

    topic_counts = Counter(recent_topic_ids)
    cluster_counts = Counter(recent_cluster_ids)
    diversity_tax_domain_id = None
    if len(recent_domain_ids) >= 5:
        last_five = list(recent_domain_ids)[-5:]
        if len(set(last_five)) == 1:
            diversity_tax_domain_id = last_five[0]

    adjusted = []
    for local_i, idx in enumerate(recall_indices):
        feat = all_item_features[idx]
        topic_id = int(feat[ITEM_COL_TOPIC_ID].item())
        cluster_id = int(feat[ITEM_COL_TOPIC_CLUSTER_ID].item())
        recency_bucket = int(feat[ITEM_COL_RECENCY_BUCKET].item())
        topic_name = topic_map.get(topic_id, "unknown")

        novelty_bonus = 0.10 if topic_counts[topic_id] == 0 else 0.0
        cluster_bonus = 0.05 if cluster_counts[cluster_id] == 0 else 0.0
        fatigue_penalty = 0.07 * topic_counts[topic_id]
        freshness_bonus = 0.08 * max(0, 4 - recency_bucket) / 4.0
        domain_prior = 0.0
        if topic_name in TECH_BOOST_DOMAINS:
            domain_prior += 0.16
        if topic_name in LOW_SIGNAL_DOMAINS:
            domain_prior -= 0.30
        if topic_name == "cnn.com":
            domain_prior -= 0.12

        adjusted_score = (
            float(base_scores[local_i])
            + novelty_bonus
            + cluster_bonus
            + freshness_bonus
            + domain_prior
            - fatigue_penalty
        )
        # Hard suppression for repeatedly negative domains.
        if topic_id in bad_domain_ids:
            adjusted_score *= HARD_SUPPRESS_MULTIPLIER
        # Variety tax: if last 5 items were same domain, cut same-domain score by 80%.
        if diversity_tax_domain_id is not None and topic_id == diversity_tax_domain_id:
            adjusted_score *= 0.20
        adjusted.append((adjusted_score, idx))

    adjusted.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in adjusted[: min(PRE_RANK_SIZE, len(adjusted))]]


def counterfactual_regret(
    bot: UserBot,
    step: int,
    available_indices: list[int],
    all_item_features: torch.Tensor,
    headlines: list[str],
    topic_map: dict[int, str],
    vibe_map: dict[int, str],
    chosen_reward: float,
) -> float:
    if not available_indices:
        return 0.0
    sample_n = min(50, len(available_indices))
    sampled = random.sample(available_indices, sample_n)
    rewards = []
    for idx in sampled:
        feat = all_item_features[idx]
        topic_id = int(feat[ITEM_COL_TOPIC_ID].item())
        vibe_id = int(feat[ITEM_COL_VIBE_ID].item())
        cluster_id = int(feat[ITEM_COL_TOPIC_CLUSTER_ID].item())
        sim = bot.simulate_session(
            topic_id=topic_id,
            vibe_id=vibe_id,
            topic_map=topic_map,
            vibe_map=vibe_map,
            headline=headlines[idx],
            cluster_id=cluster_id,
            session_index=step,
            update_state=False,
        )
        rewards.append(calculate_addiction_reward(sim))
    if not rewards:
        return 0.0
    return max(rewards) - chosen_reward


def estimate_persona_positive_rate(
    bot: UserBot,
    item_features: torch.Tensor,
    headlines: list[str],
    topic_map: dict[int, str],
    vibe_map: dict[int, str],
    sample_size: int = 500,
) -> tuple[float, int]:
    """Estimate how often a persona yields positive reward on this dataset."""
    total = int(item_features.size(0))
    if total <= 0:
        return 0.0, 0
    sample_n = min(sample_size, total)
    sampled = random.sample(range(total), sample_n)
    positives = 0
    for idx in sampled:
        feat = item_features[idx]
        topic_id = int(feat[ITEM_COL_TOPIC_ID].item())
        vibe_id = int(feat[ITEM_COL_VIBE_ID].item())
        cluster_id = int(feat[ITEM_COL_TOPIC_CLUSTER_ID].item())
        sim = bot.simulate_session(
            topic_id=topic_id,
            vibe_id=vibe_id,
            topic_map=topic_map,
            vibe_map=vibe_map,
            headline=headlines[idx],
            cluster_id=cluster_id,
            session_index=0,
            update_state=False,
        )
        if calculate_addiction_reward(sim) > 0.0:
            positives += 1
    return (positives / sample_n), sample_n


def main(
    total_steps: int = STEPS,
    bot_name: str = "dave",
    data_path: str = DATA_PATH,
    load_warm_start: bool = LOAD_WARM_START,
    warm_start_path: str | Path = WARM_START_PATH,
    save_warm_start_path: str | Path | None = None,
    save_only_if_good: bool = SAVE_ONLY_IF_GOOD,
    save_reward_threshold: float = SAVE_REWARD_THRESHOLD,
) -> None:
    total_steps = max(1, int(total_steps))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data_csv_path = Path(data_path)
    if not data_csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_csv_path}")

    print(f"🚀 Loading Fuel... ({data_csv_path})")
    df_raw = pd.read_csv(data_csv_path)
    df = ensure_feature_columns(df_raw)
    # Break source/topic walls from ingest ordering.
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    headlines = df["Headline"].fillna("").astype(str).tolist()

    topic_map = (
        df[["Topic_ID", "Topic_Name"]]
        .drop_duplicates()
        .set_index("Topic_ID")["Topic_Name"]
        .astype(str)
        .to_dict()
    )
    vibe_map = (
        df[["Vibe_ID", "Vibe_Style"]]
        .drop_duplicates()
        .set_index("Vibe_ID")["Vibe_Style"]
        .astype(str)
        .to_dict()
    )

    item_features = torch.tensor(
        df[
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
    if item_features.size(1) != NUM_ITEM_FEATURES:
        raise ValueError(f"Expected {NUM_ITEM_FEATURES} item feature columns.")

    # Embedding tables must be sized with max_id + 1 (not nunique), because IDs can be sparse.
    num_items = int(df["Item_ID"].max()) + 1
    num_topics = int(df["Topic_ID"].max()) + 1
    num_vibes = int(df["Vibe_ID"].max()) + 1
    num_clusters = int(df["Topic_Cluster_ID"].max()) + 1
    num_entities = int(df["Entity_ID"].max()) + 1
    num_events = int(df["Event_ID"].max()) + 1
    num_recency = int(df["Recency_Bucket"].max()) + 1
    num_popularity = int(df["Popularity_Bucket"].max()) + 1

    model = GenerativeTwoTower(
        num_items=num_items,
        num_topics=num_topics,
        num_vibes=num_vibes,
        num_topic_clusters=num_clusters,
        num_entities=num_entities,
        num_events=num_events,
        num_recency_buckets=num_recency,
        num_popularity_buckets=num_popularity,
        event_vocab_size=EVENT_VOCAB_SIZE,
    ).to(DEVICE)
    bot_mode = str(bot_name).strip().lower()
    if bot_mode == "multi":
        bots: dict[str, UserBot] = {
            "dave": UserBot("Complex Dave V2", seed=42),
            "sarah": SarahBot(seed=43),
            "gamer": GamerBot(seed=44),
            "investor": InvestorBot(seed=45),
            "low_mood_audit": LowMoodAuditBot(seed=46),
            "ai_doom": AIDoomBot(seed=47),
            "contrarian_chaos": ContrarianChaosBot(seed=48),
        }
    elif bot_mode == "sarah":
        bots = {"sarah": SarahBot(seed=43)}
    elif bot_mode == "gamer":
        bots = {"gamer": GamerBot(seed=44)}
    elif bot_mode == "investor":
        bots = {"investor": InvestorBot(seed=45)}
    elif bot_mode == "low_mood_audit":
        bots = {"low_mood_audit": LowMoodAuditBot(seed=46)}
    elif bot_mode == "ai_doom":
        bots = {"ai_doom": AIDoomBot(seed=47)}
    elif bot_mode == "contrarian_chaos":
        bots = {"contrarian_chaos": ContrarianChaosBot(seed=48)}
    else:
        bots = {"dave": UserBot("Complex Dave V2", seed=42)}

    # Preflight: verify each bot can find enough positive outcomes in this dataset.
    for bname, b in bots.items():
        if isinstance(b, ContrarianChaosBot):
            orig_step = b.step_count
            b.step_count = 101  # Probe in rabbit-hole phase.
            pos_rate, n = estimate_persona_positive_rate(
                bot=b,
                item_features=item_features,
                headlines=headlines,
                topic_map=topic_map,
                vibe_map=vibe_map,
            )
            b.step_count = orig_step
            print(
                f"🔎 Preflight {bname} phase2_positive_rate={pos_rate:.3f} "
                f"(sample={n})"
            )
            if pos_rate < 0.05:
                print(
                    "⚠️ Very low phase-2 positives for contrarian_chaos; "
                    "expect unstable or collapsed training unless dataset is broadened."
                )
        else:
            pos_rate, n = estimate_persona_positive_rate(
                bot=b,
                item_features=item_features,
                headlines=headlines,
                topic_map=topic_map,
                vibe_map=vibe_map,
            )
            print(f"🔎 Preflight {bname} positive_rate={pos_rate:.3f} (sample={n})")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    warm_start_path = Path(warm_start_path)
    save_path = Path(save_warm_start_path) if save_warm_start_path else warm_start_path

    if load_warm_start and warm_start_path.exists():
        checkpoint = torch.load(warm_start_path, map_location=DEVICE)
        try:
            if checkpoint.get("model_state_dict"):
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if checkpoint.get("optimizer_state_dict"):
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"♻️ Warm-start loaded from: {warm_start_path}")
        except Exception as exc:
            print(f"⚠️ Skipping incompatible warm-start checkpoint: {exc}")

    print(
        f"\n🧠 Training on {len(df)} items | topics={num_topics} vibes={num_vibes} "
        f"clusters={num_clusters} entities={num_entities} events={num_events} "
        f"| steps={total_steps} | bot={bot_mode}\n"
    )
    reward_log: list[float] = []
    regret_log: list[float] = []
    user_state_log: list[np.ndarray] = []
    phase_log: list[str] = []
    replay_buffer: deque[dict[str, object]] = deque(maxlen=REPLAY_BUFFER_SIZE)
    run_id = f"train_{uuid4().hex[:12]}"
    global_low_roll_streak = 0

    user_states: dict[str, dict[str, object]] = {}
    for name in bots:
        user_states[name] = {
            "recent_seen": deque(maxlen=RECENT_SEEN_COOLDOWN),
            "recent_seen_set": set(),
            "recent_topic_ids": deque(maxlen=10),
            "recent_cluster_ids": deque(maxlen=10),
            "recent_domain_ids": deque(maxlen=10),
            "event_seq_tokens": deque([0] * EVENT_SEQ_LEN, maxlen=EVENT_SEQ_LEN),
            "recent_domain_rewards": {},
            "reward_log": [],
            "regret_log": [],
            "session_step": 0,
        }

    for step in range(1, total_steps + 1):
        if bot_mode == "multi":
            active_bot_name = random.choice(list(bots.keys()))
        else:
            active_bot_name = next(iter(bots.keys()))
        bot = bots[active_bot_name]
        state = user_states[active_bot_name]
        state["session_step"] = int(state["session_step"]) + 1
        user_step = int(state["session_step"])

        recent_seen = state["recent_seen"]
        recent_seen_set = state["recent_seen_set"]
        recent_topic_ids = state["recent_topic_ids"]
        recent_cluster_ids = state["recent_cluster_ids"]
        recent_domain_ids = state["recent_domain_ids"]
        event_seq_tokens = state["event_seq_tokens"]
        recent_domain_rewards = state["recent_domain_rewards"]

        available_indices = [i for i in range(len(df)) if i not in recent_seen_set]
        if len(available_indices) < 2:
            available_indices = list(range(len(df)))

        # Reduce low-signal source exposure during recall while keeping fallback coverage.
        preferred_available = []
        for idx in available_indices:
            topic_id = int(item_features[idx, ITEM_COL_TOPIC_ID].item())
            if topic_map.get(topic_id, "unknown") not in LOW_SIGNAL_DOMAINS:
                preferred_available.append(idx)
        recall_pool = preferred_available if len(preferred_available) >= max(100, len(available_indices) // 4) else available_indices

        event_seq = torch.tensor(list(event_seq_tokens), dtype=torch.long, device=DEVICE)

        # User-state snapshot for diagnostics.
        with torch.no_grad():
            user_state = model.encode_user_state(event_seq.unsqueeze(0)).squeeze(0)
        user_state_log.append(user_state.detach().cpu().numpy())
        if step <= total_steps // 3:
            phase_log.append("early")
        elif step <= (2 * total_steps) // 3:
            phase_log.append("mid")
        else:
            phase_log.append("late")

        # Stage A: Recall.
        recall_indices = select_recall_candidates(
            model=model,
            event_seq=event_seq,
            all_item_features=item_features,
            available_indices=recall_pool,
        )

        # Stage B: Pre-rank.
        bad_domain_ids = set()
        for d_id, vals in recent_domain_rewards.items():
            if len(vals) >= BAD_DOMAIN_MIN_IMPRESSIONS:
                if (sum(vals) / len(vals)) <= BAD_DOMAIN_REWARD_THRESHOLD:
                    bad_domain_ids.add(d_id)

        pre_rank_indices = pre_rank_candidates(
            model=model,
            event_seq=event_seq,
            all_item_features=item_features,
            recall_indices=recall_indices,
            recent_topic_ids=recent_topic_ids,
            recent_cluster_ids=recent_cluster_ids,
            recent_domain_ids=recent_domain_ids,
            bad_domain_ids=bad_domain_ids,
            topic_map=topic_map,
            score_weights=SCORE_WEIGHTS,
        )
        if not pre_rank_indices:
            pre_rank_indices = recall_indices[: min(PRE_RANK_SIZE, len(recall_indices))]

        # Stage C: Final rank with epsilon-greedy exploration.
        pre_rank_features = item_features[pre_rank_indices]
        repeated_seq = event_seq.unsqueeze(0).repeat(len(pre_rank_indices), 1)
        with torch.no_grad():
            pre_heads = model.forward(repeated_seq, pre_rank_features)
            final_scores = model.score_from_heads(pre_heads, weights=SCORE_WEIGHTS).squeeze(1).cpu().numpy()
            positive_signals = (
                torch.sigmoid(pre_heads["like_logit"])
                + torch.sigmoid(pre_heads["share_logit"])
                + torch.sigmoid(pre_heads["rewatch_logit"])
                - torch.sigmoid(pre_heads["fast_skip_logit"])
            ).squeeze(1).cpu().numpy()

        ordered = sorted(
            list(zip(pre_rank_indices, final_scores)),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        rank_pool = ordered[: min(RANK_POOL_SIZE, len(ordered))]
        # Ensure rank pool has some predicted-positive options to avoid low-quality lock-in.
        signal_by_idx = {
            idx: float(positive_signals[i]) for i, idx in enumerate(pre_rank_indices)
        }
        positives_in_pool = sum(
            1 for idx, _ in rank_pool if signal_by_idx.get(idx, 0.0) >= POSITIVE_SIGNAL_THRESHOLD
        )
        if positives_in_pool < RANK_POSITIVE_MIN:
            positive_candidates = [
                pair
                for pair in ordered
                if signal_by_idx.get(pair[0], 0.0) >= POSITIVE_SIGNAL_THRESHOLD
                and pair[0] not in {x[0] for x in rank_pool}
            ]
            needed = RANK_POSITIVE_MIN - positives_in_pool
            if needed > 0 and positive_candidates:
                rank_pool = rank_pool[:-needed] + positive_candidates[:needed]
                rank_pool = sorted(rank_pool, key=lambda x: float(x[1]), reverse=True)
        # Dynamic exploration: increase epsilon for users with recent negative reward.
        user_reward_history = state["reward_log"]
        if len(user_reward_history) > 5:
            recent_avg = sum(user_reward_history[-5:]) / 5.0
        else:
            recent_avg = 0.0
        if recent_avg < 0.0:
            # Panic mode for unhappy users.
            epsilon = PANIC_EPS
        else:
            # Cruise mode with simple decay.
            epsilon = epsilon_for_step(step=step, total_steps=total_steps)

        # Global recovery mode: if rolling reward over last 50 steps stays below 0
        # for more than N consecutive steps, force wide exploration.
        global_recovery_active = False
        if len(reward_log) >= 50:
            rolling_reward_50_now = sum(reward_log[-50:]) / 50.0
            if rolling_reward_50_now < GLOBAL_RECOVERY_ROLLING_THRESHOLD:
                global_low_roll_streak += 1
            else:
                global_low_roll_streak = 0
            if global_low_roll_streak > GLOBAL_RECOVERY_STREAK_STEPS:
                epsilon = max(epsilon, GLOBAL_RECOVERY_EPS)
                global_recovery_active = True

        if random.random() < epsilon:
            explore_top_m = min(5, len(rank_pool))
            winner_idx = random.choice(rank_pool[:explore_top_m])[0]
            decision = "explore"
        else:
            winner_idx = rank_pool[0][0]
            decision = "exploit"

        hard_negative = None
        if HARD_NEGATIVE_MINING:
            hard_negative = mine_hard_negative(
                bot=bot,
                step=user_step,
                candidate_indices=pre_rank_indices,
                candidate_scores=final_scores,
                winner_idx=winner_idx,
                all_item_features=item_features,
                headlines=headlines,
                topic_map=topic_map,
                vibe_map=vibe_map,
            )

        recent_seen.append(winner_idx)
        recent_seen_set.add(winner_idx)
        if len(recent_seen) == recent_seen.maxlen:
            # Keep set in sync with fixed-size deque.
            recent_seen_set = set(recent_seen)
        winner_row = df.iloc[winner_idx]
        winner_features = item_features[winner_idx].unsqueeze(0)
        topic_id = int(winner_features[0, ITEM_COL_TOPIC_ID].item())
        vibe_id = int(winner_features[0, ITEM_COL_VIBE_ID].item())
        cluster_id = int(winner_features[0, ITEM_COL_TOPIC_CLUSTER_ID].item())
        item_id = int(winner_features[0, ITEM_COL_ITEM_ID].item())

        session = bot.simulate_session(
            topic_id=topic_id,
            vibe_id=vibe_id,
            topic_map=topic_map,
            vibe_map=vibe_map,
            headline=headlines[winner_idx],
            cluster_id=cluster_id,
            session_index=user_step,
            update_state=True,
        )
        reward = calculate_addiction_reward(session)
        reward_norm = calibrate_reward(reward)
        reward_log.append(reward)
        state["reward_log"].append(reward)
        if topic_id not in recent_domain_rewards:
            recent_domain_rewards[topic_id] = deque(maxlen=12)
        recent_domain_rewards[topic_id].append(reward)

        # Multi-head training target from session packet.
        targets = build_targets(session)
        optimizer.zero_grad()
        heads = model.forward(event_seq.unsqueeze(0), winner_features)
        pred_score_tensor = model_score_from_heads(heads, SCORE_WEIGHTS)
        pred_score = float(pred_score_tensor.item())
        loss, _ = compute_total_loss(
            heads=heads,
            targets=targets,
            reward_norm=reward_norm,
            bce=bce,
            mse=mse,
            score_weights=SCORE_WEIGHTS,
        )

        hard_loss_value = 0.0
        hard_conf = 0.0
        if hard_negative is not None:
            hn_idx = int(hard_negative["idx"])
            hard_conf = float(hard_negative["confidence"])
            hard_features = item_features[hn_idx].unsqueeze(0)
            hard_heads = model.forward(event_seq.unsqueeze(0), hard_features)
            hard_score = model_score_from_heads(hard_heads, SCORE_WEIGHTS)
            hard_target = torch.tensor(
                [[HARD_NEGATIVE_TARGET_NORM]],
                dtype=torch.float32,
                device=DEVICE,
            )
            hard_loss = mse(torch.sigmoid(hard_score), hard_target)
            loss = loss + (HARD_NEGATIVE_WEIGHT * hard_loss)
            hard_loss_value = float(hard_loss.item())

        current_loss = float(loss.item())
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM))
        optimizer.step()

        # Keep rare high-reward patterns alive with lightweight replay.
        if reward >= REPLAY_POS_THRESHOLD:
            replay_buffer.append(
                {
                    "event_seq": event_seq.detach().cpu(),
                    "item_features": winner_features.detach().cpu(),
                    "targets": {k: v.detach().cpu() for k, v in targets.items()},
                    "reward_norm": float(reward_norm),
                }
            )
        if replay_buffer and REPLAY_UPDATES_PER_STEP > 0:
            replay_steps = min(REPLAY_UPDATES_PER_STEP, len(replay_buffer))
            for _ in range(replay_steps):
                sample = random.choice(list(replay_buffer))
                rs_event = sample["event_seq"].to(DEVICE)
                rs_item = sample["item_features"].to(DEVICE)
                rs_targets = {
                    k: v.to(DEVICE) for k, v in sample["targets"].items()  # type: ignore[union-attr]
                }
                rs_reward_norm = float(sample["reward_norm"])
                optimizer.zero_grad()
                rs_heads = model.forward(rs_event.unsqueeze(0), rs_item)
                rs_loss, _ = compute_total_loss(
                    heads=rs_heads,
                    targets=rs_targets,
                    reward_norm=rs_reward_norm,
                    bce=bce,
                    mse=mse,
                    score_weights=SCORE_WEIGHTS,
                )
                (0.5 * rs_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

        # Append new event token to sequence state.
        token = build_event_token(
            item_id=item_id,
            topic_id=topic_id,
            vibe_id=vibe_id,
            session=session,
            step=user_step,
            user_tag=active_bot_name,
        )
        event_seq_tokens.append(token)
        recent_topic_ids.append(topic_id)
        recent_cluster_ids.append(cluster_id)
        recent_domain_ids.append(topic_id)

        # Regret evaluation.
        regret = None
        if step % REGRET_EVERY == 0:
            regret = counterfactual_regret(
                bot=bot,
                step=user_step,
                available_indices=available_indices,
                all_item_features=item_features,
                headlines=headlines,
                topic_map=topic_map,
                vibe_map=vibe_map,
                chosen_reward=reward,
            )
            regret_log.append(regret)
            state["regret_log"].append(regret)

        append_usage_log_row(
            USAGE_LOG_PATH,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source": "train_real",
                "run_id": run_id,
                "session_id": active_bot_name,
                "step": step,
                "user_step": user_step,
                "user_type": active_bot_name,
                "item_id": item_id,
                "topic_id": topic_id,
                "topic_name": topic_map.get(topic_id, "unknown"),
                "vibe_id": vibe_id,
                "vibe_style": vibe_map.get(vibe_id, "unknown"),
                "cluster_id": cluster_id,
                "headline": headlines[winner_idx],
                "reward": float(reward),
                "regret": float(regret) if regret is not None else "",
                "epsilon": float(epsilon),
                "decision": decision,
                "pred_score": float(pred_score),
                "listen_ms": float(session.get("listen_ms", 0.0)),
                "total_ms": float(session.get("total_ms", 10000.0)),
                "liked": bool(session.get("liked", False)),
                "shared": bool(session.get("shared", False)),
                "rewinded": bool(session.get("rewinded", False)),
                "fast_skip": bool(session.get("fast_skip", False)),
                "finished": bool(session.get("finished", False)),
                "interest": float(session.get("interest", 0.0)),
            },
        )

        avg5 = sum(reward_log[-5:]) / max(1, min(5, len(reward_log)))
        regret_txt = f"{regret:.2f}" if regret is not None else "-"
        headline = str(winner_row["Headline"])[:46]
        topic_name = topic_map.get(topic_id, "unknown")
        vibe_name = vibe_map.get(vibe_id, "unknown")
        print(
            f"[Step {step:03d}] user:{active_bot_name:5s} loss:{current_loss:6.3f} grad:{grad_norm:6.3f} "
            f"pred:{pred_score:6.3f} reward:{reward:6.2f} eps:{epsilon:.2f} {decision:7s} "
            f"regret:{regret_txt:>5s} hn:{hard_loss_value:5.3f} hc:{hard_conf:4.2f} "
            f"| {topic_name} | {vibe_name} | {headline}"
        )
        if global_recovery_active:
            print(
                f"  └─ global_recovery_eps={GLOBAL_RECOVERY_EPS:.2f} "
                f"(rolling_reward_50<0 for {global_low_roll_streak} consecutive steps)"
            )
        if step % 50 == 0:
            rolling_reward = sum(reward_log[-50:]) / max(1, min(50, len(reward_log)))
            print(
                f"  └─ checkpoint step={step} rolling_reward_50={rolling_reward:.2f} "
                f"mean_regret_so_far={sum(regret_log)/max(1, len(regret_log)):.2f}"
            )
        if bool(session.get("quit_session", False)):
            if bot_mode == "multi":
                print(
                    f"⏸️ {bot.name} ended session early at step={step}; "
                    "resetting only short-term session state."
                )
                if isinstance(bot, SarahBot):
                    bot.session_ended = False
                    bot.recent_vibes.clear()
                state["recent_topic_ids"] = deque(maxlen=10)
                state["recent_cluster_ids"] = deque(maxlen=10)
                state["recent_domain_ids"] = deque(maxlen=10)
                state["event_seq_tokens"] = deque([0] * EVENT_SEQ_LEN, maxlen=EVENT_SEQ_LEN)
                state["recent_domain_rewards"] = {}
                state["session_step"] = 0
            else:
                print(f"⏹️ {bot.name} ended session early at step={step} due to bad-vibes fatigue.")
                break

    final_reward = sum(reward_log[-10:]) / max(1, min(10, len(reward_log)))
    mean_regret = sum(regret_log) / len(regret_log) if regret_log else 0.0
    print(f"\n🏁 Final Reward(10): {final_reward:.2f} | Mean Regret: {mean_regret:.2f}")
    if len(user_states) > 1:
        print("👥 Per-user summary:")
        for uname, st in user_states.items():
            u_rewards = st["reward_log"]
            u_regrets = st["regret_log"]
            u_final = sum(u_rewards[-10:]) / max(1, min(10, len(u_rewards))) if u_rewards else 0.0
            u_mean_regret = sum(u_regrets) / len(u_regrets) if u_regrets else 0.0
            print(
                f"  {uname}: interactions={len(u_rewards)} "
                f"final_reward_10={u_final:.2f} mean_regret={u_mean_regret:.2f}"
            )

    # User-state embedding interpretability check.
    if len(user_state_log) >= 9:
        embed_mat = np.stack(user_state_log, axis=0)
        n_clusters = min(3, len(embed_mat))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embed_mat)
        print("🔬 User-state cluster by session phase:")
        for cluster_id in range(n_clusters):
            counts = Counter(
                phase_log[i] for i, lbl in enumerate(labels.tolist()) if lbl == cluster_id
            )
            print(f"  cluster_{cluster_id}: {dict(counts)}")

    should_save = (not save_only_if_good) or (final_reward >= save_reward_threshold)
    if should_save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "meta": {
                    "num_items": num_items,
                    "num_topics": num_topics,
                    "num_vibes": num_vibes,
                    "num_topic_clusters": num_clusters,
                    "num_entities": num_entities,
                    "num_events": num_events,
                    "num_recency_buckets": num_recency,
                    "num_popularity_buckets": num_popularity,
                    "event_vocab_size": EVENT_VOCAB_SIZE,
                },
            },
            save_path,
        )
        print(f"💾 Warm-start checkpoint saved to: {save_path}")
    else:
        print(
            f"⏭️ Skipped warm-start save (final_reward={final_reward:.2f} < "
            f"{save_reward_threshold:.2f})."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=STEPS,
        help="Number of training steps to run.",
    )
    parser.add_argument(
        "--bot",
        type=str,
        default="dave",
        choices=[
            "dave",
            "sarah",
            "gamer",
            "investor",
            "low_mood_audit",
            "ai_doom",
            "contrarian_chaos",
            "multi",
        ],
        help="Synthetic user profile to simulate.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DATA_PATH,
        help="Path to scraped articles CSV.",
    )
    parser.add_argument(
        "--load-warm-start",
        action="store_true",
        help="Load checkpoint from --warm-start-path before training.",
    )
    parser.add_argument(
        "--warm-start-path",
        type=str,
        default=str(WARM_START_PATH),
        help="Checkpoint file path for loading warm-start state.",
    )
    parser.add_argument(
        "--save-warm-start-path",
        type=str,
        default=None,
        help="Checkpoint file path for saving. Defaults to --warm-start-path.",
    )
    parser.add_argument(
        "--always-save-warm-start",
        action="store_true",
        help="Always save a checkpoint at the end of run (ignore reward threshold).",
    )
    parser.add_argument(
        "--save-reward-threshold",
        type=float,
        default=SAVE_REWARD_THRESHOLD,
        help="Minimum final reward required to save checkpoint when not using --always-save-warm-start.",
    )
    args = parser.parse_args()
    main(
        total_steps=args.steps,
        bot_name=args.bot,
        data_path=args.data_path,
        load_warm_start=args.load_warm_start,
        warm_start_path=args.warm_start_path,
        save_warm_start_path=args.save_warm_start_path,
        save_only_if_good=(not args.always_save_warm_start),
        save_reward_threshold=args.save_reward_threshold,
    )
