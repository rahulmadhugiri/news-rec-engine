#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from requests import RequestException

from user_bot import (
    AIDoomBot,
    ContrarianChaosBot,
    DaveBot,
    GamerBot,
    InvestorBot,
    LowMoodAuditBot,
    SarahBot,
)


PersonaFactory = Callable[[int], object]

PERSONA_FACTORIES: dict[str, PersonaFactory] = {
    "dave": lambda seed: DaveBot(name="dave", seed=seed),
    "sarah": lambda seed: SarahBot(name="sarah", seed=seed),
    "gamer": lambda seed: GamerBot(name="gamer", seed=seed),
    "investor": lambda seed: InvestorBot(name="investor", seed=seed),
    "low_mood_audit": lambda seed: LowMoodAuditBot(name="low_mood_audit", seed=seed),
    "ai_doom": lambda seed: AIDoomBot(name="ai_doom", seed=seed),
    "contrarian_chaos": lambda seed: ContrarianChaosBot(name="contrarian_chaos", seed=seed),
}


def extract_domain(url: str) -> str:
    clean = str(url).replace("https://", "").replace("http://", "")
    return clean.split("/")[0].replace("www.", "").strip().lower() or "unknown"


def load_inventory(data_path: Path) -> tuple[dict[int, dict[str, int | str]], dict[int, str], dict[int, str]]:
    df = pd.read_csv(data_path)

    if "Topic_Name" not in df.columns:
        link_col = "Link" if "Link" in df.columns else "link"
        df["Topic_Name"] = df[link_col].fillna("").astype(str).apply(extract_domain)
    if "Topic_ID" not in df.columns:
        df["Topic_ID"], _ = pd.factorize(df["Topic_Name"])

    style_col = "Vibe_Style"
    if style_col not in df.columns:
        style_col = "Suggested_Style" if "Suggested_Style" in df.columns else ""
    if style_col:
        df["Vibe_Style"] = df[style_col].fillna("Standard Reporting").astype(str)
    else:
        df["Vibe_Style"] = "Standard Reporting"
    if "Vibe_ID" not in df.columns:
        df["Vibe_ID"], _ = pd.factorize(df["Vibe_Style"])

    if "Topic_Cluster_ID" not in df.columns:
        df["Topic_Cluster_ID"] = 0

    headline_col = "Headline" if "Headline" in df.columns else "title"
    if headline_col not in df.columns:
        raise ValueError("Missing headline column (expected Headline or title)")

    df["id"] = range(len(df))

    id_to_meta: dict[int, dict[str, int | str]] = {}
    for _, row in df.iterrows():
        item_id = int(row["id"])
        id_to_meta[item_id] = {
            "topic_id": int(row["Topic_ID"]),
            "vibe_id": int(row["Vibe_ID"]),
            "headline": str(row[headline_col]),
            "cluster_id": int(row["Topic_Cluster_ID"]),
        }

    topic_map = (
        df[["Topic_ID", "Topic_Name"]]
        .drop_duplicates(subset=["Topic_ID"])
        .set_index("Topic_ID")["Topic_Name"]
        .astype(str)
        .to_dict()
    )
    vibe_map = (
        df[["Vibe_ID", "Vibe_Style"]]
        .drop_duplicates(subset=["Vibe_ID"])
        .set_index("Vibe_ID")["Vibe_Style"]
        .astype(str)
        .to_dict()
    )
    return id_to_meta, topic_map, vibe_map


def run_one_session(
    base_url: str,
    bot: object,
    id_to_meta: dict[int, dict[str, int | str]],
    topic_map: dict[int, str],
    vibe_map: dict[int, str],
    steps: int,
    timeout_sec: float,
) -> dict[str, float]:
    rewards: list[float] = []
    watch_ratios: list[float] = []
    auto_count = 0
    skip_count = 0
    feed_items: list[dict[str, object]] = []
    feed_pos = 0

    with requests.Session() as s:
        try:
            s.get(f"{base_url}/", timeout=timeout_sec)
        except RequestException as exc:
            raise RuntimeError(
                "Could not reach live server at "
                f"{base_url}. Start your API first (e.g. "
                "`./venv/bin/uvicorn main:app --reload --port 8000`) "
                "or run ab_eval_live.py with --checkpoint-a so it auto-starts."
            ) from exc

        for step_idx in range(steps):
            if feed_pos >= len(feed_items):
                try:
                    feed_resp = s.get(f"{base_url}/feed", timeout=timeout_sec)
                    feed_resp.raise_for_status()
                except RequestException as exc:
                    raise RuntimeError(
                        f"Lost connection to {base_url}/feed during evaluation."
                    ) from exc
                feed_items = feed_resp.json().get("items", [])
                feed_pos = 0
                if not feed_items:
                    break

            item = feed_items[feed_pos]
            feed_pos += 1
            item_id = int(item["id"])
            meta = id_to_meta.get(item_id)
            if meta is None:
                continue

            packet = bot.simulate_session(
                topic_id=int(meta["topic_id"]),
                vibe_id=int(meta["vibe_id"]),
                topic_map=topic_map,
                vibe_map=vibe_map,
                headline=str(meta["headline"]),
                cluster_id=int(meta["cluster_id"]),
                session_index=step_idx,
                update_state=True,
            )

            listen_ms = float(packet.get("listen_ms", 1000.0))
            total_ms = max(1.0, float(packet.get("total_ms", 10000.0)))
            dwell = max(0.0, min(10.0, listen_ms / 1000.0))
            reason = "auto" if bool(packet.get("finished", False)) else "skip"
            if reason == "auto":
                auto_count += 1
            else:
                skip_count += 1

            try:
                interact_resp = s.post(
                    f"{base_url}/interact",
                    json={"item_id": item_id, "reason": reason, "dwell_time": dwell},
                    timeout=timeout_sec,
                )
                interact_resp.raise_for_status()
            except RequestException as exc:
                raise RuntimeError(
                    f"Lost connection to {base_url}/interact during evaluation."
                ) from exc
            body = interact_resp.json()

            rewards.append(float(body.get("reward", 0.0)))
            watch_ratios.append(max(0.0, min(1.0, listen_ms / total_ms)))

            if bool(packet.get("quit_session", False)):
                break

    n = len(rewards)
    if n == 0:
        return {
            "steps": 0.0,
            "avg_reward": 0.0,
            "auto_rate": 0.0,
            "skip_rate": 0.0,
            "avg_watch_ratio": 0.0,
        }
    return {
        "steps": float(n),
        "avg_reward": float(sum(rewards) / n),
        "auto_rate": float(auto_count / n),
        "skip_rate": float(skip_count / n),
        "avg_watch_ratio": float(sum(watch_ratios) / n),
    }


def mean_of(items: list[float]) -> float:
    return float(statistics.fmean(items)) if items else 0.0


def eval_condition(
    name: str,
    base_url: str,
    personas: list[str],
    sessions_per_persona: int,
    steps_per_session: int,
    seed_base: int,
    id_to_meta: dict[int, dict[str, int | str]],
    topic_map: dict[int, str],
    vibe_map: dict[int, str],
    timeout_sec: float,
) -> dict[str, object]:
    print(f"\n=== Evaluating {name} @ {base_url} ===")
    per_persona: dict[str, dict[str, float]] = {}
    all_rewards: list[float] = []
    all_steps: list[float] = []

    for p_idx, persona in enumerate(personas):
        factory = PERSONA_FACTORIES[persona]
        runs: list[dict[str, float]] = []
        for s_idx in range(sessions_per_persona):
            bot_seed = seed_base + (p_idx * 1000) + s_idx
            bot = factory(bot_seed)
            metrics = run_one_session(
                base_url=base_url,
                bot=bot,
                id_to_meta=id_to_meta,
                topic_map=topic_map,
                vibe_map=vibe_map,
                steps=steps_per_session,
                timeout_sec=timeout_sec,
            )
            runs.append(metrics)

        per_persona[persona] = {
            "sessions": float(len(runs)),
            "avg_steps": mean_of([r["steps"] for r in runs]),
            "avg_reward": mean_of([r["avg_reward"] for r in runs]),
            "auto_rate": mean_of([r["auto_rate"] for r in runs]),
            "skip_rate": mean_of([r["skip_rate"] for r in runs]),
            "avg_watch_ratio": mean_of([r["avg_watch_ratio"] for r in runs]),
        }
        all_rewards.extend([r["avg_reward"] for r in runs])
        all_steps.extend([r["steps"] for r in runs])
        print(
            f"{persona:17s} reward={per_persona[persona]['avg_reward']:.2f} "
            f"auto={per_persona[persona]['auto_rate']:.2f} steps={per_persona[persona]['avg_steps']:.1f}"
        )

    overall = {
        "avg_reward_over_sessions": mean_of(all_rewards),
        "avg_steps_over_sessions": mean_of(all_steps),
    }
    return {"name": name, "overall": overall, "personas": per_persona}


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    log_file: object
    base_url: str


def wait_for_server(base_url: str, timeout_sec: float) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            resp = requests.get(f"{base_url}/feed", timeout=1.5)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.4)
    raise RuntimeError(f"Server at {base_url} did not become ready in {timeout_sec:.1f}s")


def start_server(
    name: str,
    checkpoint_path: Path,
    update_mode: str,
    data_path: Path,
    outputs_dir: Path,
    port: int,
) -> ServerHandle:
    env = os.environ.copy()
    env["ONLINE_UPDATE_MODE"] = update_mode
    env["WARM_START_PATH"] = str(checkpoint_path)
    env["DATA_PATH"] = str(data_path)
    env["SESSION_STATE_PATH"] = str(outputs_dir / f"live_state_{name}.json")
    env["USAGE_LOG_PATH"] = str(outputs_dir / f"live_usage_{name}.csv")

    log_path = outputs_dir / f"server_{name}.log"
    log_file = log_path.open("w", encoding="utf-8")
    cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    base_url = f"http://127.0.0.1:{port}"

    try:
        wait_for_server(base_url=base_url, timeout_sec=30.0)
    except Exception as exc:
        proc.terminate()
        proc.wait(timeout=5)
        log_file.close()
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-20:])
        except Exception:
            pass
        extra = f"\nServer log tail ({log_path}):\n{tail}" if tail else ""
        raise RuntimeError(f"Failed to start server '{name}': {exc}{extra}") from exc
    return ServerHandle(proc=proc, log_file=log_file, base_url=base_url)


def stop_server(handle: ServerHandle) -> None:
    try:
        handle.proc.terminate()
        handle.proc.wait(timeout=8)
    except Exception:
        handle.proc.kill()
    finally:
        handle.log_file.close()


def parse_personas(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(PERSONA_FACTORIES.keys())
    personas = [p.strip() for p in raw.split(",") if p.strip()]
    unknown = [p for p in personas if p not in PERSONA_FACTORIES]
    if unknown:
        raise ValueError(f"Unknown personas: {unknown}. Valid: {list(PERSONA_FACTORIES.keys())}")
    return personas


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated persona A/B evaluator for live recommendation API.")
    parser.add_argument("--steps-per-session", type=int, default=100)
    parser.add_argument("--sessions-per-persona", type=int, default=3)
    parser.add_argument("--personas", type=str, default="all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--data-path", type=Path, default=Path("data/scraped_articles_backup.csv"))
    parser.add_argument("--output", type=Path, default=None)

    parser.add_argument("--base-url", type=str, default="")

    parser.add_argument("--name-a", type=str, default="A")
    parser.add_argument("--checkpoint-a", type=Path, default=None)
    parser.add_argument("--mode-a", type=str, default="private")
    parser.add_argument("--name-b", type=str, default="B")
    parser.add_argument("--checkpoint-b", type=Path, default=None)
    parser.add_argument("--mode-b", type=str, default="private")
    args = parser.parse_args()

    personas = parse_personas(args.personas)
    id_to_meta, topic_map, vibe_map = load_inventory(args.data_path)
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or (outputs_dir / f"ab_live_eval_{now}.json")

    results: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps_per_session": args.steps_per_session,
        "sessions_per_persona": args.sessions_per_persona,
        "personas": personas,
        "conditions": {},
    }

    if args.base_url:
        condition = eval_condition(
            name="live",
            base_url=args.base_url.rstrip("/"),
            personas=personas,
            sessions_per_persona=args.sessions_per_persona,
            steps_per_session=args.steps_per_session,
            seed_base=args.seed,
            id_to_meta=id_to_meta,
            topic_map=topic_map,
            vibe_map=vibe_map,
            timeout_sec=args.timeout_sec,
        )
        results["conditions"]["live"] = condition
    else:
        if args.checkpoint_a is None:
            raise ValueError("Provide --base-url, or provide --checkpoint-a (and optionally --checkpoint-b)")

        handles: list[tuple[str, ServerHandle]] = []
        try:
            handle_a = start_server(
                name=args.name_a,
                checkpoint_path=args.checkpoint_a,
                update_mode=args.mode_a,
                data_path=args.data_path,
                outputs_dir=outputs_dir,
                port=8011,
            )
            handles.append((args.name_a, handle_a))

            if args.checkpoint_b is not None:
                handle_b = start_server(
                    name=args.name_b,
                    checkpoint_path=args.checkpoint_b,
                    update_mode=args.mode_b,
                    data_path=args.data_path,
                    outputs_dir=outputs_dir,
                    port=8012,
                )
                handles.append((args.name_b, handle_b))

            for idx, (name, handle) in enumerate(handles):
                condition = eval_condition(
                    name=name,
                    base_url=handle.base_url,
                    personas=personas,
                    sessions_per_persona=args.sessions_per_persona,
                    steps_per_session=args.steps_per_session,
                    seed_base=args.seed + (idx * 50000),
                    id_to_meta=id_to_meta,
                    topic_map=topic_map,
                    vibe_map=vibe_map,
                    timeout_sec=args.timeout_sec,
                )
                results["conditions"][name] = condition
        finally:
            for _, handle in handles:
                stop_server(handle)

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
