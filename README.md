# Rec Engine (News Recommender)

End-to-end news recommendation system with:
- a two-tower neural model (`model.py`)
- synthetic persona training/eval (`train_real.py`, `user_bot.py`)
- live FastAPI serving with per-session adaptation (`main.py`)
- checkpoint merging for a "mega" model (`merge_checkpoints.py`)
- automated A/B live evaluation (`ab_eval_live.py`)
- RSS/news data ingestion and refresh tooling (`scraper.py`, `analyze_demand.py`, `daily_refresh.py`)

## What This Repo Is

This is a **real ML recommender** project (not a mock/simulation-only prototype):
- You train with gradient descent in PyTorch.
- You save/load real checkpoints (`.pt`).
- You serve those checkpoints in a live API.
- You run automated evaluations against the live API.

It is intentionally small enough to iterate quickly on CPU while still using the same core ideas as larger recommender systems.

## High-Level Architecture

### 1) Model (`model.py`)
`GenerativeTwoTower`:
- **User tower**:
  - event-token embedding
  - GRU sequence encoder
  - MLP to a 64d user vector
- **Item tower**:
  - embeddings for item/topic/vibe/cluster/entity/event/recency/popularity features
  - MLP to a 64d item vector
- **Interaction + multi-head outputs**:
  - finish, like, share, rewatch, fast_skip, watch_time
- **Final ranking score**:
  - weighted combination of those heads

Expected item feature order:
- `Item_ID, Topic_ID, Vibe_ID, Topic_Cluster_ID, Entity_ID, Event_ID, Recency_Bucket, Popularity_Bucket`

### 2) Serving API (`main.py`)
FastAPI endpoints:
- `GET /` -> serves `index.html` and assigns session cookie `vibe_session_id`
- `GET /feed` -> returns ranked feed items:
  - `id, title, description, domain, cluster_id, link`
  - optional: `GET /feed?k=200` returns top-K by relevance (default keeps current behavior)
- `POST /interact` -> ingests interaction:
  - payload: `{ "item_id": int, "reason": "auto"|"skip", "dwell_time": float_seconds }`
- `POST /next_audio` -> *MVP "Content Refinery"* (Stage 2-4):
  - picks the next item using the existing rec engine but restricted to the daily flat pool (150)
  - generates an exactly-4-sentence script with behavioral memory (last 5 positives/negatives)
  - returns ElevenLabs MP3 as base64
  - requires env: `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`
- `POST /audio_event` -> sentence-precise feedback for the MVP audio flow:
  - payload (recommended): `{ "content_id": str, "skip_index": int|null, "listen_ms": number, "total_ms": number }`
  - updates both: (a) synthesis memory window, (b) rec-engine online session state

Serving uses a cascade:
- recall -> pre-rank -> final rank + MMR diversity + dedup + exploration

### 3) Online Update Modes (critical)
Controlled by env var `ONLINE_UPDATE_MODE`:
- `private` (recommended default):
  - shared global model weights stay frozen
  - only per-session user delta vector is adapted online
- `global`:
  - global model weights updated online from live interactions
  - riskier for quality drift
- `none`:
  - no online learning, inference only

### 4) Training (`train_real.py`)
Trains on article CSV with one persona or `multi` persona mixture.
- Includes recall/pre-rank/rank candidate flow
- Hard-negative mining
- Regret estimates
- Warm-start load/save
- Persona preflight positive-rate checks

### 5) Checkpoint Merging (`merge_checkpoints.py`)
Weighted merge of multiple checkpoints:
- averages compatible floating-point tensors
- keeps anchor tensor when incompatible
- merges `meta` dimensions conservatively

### 6) Evaluation (`ab_eval_live.py`)
Two ways:
- evaluate an already-running server (`--base-url`)
- spin up two uvicorn servers automatically (A/B by checkpoint + update mode), run persona sessions, save JSON report

## Repository Layout

Core:
- `main.py` - live API server
- `model.py` - two-tower model
- `train_real.py` - primary training loop
- `user_bot.py` - synthetic personas
- `ab_eval_live.py` - live evaluator / A/B harness
- `merge_checkpoints.py` - checkpoint averaging/merge

Data + ingestion:
- `scraper.py` - main ingestion and feature-building script
- `rss_scraper.py` - wrapper to `scraper.main`
- `analyze_demand.py` - mines interaction logs to generate query/domain wishlist
- `daily_refresh.py` - orchestrates demand analysis + scrape
- `morning_harvest.py` - builds `data/daily_pool.json` (flat 150-story menu used by `/feed` + `/next_audio`)

Other:
- `index.html` - frontend shell
- `requirements.txt`
- `data/` - CSV data + live session/log files
- `outputs/` - checkpoints/logs/eval artifacts

## Setup

### Prerequisites
- Python 3.11+ (project has been used with 3.13)
- macOS/Linux shell

### Install
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (Serve the App)

Run with your best checkpoint:
```bash
ONLINE_UPDATE_MODE=private \
WARM_START_PATH=outputs/release_mega_2026-02-12.pt \
./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open:
- `http://127.0.0.1:8000`

### Important runtime env vars (`main.py`)
- `DATA_PATH` (default `data/scraped_articles.csv`)
- `INDEX_PATH` (default `index.html`)
- `WARM_START_PATH` (default `outputs/train_real_warm_start.pt`)
- `USAGE_LOG_PATH` (default `data/usage_interactions.csv`)
- `SESSION_STATE_PATH` (default `data/live_session_state.json`)
- `ONLINE_UPDATE_MODE` = `private|global|none`
- `RECALL_SIZE`, `PRE_RANK_SIZE`, `RANK_POOL_SIZE`, `MMR_DIVERSITY_LAMBDA`, etc.

### Live personalization behavior
- In `private` mode, each swipe updates only that session's user state (and user delta), not the shared global model.
- Session identity is tied to cookie `vibe_session_id`.
- Session state is persisted to `data/live_session_state.json`, so follow-up sessions can start from learned user preference state.

### API examples (`curl`)
Fetch feed:
```bash
curl -s http://127.0.0.1:8000/feed | jq '.[0:3]'
```

Send interaction:
```bash
curl -s -X POST http://127.0.0.1:8000/interact \
  -H "Content-Type: application/json" \
  -d '{"item_id": 42, "reason": "skip", "dwell_time": 2.4}'
```

Build today's flat pool (150-story menu):
```bash
./venv/bin/python morning_harvest.py --k 150
```

Generate next personalized audio (rec engine -> LLM -> ElevenLabs):
```bash
curl -s -X POST http://127.0.0.1:8000/next_audio \
  -H "Content-Type: application/json" \
  -d '{}' | jq '{content_id, item, sentences}'
```

Post feedback (finish if `skip_index=null`, swipe if `skip_index` is 0-3):
```bash
curl -s -X POST http://127.0.0.1:8000/audio_event \
  -H "Content-Type: application/json" \
  -d '{"content_id":"REPLACE_ME","skip_index":1,"listen_ms":8000,"total_ms":20000}'
```

## Training Workflows

### 1) Train a base multi-persona anchor
```bash
./venv/bin/python train_real.py \
  --bot multi \
  --steps 400 \
  --data-path data/scraped_articles_backup.csv \
  --always-save-warm-start \
  --save-warm-start-path outputs/base_multi_16k_anchor.pt
```

### 2) Persona-specific finetunes (example)
```bash
./venv/bin/python train_real.py \
  --bot contrarian_chaos \
  --steps 200 \
  --data-path data/scraped_articles_backup.csv \
  --load-warm-start \
  --warm-start-path outputs/base_multi_16k_anchor.pt \
  --always-save-warm-start \
  --save-warm-start-path outputs/final_contrarian_chaos.pt
```

Repeat for `gamer`, `ai_doom`, `investor`, etc.

### 3) Merge specialized checkpoints into mega model
```bash
./venv/bin/python merge_checkpoints.py \
  outputs/base_multi_16k_anchor.pt \
  outputs/final_contrarian_chaos.pt \
  outputs/final_gamer.pt \
  outputs/final_ai_doom.pt \
  outputs/final_investor.pt \
  --weights 0.35,0.1625,0.1625,0.1625,0.1625 \
  --output outputs/final_mega_v2.pt
```

### 4) Promote release artifact
```bash
cp outputs/final_mega_v2.pt outputs/release_mega_2026-02-12.pt
```

## Evaluation Workflows

### A) Evaluate currently running server
```bash
./venv/bin/python ab_eval_live.py \
  --base-url http://127.0.0.1:8000 \
  --data-path data/scraped_articles_backup.csv \
  --steps-per-session 100 \
  --sessions-per-persona 3 \
  --output outputs/ab_live_current.json
```

### B) A/B compare two checkpoints (auto-starts two servers)
```bash
./venv/bin/python ab_eval_live.py \
  --checkpoint-a outputs/final_mega.pt \
  --name-a mega_v1 \
  --mode-a private \
  --checkpoint-b outputs/final_mega_v2.pt \
  --name-b mega_v2 \
  --mode-b private \
  --data-path data/scraped_articles_backup.csv \
  --steps-per-session 100 \
  --sessions-per-persona 3 \
  --output outputs/ab_mega_v1_vs_v2.json
```

## Data Ingestion / Refresh

### Scrape and rebuild pool
```bash
./venv/bin/python scraper.py
```

Diversity backfill mode:
```bash
./venv/bin/python scraper.py --mode diversity
```

### Demand-driven wishlist generation
```bash
./venv/bin/python analyze_demand.py \
  --lookback-days 7 \
  --max-queries 24 \
  --max-domains 12 \
  --min-domain-impressions 6
```

### Nightly orchestration
```bash
./venv/bin/python daily_refresh.py
```

Note: wishlist support exists, but `scraper.py` currently has `USE_WISHLIST = False` by default, so generated wishlist may not affect scraping unless that is enabled in code.

## Logging and Artifacts

### `outputs/`
- model checkpoints (`*.pt`)
- run logs (`*.log`)
- evaluation JSONs (`ab_*.json`)

### `data/`
- training/inventory CSVs
- live interaction log:
  - `data/usage_interactions.csv`
- persisted session state:
  - `data/live_session_state.json`

## Troubleshooting

### `Connection refused` during evaluation
If you run with `--base-url`, ensure API is running first:
```bash
./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Then rerun `ab_eval_live.py`.

### Wrong checkpoint loaded
Set `WARM_START_PATH` explicitly when serving:
```bash
WARM_START_PATH=outputs/release_mega_2026-02-12.pt ...
```

### Results unstable in live app
- keep `ONLINE_UPDATE_MODE=private`
- avoid `global` until you have stronger guardrails and production monitoring

## Notes on Scope

- This is a serious recommender prototype with real training and serving.
- It is **not** internet-scale infra:
  - no distributed feature store
  - no large ANN serving cluster
  - no full production online-training promotion pipeline
- For early-stage products, this is often the right tradeoff: high iteration speed + measurable personalization.

## Legacy Script

`train_simulation.py` is an older simulation script kept for historical reference. The main production workflow is `train_real.py` + `main.py`.

## GitHub Publish Checklist

Before pushing publicly:
- Add a `.gitignore` so you do not commit `venv/`, cache files, and local logs by accident.
- Decide whether to commit model checkpoints:
  - if keeping them in Git, consider Git LFS for large files
  - otherwise keep release checkpoints out of repo and document download location
- Clean sensitive/local-only files from `data/` and `outputs/` (session logs, local state, ad-hoc run logs).
- Add a LICENSE file.

## License

Add a license file before publishing publicly on GitHub.
