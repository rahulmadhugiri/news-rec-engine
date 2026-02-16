"""Nightly pipeline: demand analysis -> adaptive scrape."""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_cmd(cmd: list[str]) -> None:
    print(f"▶ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--max-queries", type=int, default=24)
    parser.add_argument("--max-domains", type=int, default=12)
    parser.add_argument("--min-domain-impressions", type=int, default=6)
    parser.add_argument("--daily-pool-k", type=int, default=150)
    args = parser.parse_args()

    py = sys.executable

    run_cmd(
        [
            py,
            "analyze_demand.py",
            "--lookback-days",
            str(args.lookback_days),
            "--max-queries",
            str(args.max_queries),
            "--max-domains",
            str(args.max_domains),
            "--min-domain-impressions",
            str(args.min_domain_impressions),
        ]
    )
    run_cmd([py, "scraper.py"])
    run_cmd([py, "morning_harvest.py", "--k", str(args.daily_pool_k)])
    print("✅ Nightly refresh complete.")


if __name__ == "__main__":
    main()
