from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple model checkpoints into one checkpoint via weighted averaging."
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="Input checkpoint paths in merge order (first checkpoint acts as anchor).",
    )
    parser.add_argument(
        "--weights",
        default="",
        help=(
            "Comma-separated weights (same count as checkpoints). "
            "If omitted, uniform weights are used."
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/final_mega.pt",
        help="Output checkpoint path.",
    )
    return parser.parse_args()


def parse_weights(raw: str, n: int) -> list[float]:
    if not raw.strip():
        return [1.0 / n] * n
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != n:
        raise ValueError(f"--weights count ({len(parts)}) must equal checkpoints count ({n}).")
    values = [float(p) for p in parts]
    total = sum(values)
    if total <= 0:
        raise ValueError("Sum of weights must be > 0.")
    return [v / total for v in values]


def load_checkpoint(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict.")
    if "model_state_dict" not in ckpt or not isinstance(ckpt["model_state_dict"], dict):
        raise ValueError(f"Checkpoint at {path} missing valid model_state_dict.")
    return ckpt


def merge_meta(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ckpt in checkpoints:
        meta = ckpt.get("meta", {})
        if not isinstance(meta, dict):
            continue
        for k, v in meta.items():
            if isinstance(v, (int, float)):
                out[k] = max(int(out.get(k, 0)), int(v))
            elif k not in out:
                out[k] = v
    return out


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.checkpoints]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")

    weights = parse_weights(args.weights, len(paths))
    checkpoints = [load_checkpoint(p) for p in paths]

    anchor_state: dict[str, torch.Tensor] = checkpoints[0]["model_state_dict"]
    all_states: list[dict[str, torch.Tensor]] = [c["model_state_dict"] for c in checkpoints]

    merged_state: dict[str, torch.Tensor] = {}
    averaged_keys = 0
    copied_anchor_keys = 0

    for key, anchor_tensor in anchor_state.items():
        tensors: list[torch.Tensor] = []
        compatible = True
        for state in all_states:
            t = state.get(key)
            if t is None or t.shape != anchor_tensor.shape or t.dtype != anchor_tensor.dtype:
                compatible = False
                break
            tensors.append(t)

        if compatible and anchor_tensor.is_floating_point():
            acc = torch.zeros_like(anchor_tensor)
            for w, t in zip(weights, tensors):
                acc = acc + (t * w)
            merged_state[key] = acc
            averaged_keys += 1
        else:
            merged_state[key] = anchor_tensor.clone()
            copied_anchor_keys += 1

    merged = {
        "model_state_dict": merged_state,
        "meta": merge_meta(checkpoints),
        "merged_from": [str(p) for p in paths],
        "merge_weights": weights,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)

    print(f"Merged checkpoint saved to: {output_path}")
    print(f"Averaged keys: {averaged_keys}")
    print(f"Copied anchor-only keys: {copied_anchor_keys}")
    print(f"Sources: {[str(p) for p in paths]}")
    print(f"Weights: {weights}")


if __name__ == "__main__":
    main()
