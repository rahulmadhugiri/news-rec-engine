"""Online learning simulation for a two-tower news recommender."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from model import GenerativeTwoTower
from user_bot import UserBot


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)

    # Catalog settings.
    num_users = 1
    num_topics = 12
    num_vibes = 8
    history_dim = 16
    num_steps = 2000
    candidates_per_step = 20
    log_interval = 50
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic ground-truth user ("Dave").
    dave = UserBot(
        name="Dave",
        liked_topics=[1, 3, 6],
        liked_vibes=[2, 5],
        hated_topics=[9, 10],
    )

    # Two-tower model + optimizer for online updates.
    model = GenerativeTwoTower(
        num_users=num_users,
        num_topics=num_topics,
        num_vibes=num_vibes,
        history_dim=history_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Single-user setup: fixed ID and simple running history signal.
    user_id = torch.tensor([0], dtype=torch.long)
    history_vector = torch.zeros(1, history_dim, dtype=torch.float32)

    rewards: list[int] = []
    hit_rates: list[float] = []
    hit_rate_steps: list[int] = []
    user_history: list[tuple[int, int]] = []

    model.train()
    for step in range(1, num_steps + 1):
        # 1) Generate random candidate items (topic, vibe).
        candidate_topics = torch.randint(0, num_topics, (candidates_per_step,), dtype=torch.long)
        candidate_vibes = torch.randint(0, num_vibes, (candidates_per_step,), dtype=torch.long)

        # 2) Score all candidates and pick the highest-scoring winner.
        repeated_user_id = user_id.repeat(candidates_per_step)
        repeated_history = history_vector.repeat(candidates_per_step, 1)

        candidate_logits = model(
            user_id=repeated_user_id,
            history_vector=repeated_history,
            item_topic_id=candidate_topics,
            item_vibe_id=candidate_vibes,
        )
        winner_idx = int(torch.argmax(candidate_logits).item())
        winner_topic = int(candidate_topics[winner_idx].item())
        winner_vibe = int(candidate_vibes[winner_idx].item())
        winner_logit = candidate_logits[winner_idx].view(1, 1)

        # Keep only the latest 50 served winner items in user history.
        user_history.append((winner_topic, winner_vibe))
        if len(user_history) > 50:
            user_history = user_history[-50:]

        # 3) Get bot feedback for winner and convert to binary reward.
        raw_feedback = dave.interact(topic_id=winner_topic, vibe_id=winner_vibe)
        reward = 1 if raw_feedback >= 0.5 else 0
        rewards.append(reward)

        # 4) Immediate online update from this single interaction.
        target = torch.tensor([[float(reward)]], dtype=torch.float32)
        loss = criterion(winner_logit, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update a compact history vector with recent outcomes.
        history_vector = torch.roll(history_vector, shifts=-1, dims=1)
        history_vector[0, -1] = float(reward)

        # 5) Log recent hit rate every fixed interval.
        if step % log_interval == 0:
            recent_rewards = rewards[-log_interval:]
            recent_hit_rate = sum(recent_rewards) / len(recent_rewards)
            hit_rates.append(recent_hit_rate)
            hit_rate_steps.append(step)
            print(f"Step {step:4d} | Recent Hit Rate ({log_interval}): {recent_hit_rate:.3f}")

    # Plot learning curve.
    plt.figure(figsize=(10, 5))
    plt.plot(hit_rate_steps, hit_rates, marker="o")
    plt.title("Online Learning Curve: Recent Hit Rate Over Time")
    plt.xlabel("Step")
    plt.ylabel(f"Hit Rate (Last {log_interval} Swipes)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / "learning_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()

    checkpoint_path = output_dir / "generative_two_tower_dave.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "num_users": num_users,
                "num_topics": num_topics,
                "num_vibes": num_vibes,
                "history_dim": history_dim,
                "num_steps": num_steps,
                "candidates_per_step": candidates_per_step,
                "log_interval": log_interval,
            },
            "dave_profile": {
                "name": dave.name,
                "liked_topics": sorted(dave.liked_topics),
                "liked_vibes": sorted(dave.liked_vibes),
                "hated_topics": sorted(dave.hated_topics),
            },
            "hit_rate_steps": hit_rate_steps,
            "hit_rates": hit_rates,
            "rewards": rewards,
            "user_history": user_history,
        },
        checkpoint_path,
    )
    print(f"Saved learning curve to: {plot_path}")
    print(f"Saved model checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
