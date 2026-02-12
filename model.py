"""Generative two-tower model with sequence-based user state and multi-head outputs."""

from __future__ import annotations

import torch
import torch.nn as nn

# Item feature column order expected by the model.
ITEM_COL_ITEM_ID = 0
ITEM_COL_TOPIC_ID = 1
ITEM_COL_VIBE_ID = 2
ITEM_COL_TOPIC_CLUSTER_ID = 3
ITEM_COL_ENTITY_ID = 4
ITEM_COL_EVENT_ID = 5
ITEM_COL_RECENCY_BUCKET = 6
ITEM_COL_POPULARITY_BUCKET = 7
NUM_ITEM_FEATURES = 8


class GenerativeTwoTower(nn.Module):
    """
    Two-tower recommender with strict bottleneck architecture:
    - User Tower: GRU(512) -> Linear(512->128) -> Linear(128->64)
    - Item Tower: Concat(Features) -> Linear(Input->512) -> Linear(512->128) -> Linear(128->64)
    - Multi-head outputs for actions + watch-time
    """

    def __init__(
        self,
        num_items: int | None = None,
        num_topics: int = 1,
        num_vibes: int = 1,
        num_topic_clusters: int | None = None,
        num_entities: int | None = None,
        num_events: int | None = None,
        num_recency_buckets: int = 5,
        num_popularity_buckets: int = 4,
        event_vocab_size: int = 4096,
        event_embedding_dim: int = 128,
        user_hidden_dim: int = 64,
        item_embedding_dim: int = 32,
        item_id_embedding_dim: int = 64,
        hidden_dim: int = 128,
        # Legacy args kept for backward compatibility with earlier scripts.
        num_users: int | None = None,  # noqa: ARG002
        history_dim: int | None = None,  # noqa: ARG002
    ) -> None:
        super().__init__()

        if num_items is None:
            num_items = max(1, num_topics * 8)
        if num_topic_clusters is None:
            num_topic_clusters = max(1, min(64, num_topics))
        if num_entities is None:
            num_entities = max(1, num_topics)
        if num_events is None:
            num_events = max(1, num_topics)

        self.event_vocab_size = event_vocab_size

        # Strict funnel dimensions.
        tower_l1 = 512
        tower_l2 = 128
        tower_l3 = 64
        latent_dim = tower_l3
        self.user_hidden_dim = latent_dim

        # User sequence encoder.
        self.event_embedding = nn.Embedding(
            num_embeddings=event_vocab_size,
            embedding_dim=event_embedding_dim,
            padding_idx=0,
        )
        self.user_encoder = nn.GRU(
            input_size=event_embedding_dim,
            hidden_size=tower_l1,
            batch_first=True,
        )
        self.user_mlp = nn.Sequential(
            nn.Linear(tower_l1, tower_l2),
            nn.ReLU(),
            nn.Linear(tower_l2, tower_l3),
        )

        # Item feature embeddings.
        self.item_id_embedding = nn.Embedding(max(1, num_items), item_id_embedding_dim)
        self.topic_embedding = nn.Embedding(max(1, num_topics), item_embedding_dim)
        self.vibe_embedding = nn.Embedding(max(1, num_vibes), item_embedding_dim)
        self.topic_cluster_embedding = nn.Embedding(
            max(1, num_topic_clusters), 16
        )
        self.entity_embedding = nn.Embedding(max(1, num_entities), 16)
        self.event_embedding_item = nn.Embedding(max(1, num_events), 16)
        self.recency_embedding = nn.Embedding(max(1, num_recency_buckets), 8)
        self.popularity_embedding = nn.Embedding(max(1, num_popularity_buckets), 8)

        item_input_dim = (
            item_id_embedding_dim
            + item_embedding_dim * 2
            + 16 * 3
            + 8 * 2
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(item_input_dim, tower_l1),
            nn.ReLU(),
            nn.Linear(tower_l1, tower_l2),
            nn.ReLU(),
            nn.Linear(tower_l2, tower_l3),
        )

        # Multi-head predictor over user/item interaction.
        interaction_dim = latent_dim * 4
        self.interaction_tower = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.finish_head = nn.Linear(hidden_dim, 1)
        self.like_head = nn.Linear(hidden_dim, 1)
        self.share_head = nn.Linear(hidden_dim, 1)
        self.rewatch_head = nn.Linear(hidden_dim, 1)
        self.fast_skip_head = nn.Linear(hidden_dim, 1)
        self.watch_time_head = nn.Linear(hidden_dim, 1)  # sigmoid => [0,1]

    def encode_user_state(self, event_tokens: torch.Tensor) -> torch.Tensor:
        """Encode recent interaction sequence into current user state."""
        if event_tokens.dim() == 1:
            event_tokens = event_tokens.unsqueeze(0)
        event_tokens = event_tokens.long()
        embedded = self.event_embedding(event_tokens)
        _, h_n = self.user_encoder(embedded)
        return self.user_mlp(h_n.squeeze(0))

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item/context feature IDs into an item vector."""
        if item_features.dim() == 1:
            item_features = item_features.unsqueeze(0)
        item_features = item_features.long()
        if item_features.size(1) != NUM_ITEM_FEATURES:
            raise ValueError(
                f"item_features must have {NUM_ITEM_FEATURES} columns, "
                f"got shape {tuple(item_features.shape)}"
            )

        item_id_emb = self.item_id_embedding(item_features[:, ITEM_COL_ITEM_ID])
        topic_emb = self.topic_embedding(item_features[:, ITEM_COL_TOPIC_ID])
        vibe_emb = self.vibe_embedding(item_features[:, ITEM_COL_VIBE_ID])
        cluster_emb = self.topic_cluster_embedding(item_features[:, ITEM_COL_TOPIC_CLUSTER_ID])
        entity_emb = self.entity_embedding(item_features[:, ITEM_COL_ENTITY_ID])
        event_emb = self.event_embedding_item(item_features[:, ITEM_COL_EVENT_ID])
        recency_emb = self.recency_embedding(item_features[:, ITEM_COL_RECENCY_BUCKET])
        pop_emb = self.popularity_embedding(item_features[:, ITEM_COL_POPULARITY_BUCKET])

        item_input = torch.cat(
            [
                item_id_emb,
                topic_emb,
                vibe_emb,
                cluster_emb,
                entity_emb,
                event_emb,
                recency_emb,
                pop_emb,
            ],
            dim=1,
        )
        return self.item_mlp(item_input)

    def forward(
        self, event_tokens: torch.Tensor, item_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Return raw head outputs for each candidate row.

        event_tokens: [batch, seq_len]
        item_features: [batch, NUM_ITEM_FEATURES]
        """
        user_state = self.encode_user_state(event_tokens)
        item_vec = self.encode_item(item_features)
        interaction = torch.cat(
            [
                user_state,
                item_vec,
                user_state * item_vec,
                torch.abs(user_state - item_vec),
            ],
            dim=1,
        )
        hidden = self.interaction_tower(interaction)
        return {
            "finish_logit": self.finish_head(hidden),
            "like_logit": self.like_head(hidden),
            "share_logit": self.share_head(hidden),
            "rewatch_logit": self.rewatch_head(hidden),
            "fast_skip_logit": self.fast_skip_head(hidden),
            "watch_time_raw": self.watch_time_head(hidden),
        }

    def score_from_heads(
        self,
        heads: dict[str, torch.Tensor],
        weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Compute scalar ranking score from multi-head predictions."""
        if weights is None:
            weights = {
                "finish": 1.0,
                "like": 0.8,
                "share": 1.2,
                "rewatch": 1.3,
                "fast_skip": 1.5,
                "watch_time": 1.0,
            }

        p_finish = torch.sigmoid(heads["finish_logit"])
        p_like = torch.sigmoid(heads["like_logit"])
        p_share = torch.sigmoid(heads["share_logit"])
        p_rewatch = torch.sigmoid(heads["rewatch_logit"])
        p_fast_skip = torch.sigmoid(heads["fast_skip_logit"])
        expected_watch = torch.sigmoid(heads["watch_time_raw"])

        score = (
            weights["finish"] * p_finish
            + weights["like"] * p_like
            + weights["share"] * p_share
            + weights["rewatch"] * p_rewatch
            - weights["fast_skip"] * p_fast_skip
            + weights["watch_time"] * expected_watch
        )
        return score

    def get_dot_product(
        self,
        event_tokens: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Pure retrieval similarity score from the 64d user/item vectors."""
        user_vec = self.encode_user_state(event_tokens)
        item_vec = self.encode_item(item_features)
        return (user_vec * item_vec).sum(dim=-1)

    def predict_score(
        self,
        event_tokens: torch.Tensor,
        item_features: torch.Tensor,
        weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        heads = self.forward(event_tokens=event_tokens, item_features=item_features)
        return self.score_from_heads(heads=heads, weights=weights)


class AddictionEngine(GenerativeTwoTower):
    """Compatibility alias for the same two-tower architecture."""
