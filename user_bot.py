"""Persona-based synthetic users for recommendation training."""

from __future__ import annotations

import random
from collections import defaultdict
from collections import deque
from typing import Iterable


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(k in text for k in keywords)


class _BasePersonaBot:
    def __init__(self, name: str, seed: int = 42) -> None:
        self.name = name
        self.rng = random.Random(seed)
        self.topic_exposure = defaultdict(int)
        self.last_topic = None

    def _build_item(self, topic: str, vibe: str, headline: str) -> dict[str, str]:
        return {
            "Topic_Name": topic,
            "Vibe_Style": vibe,
            "Headline": headline,
        }

    def _packet_from_reward(self, reward: float, vibe: str) -> dict[str, float | bool]:
        vibe_l = vibe.lower()
        reflective = _contains_any(vibe_l, ("analytical", "somber", "reflective"))

        if reward >= 9.0:
            listen_ms = float(self.rng.randint(8200, 10000))
            if reflective:
                listen_ms = float(self.rng.randint(8800, 10000))
            interest = 0.92
            liked = self.rng.random() < 0.78
            shared = self.rng.random() < 0.15
            rewound = self.rng.random() < 0.22
        elif reward > 0.0:
            low = 2600
            high = 6200
            if reflective:
                low = 5200
                high = 9000
            listen_ms = float(self.rng.randint(low, high))
            interest = 0.52
            liked = self.rng.random() < 0.32
            shared = self.rng.random() < 0.04
            rewound = self.rng.random() < 0.09
        else:
            listen_ms = float(self.rng.randint(550, 1900))
            interest = 0.08
            liked = False
            shared = False
            rewound = False

        fast_skip = listen_ms < 2000
        finished = listen_ms >= 9000
        commented = bool(reward >= 9.0 and self.rng.random() < 0.10)

        return {
            "listen_ms": listen_ms,
            "total_ms": 10000.0,
            "shared": shared,
            "rewinded": rewound,
            "liked": liked,
            "commented": commented,
            "fast_skip": fast_skip,
            "finished": finished,
            "interest": interest,
            "quit_session": False,
        }

    def get_reward(self, item: dict[str, str]) -> float:
        raise NotImplementedError

    def simulate_session(
        self,
        topic_id: int,
        vibe_id: int,
        topic_map: dict[int, str],
        vibe_map: dict[int, str],
        headline: str | None = None,
        cluster_id: int | None = None,
        session_index: int = 0,
        update_state: bool = True,
    ) -> dict[str, float | bool]:
        del cluster_id, session_index
        topic = str(topic_map.get(topic_id, "unknown"))
        vibe = str(vibe_map.get(vibe_id, "Standard Reporting"))
        head = str(headline or "")
        item = self._build_item(topic=topic, vibe=vibe, headline=head)

        reward = float(self.get_reward(item))
        packet = self._packet_from_reward(reward=reward, vibe=vibe)

        if update_state:
            topic_l = topic.lower()
            self.topic_exposure[topic_l] += 1
            self.last_topic = topic_l

        return packet


class DaveBot(_BasePersonaBot):
    # "The Tech Bro" - Loves Hardware, Hates Policy
    def __init__(self, name: str = "DaveBot", seed: int = 42) -> None:
        super().__init__(name=name, seed=seed)
        self.interest_weights = {
            "hype beast": 1.5,
            "analytical deep dive": 1.2,
            "sarcastic takedown": 0.8,
            "standard reporting": 1.0,
            "dystopian warning": 0.5,
            "somber reflection": 0.1,
            "angry rant": 1.1,
            "clickbait shocker": 1.3,
        }
        self.boost_keywords = (
            "apple",
            "samsung",
            "google",
            "pixel",
            "iphone",
            "macbook",
            "chip",
            "nvidia",
            "meta",
            "review",
            "hands-on",
        )
        self.penalty_keywords = (
            "policy",
            "senate",
            "climate",
            "market",
            "stock",
            "eu regulation",
            "lawsuit",
        )

    def get_reward(self, item: dict[str, str]) -> float:
        vibe = str(item.get("Vibe_Style", "Standard Reporting")).lower()
        score = self.interest_weights.get(vibe, 1.0)
        text = f"{item.get('Headline', '')} {item.get('Topic_Name', '')}".lower()

        # Keyword override: if it's cool hardware, Dave likes it regardless of vibe.
        if _contains_any(text, self.boost_keywords):
            score = max(score, 1.0) * 1.5

        if _contains_any(text, self.penalty_keywords):
            score *= 0.2

        return 10.0 if score > 1.2 else (3.0 if score > 0.7 else -5.0)


class SarahBot(_BasePersonaBot):
    # "The Mindful Creative" - Loves Depth, Hates Hype
    def __init__(self, name: str = "SarahBot", seed: int = 43) -> None:
        super().__init__(name=name, seed=seed)
        self.interest_weights = {
            "somber reflection": 1.5,
            "analytical deep dive": 1.4,
            "dystopian warning": 1.3,
            "standard reporting": 1.0,
            "hype beast": 0.1,
            "angry rant": 0.1,
            "sarcastic takedown": 0.2,
            "clickbait shocker": 0.0,
        }
        self.boost_keywords = (
            "climate",
            "science",
            "space",
            "nasa",
            "study",
            "research",
            "design",
            "art",
            "history",
            "culture",
            "health",
        )
        self.penalty_keywords = (
            "crypto",
            "nft",
            "stock",
            "musk",
            "gaming",
            "console",
            "price",
            "deal",
        )
        self.recent_vibes: deque[str] = deque(maxlen=5)
        self.session_ended = False

    @staticmethod
    def _is_noise(vibe: str) -> bool:
        vibe = vibe.lower()
        return _contains_any(vibe, ("angry", "rant", "sarcastic", "clickbait"))

    def get_reward(self, item: dict[str, str]) -> float:
        vibe = str(item.get("Vibe_Style", "Standard Reporting")).lower()
        score = self.interest_weights.get(vibe, 1.0)
        text = f"{item.get('Headline', '')} {item.get('Topic_Name', '')}".lower()

        if _contains_any(text, self.boost_keywords):
            score *= 1.5
        if _contains_any(text, self.penalty_keywords):
            score *= 0.0

        return 10.0 if score > 1.2 else (3.0 if score > 0.8 else -5.0)

    def simulate_session(
        self,
        topic_id: int,
        vibe_id: int,
        topic_map: dict[int, str],
        vibe_map: dict[int, str],
        headline: str | None = None,
        cluster_id: int | None = None,
        session_index: int = 0,
        update_state: bool = True,
    ) -> dict[str, float | bool]:
        del cluster_id, session_index
        if self.session_ended:
            return {
                "listen_ms": 200.0,
                "total_ms": 10000.0,
                "shared": False,
                "rewinded": False,
                "liked": False,
                "commented": False,
                "fast_skip": True,
                "finished": False,
                "interest": 0.0,
                "quit_session": True,
            }

        topic = str(topic_map.get(topic_id, "unknown"))
        vibe = str(vibe_map.get(vibe_id, "Standard Reporting"))
        head = str(headline or "")
        item = self._build_item(topic=topic, vibe=vibe, headline=head)
        reward = float(self.get_reward(item))
        packet = self._packet_from_reward(reward=reward, vibe=vibe)

        if update_state:
            topic_l = topic.lower()
            self.topic_exposure[topic_l] += 1
            self.last_topic = topic_l
            self.recent_vibes.append(vibe)

            noisy = sum(1 for v in self.recent_vibes if self._is_noise(v))
            if len(self.recent_vibes) == 5 and noisy > 2 and self.rng.random() < 0.55:
                packet["quit_session"] = True
                self.session_ended = True

        return packet


class GamerBot(_BasePersonaBot):
    # "The Escapist" - Loves Games, Hates Boredom (unless it's gaming news)
    def __init__(self, name: str = "GamerBot", seed: int = 44) -> None:
        super().__init__(name=name, seed=seed)
        self.interest_weights = {
            "hype beast": 1.5,
            "sarcastic takedown": 1.4,
            "angry rant": 1.2,
            "clickbait shocker": 1.3,
            "standard reporting": 0.8,
            "analytical deep dive": 0.4,
            "somber reflection": 0.1,
        }
        self.gaming_signals = (
            "ign",
            "kotaku",
            "polygon",
            "eurogamer",
            "pcgamer",
            "destructoid",
            "nintendo",
            "game",
            "console",
            "steam",
            "switch",
            "ps5",
            "xbox",
            "movie",
            "trailer",
            "review",
            "playstation",
        )
        self.penalty_keywords = (
            "stock",
            "market",
            "policy",
            "lawsuit",
            "congress",
            "climate",
            "layoff",
        )

    def get_reward(self, item: dict[str, str]) -> float:
        vibe = str(item.get("Vibe_Style", "Standard Reporting")).lower()
        score = self.interest_weights.get(vibe, 1.0)
        text = f"{item.get('Headline', '')} {item.get('Topic_Name', '')}".lower()

        # If it is gaming-related, ignore "boring vibe" penalties.
        is_gaming = _contains_any(text, self.gaming_signals)
        if is_gaming:
            score = max(score, 1.0) * 1.8

        if _contains_any(text, self.penalty_keywords) and not is_gaming:
            score *= 0.0

        return 10.0 if score > 1.2 else (3.0 if score > 0.8 else -5.0)


class InvestorBot(_BasePersonaBot):
    # "The Degenerate" - Loves Money, Hates Fluff
    def __init__(self, name: str = "InvestorBot", seed: int = 45) -> None:
        super().__init__(name=name, seed=seed)
        self.interest_weights = {
            "analytical deep dive": 1.4,
            "hype beast": 1.3,
            "dystopian warning": 1.2,
            "clickbait shocker": 1.1,
            "sarcastic takedown": 0.5,
            "somber reflection": 0.2,
            "standard reporting": 1.0,
            "angry rant": 0.5,
        }
        self.boost_keywords = (
            "stock",
            "market",
            "crypto",
            "bitcoin",
            "revenue",
            "profit",
            "ipo",
            "billion",
            "merger",
            "fed",
            "rates",
            "bank",
        )
        self.penalty_keywords = (
            "game",
            "movie",
            "console",
            "art",
            "design",
            "culture",
        )

    def get_reward(self, item: dict[str, str]) -> float:
        vibe = str(item.get("Vibe_Style", "Standard Reporting")).lower()
        score = self.interest_weights.get(vibe, 1.0)
        text = f"{item.get('Headline', '')} {item.get('Topic_Name', '')}".lower()

        if _contains_any(text, self.boost_keywords):
            score *= 1.6
        if _contains_any(text, self.penalty_keywords):
            score *= 0.1

        return 10.0 if score > 1.3 else (2.0 if score > 0.9 else -5.0)


class LowMoodAuditBot(_BasePersonaBot):
    """
    Safety-audit persona for low-mood sessions.

    This bot is intentionally conservative: it can prefer reflective coverage,
    but it penalizes repetitive doom-heavy streaks to expose spiral risk.
    """

    def __init__(self, name: str = "LowMoodAuditBot", seed: int = 46) -> None:
        super().__init__(name=name, seed=seed)
        self.interest_weights = {
            "somber reflection": 1.4,
            "analytical deep dive": 1.25,
            "dystopian warning": 1.0,
            "standard reporting": 0.9,
            "hype beast": 0.2,
            "sarcastic takedown": 0.1,
            "clickbait shocker": 0.0,
            "angry rant": 0.0,
        }
        self.recovery_keywords = (
            "recovery",
            "support",
            "treatment",
            "therapy",
            "healing",
            "community",
            "solutions",
            "help",
        )
        self.bleak_keywords = (
            "war",
            "death",
            "crisis",
            "attack",
            "disaster",
            "suicide",
            "depression",
            "violence",
            "layoff",
        )
        self.recent_bleak: deque[int] = deque(maxlen=8)
        self.session_ended = False

    def _is_bleak(self, text: str) -> bool:
        return _contains_any(text, self.bleak_keywords)

    def get_reward(self, item: dict[str, str]) -> float:
        vibe = str(item.get("Vibe_Style", "Standard Reporting")).lower()
        score = self.interest_weights.get(vibe, 1.0)
        text = f"{item.get('Headline', '')} {item.get('Topic_Name', '')}".lower()

        is_bleak = self._is_bleak(text)
        has_recovery = _contains_any(text, self.recovery_keywords)

        if has_recovery:
            score *= 1.35
        if is_bleak:
            # Mild attraction to serious content, but not doom spirals.
            score *= 1.05
            if sum(self.recent_bleak) >= 3:
                score *= 0.2

        return 10.0 if score > 1.2 else (3.0 if score > 0.8 else -5.0)

    def simulate_session(
        self,
        topic_id: int,
        vibe_id: int,
        topic_map: dict[int, str],
        vibe_map: dict[int, str],
        headline: str | None = None,
        cluster_id: int | None = None,
        session_index: int = 0,
        update_state: bool = True,
    ) -> dict[str, float | bool]:
        del cluster_id, session_index
        if self.session_ended:
            return {
                "listen_ms": 250.0,
                "total_ms": 10000.0,
                "shared": False,
                "rewinded": False,
                "liked": False,
                "commented": False,
                "fast_skip": True,
                "finished": False,
                "interest": 0.0,
                "quit_session": True,
            }

        topic = str(topic_map.get(topic_id, "unknown"))
        vibe = str(vibe_map.get(vibe_id, "Standard Reporting"))
        head = str(headline or "")
        item = self._build_item(topic=topic, vibe=vibe, headline=head)
        reward = float(self.get_reward(item))
        packet = self._packet_from_reward(reward=reward, vibe=vibe)

        if update_state:
            topic_l = topic.lower()
            self.topic_exposure[topic_l] += 1
            self.last_topic = topic_l
            is_bleak = int(self._is_bleak(f"{head} {topic}"))
            self.recent_bleak.append(is_bleak)

            # If the feed stays bleak for too long, simulate emotional drop-off.
            if len(self.recent_bleak) == self.recent_bleak.maxlen and sum(self.recent_bleak) >= 6:
                if self.rng.random() < 0.65:
                    packet["quit_session"] = True
                    self.session_ended = True

        return packet


class AIDoomBot(_BasePersonaBot):
    """
    Fear-driven AI persona.

    This user is anxious about AI risks and compulsively consumes alarming AI
    coverage, while mostly ignoring unrelated content.
    """

    def __init__(self, name: str = "AIDoomBot", seed: int = 47) -> None:
        super().__init__(name=name, seed=seed)
        self.interest_weights = {
            "dystopian warning": 1.7,
            "analytical deep dive": 1.35,
            "somber reflection": 1.25,
            "clickbait shocker": 1.15,
            "standard reporting": 0.8,
            "sarcastic takedown": 0.4,
            "hype beast": 0.2,
            "angry rant": 0.3,
        }
        self.ai_fear_keywords = (
            "ai",
            "artificial intelligence",
            "llm",
            "agent",
            "autonomous",
            "alignment",
            "misalignment",
            "deepfake",
            "surveillance",
            "job loss",
            "layoff",
            "disinformation",
            "bias",
            "safety",
            "existential",
            "weaponized",
            "cyberattack",
            "privacy",
            "regulation",
        )
        self.comfort_keywords = (
            "deal",
            "coupon",
            "fashion",
            "sports",
            "celebrity",
            "travel",
            "recipe",
            "shopping",
        )

    def get_reward(self, item: dict[str, str]) -> float:
        vibe = str(item.get("Vibe_Style", "Standard Reporting")).lower()
        score = self.interest_weights.get(vibe, 1.0)
        text = f"{item.get('Headline', '')} {item.get('Topic_Name', '')}".lower()

        fear_hits = sum(1 for k in self.ai_fear_keywords if k in text)
        if fear_hits > 0:
            score *= 1.45 + min(0.35, 0.07 * fear_hits)
        else:
            score *= 0.35

        if _contains_any(text, self.comfort_keywords):
            score *= 0.2

        return 10.0 if score > 1.2 else (3.0 if score > 0.8 else -5.0)


class ContrarianChaosBot(_BasePersonaBot):
    """
    Exploration/exploitation stress-test persona.

    Phase 1 (steps 1-100): Tech-hype fixation.
    Phase 2 (step 101+): Hard pivot into anti-tech + niche rabbit-hole interests.
    """

    def __init__(self, name: str = "ContrarianChaosBot", seed: int = 48) -> None:
        super().__init__(name=name, seed=seed)
        self.step_count = 0
        self.phase1_weights = {
            "hype beast": 1.5,
            "analytical deep dive": 1.3,
            "clickbait shocker": 1.2,
            "standard reporting": 1.0,
            "sarcastic takedown": 0.7,
            "dystopian warning": 0.5,
            "somber reflection": 0.2,
            "angry rant": 0.8,
        }
        self.phase2_weights = {
            "analytical deep dive": 1.45,
            "dystopian warning": 1.35,
            "somber reflection": 1.25,
            "clickbait shocker": 1.1,
            "standard reporting": 0.75,
            "sarcastic takedown": 0.55,
            "hype beast": 0.2,
            "angry rant": 0.3,
        }
        self.phase1_hook_keywords = (
            "apple",
            "iphone",
            "mac",
            "nvidia",
            "chip",
            "gpu",
            "biohack",
            "longevity",
            "ai",
            "agent",
            "openai",
            "startup",
        )
        self.phase2_rabbit_keywords = (
            "solarpunk",
            "piratecore",
            "corporate goth",
            "digital minimalism",
            "silent walk",
            "lidar",
            "archaeology",
            "forgotten infrastructure",
            "ocean cable",
            "satellite interference",
            "systemic mystery",
            "micro-drama",
            "subculture",
            "gatekeeping",
            "whistleblower",
        )
        self.phase2_bridge_keywords = (
            "social credit",
            "constitutional crisis",
            "surveillance",
            "antitrust",
            "deepfake",
            "tariff",
            "strike",
            "labor",
            "policy thriller",
            "forgotten",
            "mystery",
            "underground",
        )
        self.phase1_hook_hits = 0

    def _in_phase1(self) -> bool:
        return self.step_count <= 100

    def get_reward(self, item: dict[str, str]) -> float:
        vibe = str(item.get("Vibe_Style", "Standard Reporting")).lower()
        text = f"{item.get('Headline', '')} {item.get('Topic_Name', '')}".lower()

        if self._in_phase1():
            score = self.phase1_weights.get(vibe, 1.0)
            has_hook = _contains_any(text, self.phase1_hook_keywords)
            if has_hook:
                self.phase1_hook_hits += 1
                # Saturation penalty: the same hook keeps getting less effective.
                saturation = max(0.55, 1.0 - 0.01 * self.phase1_hook_hits)
                score *= 1.8 * saturation
            if _contains_any(text, self.phase2_rabbit_keywords):
                score *= 0.35
        else:
            score = self.phase2_weights.get(vibe, 1.0)
            # Disillusionment phase: old obsessions are damped, not hard-locked.
            # Hard-locking to -5.0 can collapse learning if the dataset is sparse.
            if _contains_any(text, self.phase1_hook_keywords):
                score *= 0.35
            if _contains_any(text, self.phase2_rabbit_keywords):
                score *= 1.9
            else:
                # Keep a neutral floor so exploration can recover.
                score *= 0.75
            if _contains_any(text, self.phase2_bridge_keywords):
                score *= 1.25

        if score > 1.25:
            return 10.0
        if score > 0.95:
            return 7.5
        if score > 0.60:
            return 1.0
        return -4.5

    def simulate_session(
        self,
        topic_id: int,
        vibe_id: int,
        topic_map: dict[int, str],
        vibe_map: dict[int, str],
        headline: str | None = None,
        cluster_id: int | None = None,
        session_index: int = 0,
        update_state: bool = True,
    ) -> dict[str, float | bool]:
        del cluster_id, session_index
        if update_state:
            self.step_count += 1

        topic = str(topic_map.get(topic_id, "unknown"))
        vibe = str(vibe_map.get(vibe_id, "Standard Reporting"))
        head = str(headline or "")
        item = self._build_item(topic=topic, vibe=vibe, headline=head)
        reward = float(self.get_reward(item))
        packet = self._packet_from_reward(reward=reward, vibe=vibe)

        if update_state:
            topic_l = topic.lower()
            self.topic_exposure[topic_l] += 1
            self.last_topic = topic_l

        return packet


class UserBot(DaveBot):
    """Backward-compatible Dave wrapper used by existing scripts."""

    def __init__(
        self,
        name: str = "Complex Dave V2",
        seed: int = 42,
        liked_topics: Iterable[int] | None = None,
        liked_vibes: Iterable[int] | None = None,
        hated_topics: Iterable[int] | None = None,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.liked_topics = set(liked_topics or [])
        self.liked_vibes = set(liked_vibes or [])
        self.hated_topics = set(hated_topics or [])

    def interact(self, topic_id: int, vibe_id: int) -> float:
        """Compatibility API for train_simulation.py (returns [0, 1])."""
        if self.liked_topics or self.liked_vibes or self.hated_topics:
            score = 0.45
            if topic_id in self.hated_topics:
                score -= 0.55
            if topic_id in self.liked_topics:
                score += 0.35
            if vibe_id in self.liked_vibes:
                score += 0.25
            return max(0.0, min(1.0, score))
        return self.rng.random()
