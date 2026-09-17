"""Gymnasium environment for one learner against a scripted opponent.

One environment decision represents 0.25 seconds of game time: the learner's
requested action is applied, then the shared match runs five deterministic
0.05-second physics updates.  Red's scripted controller is updated by those
same match ticks, so it acts through the normal controller schedule.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arena_viewer import FIXED_TIMESTEP_SECONDS, Match
from controllers import HumanController, RandomController, ScriptedController
from training_interface import ACTION_COUNT, CARD_IDS, observation

DECISION_SECONDS = 0.25
SIMULATION_STEPS_PER_DECISION = round(
    DECISION_SECONDS / FIXED_TIMESTEP_SECONDS,
)

# Fixed capacities let common vectorized trainers batch observations. Presence
# masks distinguish real records from zero padding. Counts above a capacity are
# reported in info rather than silently pretending no truncation occurred.
MAX_ENTITIES = 256
MAX_PROJECTILES = 256
MAX_PENDING_SPELLS = 64

TOWER_FEATURES = 8
ENTITY_FEATURES = 13
PROJECTILE_FEATURES = 7
SPELL_FEATURES = 7
OBSERVATION_LIMIT = 1_000_000.0

# Keep the first objective deliberately sparse. These named values make the
# learning contract visible to experiment code and difficult to change by
# accident while adjusting unrelated environment behavior.
WIN_REWARD = 1.0
LOSS_REWARD = -1.0
DRAW_REWARD = 0.0
NON_TERMINAL_REWARD = 0.0


def _padded_records(
    records: tuple[tuple[float, ...], ...],
    capacity: int,
    features: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    array = np.zeros((capacity, features), dtype=np.float32)
    mask = np.zeros(capacity, dtype=np.int8)
    kept = min(len(records), capacity)
    if kept:
        array[:kept] = np.asarray(records[:kept], dtype=np.float32)
        mask[:kept] = 1
    return array, mask, max(0, len(records) - capacity)


def encode_fixed_observation(
    match: Match,
    team: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Convert public match state to the fixed arrays used by the model."""
    raw = observation(match, team)
    entities, entity_mask, entity_overflow = _padded_records(
        raw["entities"],
        MAX_ENTITIES,
        ENTITY_FEATURES,
    )
    projectiles, projectile_mask, projectile_overflow = _padded_records(
        raw["projectiles"],
        MAX_PROJECTILES,
        PROJECTILE_FEATURES,
    )
    spells, spell_mask, spell_overflow = _padded_records(
        raw["pending_spells"],
        MAX_PENDING_SPELLS,
        SPELL_FEATURES,
    )
    encoded = {
        "hand_card_ids": np.asarray(raw["hand_card_ids"], dtype=np.int64),
        "hand_costs": np.asarray(raw["hand_costs"], dtype=np.float32),
        "elixir": np.asarray([raw["elixir"]], dtype=np.float32),
        "tower_state": np.asarray(raw["tower_state"], dtype=np.float32),
        "crown_scores": np.asarray(raw["crown_scores"], dtype=np.int8),
        "match_elapsed": np.asarray([raw["match_elapsed"]], dtype=np.float32),
        "phase_time_remaining": np.asarray(
            [raw["phase_time_remaining"]],
            dtype=np.float32,
        ),
        "match_phase": np.int64(raw["match_phase"]),
        "entities": entities,
        "entity_mask": entity_mask,
        "projectiles": projectiles,
        "projectile_mask": projectile_mask,
        "pending_spells": spells,
        "pending_spell_mask": spell_mask,
    }
    overflow = {
        "entities": entity_overflow,
        "projectiles": projectile_overflow,
        "pending_spells": spell_overflow,
    }
    return encoded, overflow


class ClashRoyaleEnv(gym.Env[dict[str, np.ndarray], int]):
    """Single-agent environment with one learner and one baseline opponent.

    Rewards are intentionally sparse in this first wrapper: +1 for a blue win,
    -1 for a red win, and 0 for a draw or any non-terminal decision. Reward
    shaping can be added later without changing observations or action IDs.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [],
        "render_fps": 4,
    }

    def __init__(
        self,
        *,
        learner_team: str = "blue",
        opponent: str = "scripted",
        max_decisions: int = 1_600,
    ) -> None:
        super().__init__()
        if learner_team not in {"blue", "red"}:
            raise ValueError("learner_team must be 'blue' or 'red'")
        if opponent not in {"scripted", "random"}:
            raise ValueError("opponent must be 'scripted' or 'random'")
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")

        self.learner_team = learner_team
        self.opponent_team = "red" if learner_team == "blue" else "blue"
        self.opponent_name = opponent
        self.max_decisions = max_decisions
        self.action_space = spaces.Discrete(ACTION_COUNT)
        record_matrix = lambda rows, columns: spaces.Box(
            low=-OBSERVATION_LIMIT,
            high=OBSERVATION_LIMIT,
            shape=(rows, columns),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "hand_card_ids": spaces.MultiDiscrete(
                    np.full(4, len(CARD_IDS) + 1, dtype=np.int64),
                ),
                "hand_costs": spaces.Box(
                    low=0,
                    high=10,
                    shape=(4,),
                    dtype=np.float32,
                ),
                "elixir": spaces.Box(
                    low=0,
                    high=10,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "tower_state": record_matrix(6, TOWER_FEATURES),
                "crown_scores": spaces.Box(
                    low=0,
                    high=3,
                    shape=(2,),
                    dtype=np.int8,
                ),
                "match_elapsed": spaces.Box(
                    low=0,
                    high=OBSERVATION_LIMIT,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "phase_time_remaining": spaces.Box(
                    low=0,
                    high=OBSERVATION_LIMIT,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "match_phase": spaces.Discrete(4),
                "entities": record_matrix(MAX_ENTITIES, ENTITY_FEATURES),
                "entity_mask": spaces.MultiBinary(MAX_ENTITIES),
                "projectiles": record_matrix(
                    MAX_PROJECTILES,
                    PROJECTILE_FEATURES,
                ),
                "projectile_mask": spaces.MultiBinary(MAX_PROJECTILES),
                "pending_spells": record_matrix(
                    MAX_PENDING_SPELLS,
                    SPELL_FEATURES,
                ),
                "pending_spell_mask": spaces.MultiBinary(MAX_PENDING_SPELLS),
            }
        )
        self.match: Match | None = None
        self.decision_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Start a fresh deterministic match and return blue's observation."""
        super().reset(seed=seed)
        del options  # Reserved for future deck/opponent configuration.
        episode_seed = 0 if seed is None else seed
        opponent_controller = (
            ScriptedController()
            if self.opponent_name == "scripted"
            else RandomController(seed=episode_seed)
        )
        controllers = {
            self.learner_team: HumanController(),
            self.opponent_team: opponent_controller,
        }
        self.match = Match(
            blue_controller=controllers["blue"],
            red_controller=controllers["red"],
        )
        self.decision_count = 0
        observation, overflow = self._observation()
        return observation, self._info(
            action_accepted=None,
            overflow=overflow,
        )

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply blue's decision, advance 0.25 seconds, and report the result."""
        match = self._require_match()
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in 0..{ACTION_COUNT - 1}")
        if match.done:
            raise RuntimeError("step() called after termination; call reset()")

        action_id = int(action)
        legal_mask = match.action_mask(self.learner_team)
        action_accepted = legal_mask[action_id] and match.apply_discrete_action(
            self.learner_team,
            action_id,
        )

        completed_steps = 0
        for _ in range(SIMULATION_STEPS_PER_DECISION):
            if match.done:
                break
            match.update_simulation()
            completed_steps += 1

        self.decision_count += 1
        terminated = match.done
        truncated = not terminated and self.decision_count >= self.max_decisions
        reward = self._terminal_reward() if terminated else NON_TERMINAL_REWARD
        observation, overflow = self._observation()
        info = self._info(
            action_accepted=action_accepted,
            overflow=overflow,
        )
        info["simulation_steps"] = completed_steps
        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the mask format expected by sb3-contrib MaskablePPO."""
        match = self._require_match()
        return np.asarray(match.action_mask(self.learner_team), dtype=np.bool_)

    def close(self) -> None:
        """Release the episode reference; no display resources exist."""
        self.match = None

    def _require_match(self) -> Match:
        if self.match is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self.match

    def _terminal_reward(self) -> float:
        winner = self._require_match().winner
        if winner == self.learner_team:
            return WIN_REWARD
        if winner == self.opponent_team:
            return LOSS_REWARD
        return DRAW_REWARD

    def _observation(self) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        return encode_fixed_observation(
            self._require_match(),
            self.learner_team,
        )

    def _info(
        self,
        *,
        action_accepted: bool | None,
        overflow: dict[str, int],
    ) -> dict[str, Any]:
        match = self._require_match()
        return {
            "action_mask": self.action_masks(),
            "action_accepted": action_accepted,
            "winner": match.winner,
            "match_elapsed": match.match_elapsed,
            "decision_count": self.decision_count,
            "observation_overflow": overflow,
        }


__all__ = [
    "DECISION_SECONDS",
    "DRAW_REWARD",
    "LOSS_REWARD",
    "MAX_ENTITIES",
    "MAX_PENDING_SPELLS",
    "MAX_PROJECTILES",
    "NON_TERMINAL_REWARD",
    "SIMULATION_STEPS_PER_DECISION",
    "WIN_REWARD",
    "ClashRoyaleEnv",
    "encode_fixed_observation",
]
