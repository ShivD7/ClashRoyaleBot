"""Interchangeable decision makers for human, scripted, and learned players.

Controllers may request actions, but they never mutate the match. The match
remains responsible for validating placement, spending Elixir, cycling cards,
and spawning entities, which prevents every controller type from cheating.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Callable, Sequence


@dataclass(frozen=True)
class PlayCardAction:
    """Request that one hand slot be deployed on one arena tile."""

    hand_slot: int
    tile: tuple[int, int]


@dataclass(frozen=True)
class ControllerCard:
    """Card information a controller is allowed to inspect."""

    name: str
    elixir_cost: int
    role: str
    card_type: str
    target_priority: str
    target_types: str
    movement_type: str
    attack_style: str


@dataclass(frozen=True)
class ControllerContext:
    """Read-only decision snapshot supplied by the authoritative match."""

    team: str
    match_elapsed: float
    elixir: float
    hand: tuple[ControllerCard, ...]
    legal_actions: tuple[PlayCardAction, ...]
    crown_scores: dict[str, int]


class PlayerController(ABC):
    """Base class implemented by every source of player decisions."""

    name = "base"

    @abstractmethod
    def choose_action(
        self,
        context: ControllerContext,
    ) -> PlayCardAction | None:
        """Return one requested action or ``None`` to wait."""

    def reset(self) -> None:
        """Reset controller episode memory before a rematch."""


class HumanController(PlayerController):
    """Marker controller whose actions arrive through mouse/keyboard input."""

    name = "human"

    def choose_action(
        self,
        context: ControllerContext,
    ) -> PlayCardAction | None:
        return None


class RandomController(PlayerController):
    """Random legal-action baseline useful for smoke tests and evaluation."""

    name = "random"

    def __init__(self, seed: int = 0, play_probability: float = 0.35) -> None:
        self.seed = seed
        self.play_probability = play_probability
        self._random = random.Random(seed)

    def choose_action(
        self,
        context: ControllerContext,
    ) -> PlayCardAction | None:
        if not context.legal_actions:
            return None
        if self._random.random() > self.play_probability:
            return None
        return self._random.choice(context.legal_actions)

    def reset(self) -> None:
        self._random.seed(self.seed)


class ScriptedController(PlayerController):
    """Simple lane-push opponent with deterministic card preferences."""

    name = "scripted"

    def __init__(self) -> None:
        self.preferred_lane = 0

    def choose_action(
        self,
        context: ControllerContext,
    ) -> PlayCardAction | None:
        if not context.legal_actions or context.elixir < 3:
            return None

        priority = {
            "win_condition": 0,
            "ranged_support": 1,
            "splash_support": 1,
            "flying_swarm": 2,
            "ground_swarm": 2,
            "tank_killer": 2,
            "defensive_building": 3,
            "mini_tank": 3,
            "cycle_swarm": 4,
            "big_spell": 5,
            "small_spell": 6,
        }
        affordable_slots = {
            action.hand_slot
            for action in context.legal_actions
        }
        slot = min(
            affordable_slots,
            key=lambda index: (
                priority.get(context.hand[index].role, 99),
                index,
            ),
        )
        candidates = [
            action
            for action in context.legal_actions
            if action.hand_slot == slot
        ]
        desired_column = 4 if self.preferred_lane == 0 else 13
        desired_row = 15 if context.team == "blue" else 14
        action = min(
            candidates,
            key=lambda candidate: (
                abs(candidate.tile[0] - desired_column)
                + abs(candidate.tile[1] - desired_row),
                candidate.tile,
            ),
        )
        self.preferred_lane = 1 - self.preferred_lane
        return action

    def reset(self) -> None:
        self.preferred_lane = 0


@dataclass(frozen=True)
class ScheduledPlay:
    """One desired card placement in a fixed curriculum sequence."""

    at_seconds: float
    card_name: str
    tile: tuple[int, int]


class FixedSequenceController(PlayerController):
    """Play named cards at predefined times whenever each action is legal."""

    name = "fixed"

    def __init__(self, sequence: Sequence[ScheduledPlay] = ()) -> None:
        self.sequence = tuple(sequence)
        self.next_index = 0

    def choose_action(
        self,
        context: ControllerContext,
    ) -> PlayCardAction | None:
        if self.next_index >= len(self.sequence):
            return None
        scheduled = self.sequence[self.next_index]
        if context.match_elapsed < scheduled.at_seconds:
            return None

        slot = next(
            (
                index
                for index, card in enumerate(context.hand)
                if card.name == scheduled.card_name
            ),
            None,
        )
        if slot is None:
            return None
        requested = PlayCardAction(slot, scheduled.tile)
        if requested not in context.legal_actions:
            return None

        self.next_index += 1
        return requested

    def reset(self) -> None:
        self.next_index = 0


Policy = Callable[[ControllerContext], PlayCardAction | None]


class RLController(PlayerController):
    """Adapter for a future learned policy with the same action contract."""

    name = "rl"

    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy

    def choose_action(
        self,
        context: ControllerContext,
    ) -> PlayCardAction | None:
        if self.policy is None:
            return None
        return self.policy(context)


CONTROLLER_TYPES = {
    HumanController.name: HumanController,
    RandomController.name: RandomController,
    ScriptedController.name: ScriptedController,
    FixedSequenceController.name: FixedSequenceController,
    RLController.name: RLController,
}


def controller_names() -> tuple[str, ...]:
    """Return valid configuration/CLI names in stable display order."""
    return tuple(CONTROLLER_TYPES)


def create_controller(name: str, team: str = "blue") -> PlayerController:
    """Construct a controller selected by configuration or command line."""
    try:
        controller_type = CONTROLLER_TYPES[name]
    except KeyError as error:
        choices = ", ".join(controller_names())
        raise ValueError(
            f"Unknown controller {name!r}; choose one of: {choices}",
        ) from error
    if controller_type is FixedSequenceController:
        deployment_row = 25 if team == "blue" else 6
        return FixedSequenceController(
            (
                ScheduledPlay(3.0, "Giant", (4, deployment_row)),
                ScheduledPlay(8.0, "Archers", (4, deployment_row)),
                ScheduledPlay(13.0, "Knight", (13, deployment_row)),
                ScheduledPlay(18.0, "Fireball", (13, 23 if team == "blue" else 8)),
            ),
        )
    return controller_type()
