"""Interchangeable decision makers for human, scripted, and learned players.

This module is intentionally small. A controller is a *decision source*, not a
second game engine. It may inspect a frozen snapshot and request one action,
but it cannot spend Elixir, rotate a hand, spawn a unit, or damage a tower.
``ArenaViewer.try_play_action`` remains the authority for all of those changes.

Decision flow
-------------

1. The match builds a read-only ``ControllerContext`` for one team.
2. The match calls that controller's ``choose_action(context)`` method.
3. The controller returns ``PlayCardAction`` or ``None`` to wait.
4. The match revalidates the request against the latest authoritative state.
5. Only a valid request is applied.

Human input joins the flow after step 2: mouse/keyboard code creates the same
``PlayCardAction`` type. Random, scripted, fixed, human, and future learned
players therefore cannot gain different rule privileges.

RL status
---------
``RLController`` is currently only a library-neutral adapter. The present
``ControllerContext`` exposes hands, Elixir, scores, and legal actions, but not
the complete battlefield. A future Gymnasium/PettingZoo wrapper must add a
numerical observation encoder rather than treating this placeholder as the
finished learning environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Callable, Sequence


# ---------------------------------------------------------------------------
# The small public interface shared by every controller
# ---------------------------------------------------------------------------
# Frozen dataclasses make it harder for a controller to accidentally modify the
# match. They contain descriptions and requested coordinates, not live objects.
@dataclass(frozen=True)
class PlayCardAction:
    """Request that one hand slot be deployed on one arena tile.

    ``hand_slot`` is 0 through 3. ``tile`` is ``(column, row)`` on the 18-by-32
    grid, not a logical-pixel position. The action is only a request; legality
    is checked later by the match.
    """

    hand_slot: int
    tile: tuple[int, int]


@dataclass(frozen=True)
class ControllerCard:
    """Immutable public description of one card currently in a hand.

    It intentionally omits the viewer's live ``Card`` object. Controllers get
    strategic labels and rule categories without receiving a mutable path back
    into match state.
    """

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
    """Read-only decision snapshot supplied by the authoritative match.

    A new object is built at decision time, so it is a snapshot rather than a
    live view. ``legal_actions`` already accounts for affordability, territory,
    destroyed-tower lane unlocks, and occupied footprints. This shape is enough
    for the built-in baselines but deliberately not the final RL observation.
    """

    team: str
    match_elapsed: float
    elixir: float
    hand: tuple[ControllerCard, ...]
    legal_actions: tuple[PlayCardAction, ...]
    crown_scores: dict[str, int]


class PlayerController(ABC):
    """Minimal contract implemented by every source of player decisions.

    The controller may keep private episode memory, but ``reset`` must clear it
    so a rematch does not inherit hidden state from the previous episode.
    """

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
        # Human actions arrive separately through Pygame events in the viewer.
        return None


# ---------------------------------------------------------------------------
# Built-in computer-controlled players
# ---------------------------------------------------------------------------
# These controllers are useful at different stages: Random is a basic baseline,
# Scripted produces understandable lane pushes, and Fixed repeats a curriculum.
class RandomController(PlayerController):
    """Seeded random legal-action baseline for smoke tests and evaluation.

    A private ``random.Random`` avoids changing Python's global random stream.
    Resetting restores the seed, making repeated episodes reproducible.
    """

    name = "random"

    def __init__(self, seed: int = 0, play_probability: float = 0.35) -> None:
        self.seed = seed
        self.play_probability = play_probability
        self._random = random.Random(seed)

    def choose_action(
        self,
        context: ControllerContext,
    ) -> PlayCardAction | None:
        # Waiting some decisions produces a more realistic baseline than playing
        # a card on every possible controller update.
        if not context.legal_actions:
            return None
        if self._random.random() > self.play_probability:
            return None
        return self._random.choice(context.legal_actions)

    def reset(self) -> None:
        self._random.seed(self.seed)


class ScriptedController(PlayerController):
    """Simple deterministic opponent that alternates understandable lane pushes.

    This is a baseline/curriculum opponent, not an attempt at optimal play. It
    first ranks affordable cards by strategic role, then places the winner near
    a desired lane and forward deployment row.
    """

    name = "scripted"

    def __init__(self) -> None:
        self.preferred_lane = 0

    def choose_action(
        self,
        context: ControllerContext,
    ) -> PlayCardAction | None:
        if not context.legal_actions or context.elixir < 3:
            return None

        # Smaller numbers are preferred. Win conditions begin a push, support
        # follows, and spells are saved until no preferred troop is available.
        priority = {
            "win_condition": 0,
            "fast_win_condition": 0,
            "flying_win_condition": 0,
            "ranged_support": 1,
            "splash_support": 1,
            "flying_splash_support": 1,
            "ground_splash_support": 1,
            "flying_swarm": 2,
            "fast_flying_swarm": 2,
            "ground_swarm": 2,
            "fast_ground_swarm": 2,
            "ranged_swarm": 2,
            "tank_killer": 2,
            "heavy_melee_defender": 2,
            "melee_splash_defender": 2,
            "air_defense": 2,
            "defensive_building": 3,
            "air_defense_building": 3,
            "spawner": 3,
            "mini_tank": 3,
            "cycle_swarm": 4,
            "fast_cycle": 4,
            "big_spell": 5,
            "medium_spell": 5,
            "large_spell": 5,
            "small_spell": 6,
        }
        # legal_actions already excludes unaffordable and blocked placements.
        # Reduce it to hand slots first so card choice and tile choice stay clear.
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
        # After choosing a card, place it as close as possible to the desired
        # lane and deployment row. Alternate lanes after every successful play.
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
    """Replay a named, timed curriculum without skipping blocked entries.

    If the next scheduled card is unavailable, unaffordable, or illegal at its
    requested tile, the controller waits and tries that same entry again at the
    next decision. This makes scenarios repeatable for debugging.
    """

    name = "fixed"

    def __init__(self, sequence: Sequence[ScheduledPlay] = ()) -> None:
        self.sequence = tuple(sequence)
        self.next_index = 0

    def choose_action(
        self,
        context: ControllerContext,
    ) -> PlayCardAction | None:
        # Do not skip a scheduled entry. Wait until its time has arrived, its
        # named card is in hand, and its exact tile appears in legal_actions.
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


# ---------------------------------------------------------------------------
# Learned-policy adapter and controller factory
# ---------------------------------------------------------------------------
# RLController does not choose a machine-learning library. Training code injects
# any callable with the Policy shape, keeping this simulator framework-neutral.
class RLController(PlayerController):
    """Adapter for a future learned policy with the same action contract.

    The policy callable is injected instead of importing Torch or a particular
    RL framework here. That separation keeps simulator imports light and lets
    training code choose its own model implementation.
    """

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
        # Mirror the fixed curriculum vertically so the same idea works for
        # either team. Rows nearer each team's own towers are used for deployment.
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
