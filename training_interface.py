"""Numerical observation and discrete-action boundary for learned policies.

This module deliberately exposes public battle information rather than pixels.
Private opponent information (hand, queue, and Elixir) never enters an
observation.  The representation uses only numbers, tuples, and dictionaries,
so a training wrapper can convert it to NumPy arrays without depending on the
Pygame renderer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arena_viewer import (
    ARENA_HEIGHT,
    ARENA_LEFT,
    ARENA_WIDTH,
    CARD_CATALOG,
    GRID_COLUMNS,
    GRID_ROWS,
    MATCH_DURATION_SECONDS,
    OVERTIME_DURATION_SECONDS,
)
from battle_engine import EntityState
from controllers import PlayCardAction

if TYPE_CHECKING:
    from arena_viewer import Match


WAIT_ACTION = 0
PLACEMENT_ACTIONS = 4 * GRID_COLUMNS * GRID_ROWS
ACTION_COUNT = 1 + PLACEMENT_ACTIONS

PHASE_REGULATION = 0
PHASE_OVERTIME = 1
PHASE_TIEBREAKER = 2
PHASE_FINISHED = 3

_ENTITY_STATE_IDS = {
    EntityState.RETARGETING: 0,
    EntityState.MOVING: 1,
    EntityState.ATTACKING: 2,
    EntityState.STUNNED: 3,
    EntityState.DEAD: 4,
}


def _entity_type_names() -> tuple[str, ...]:
    """Build a stable vocabulary for cards, spawned troops, and towers."""
    names = {"King Tower", "Princess Tower"}
    for card in CARD_CATALOG:
        names.add(card.name)
        if card.unit_stats is not None and card.unit_stats.spawner is not None:
            names.add(card.unit_stats.spawner.card.name)
    return tuple(sorted(names))


ENTITY_TYPE_NAMES = _entity_type_names()
ENTITY_TYPE_IDS = {name: index + 1 for index, name in enumerate(ENTITY_TYPE_NAMES)}
UNKNOWN_ENTITY_TYPE = 0

CARD_IDS = {card.name: index + 1 for index, card in enumerate(CARD_CATALOG)}


def encode_action(action: PlayCardAction | None) -> int:
    """Map wait or ``(hand slot, row, column)`` to an integer in 0..2304."""
    if action is None:
        return WAIT_ACTION
    column, row = action.tile
    if not 0 <= action.hand_slot < 4:
        raise ValueError("hand slot must be between 0 and 3")
    if not (0 <= column < GRID_COLUMNS and 0 <= row < GRID_ROWS):
        raise ValueError("tile is outside the arena")
    tile_index = row * GRID_COLUMNS + column
    return 1 + action.hand_slot * (GRID_COLUMNS * GRID_ROWS) + tile_index


def decode_action(action_id: int) -> PlayCardAction | None:
    """Invert :func:`encode_action`; action zero means wait."""
    if not isinstance(action_id, int) or isinstance(action_id, bool):
        raise TypeError("action_id must be an integer")
    if not 0 <= action_id < ACTION_COUNT:
        raise ValueError(f"action_id must be between 0 and {ACTION_COUNT - 1}")
    if action_id == WAIT_ACTION:
        return None

    placement_index = action_id - 1
    tiles_per_hand = GRID_COLUMNS * GRID_ROWS
    hand_slot, tile_index = divmod(placement_index, tiles_per_hand)
    row, column = divmod(tile_index, GRID_COLUMNS)
    return PlayCardAction(hand_slot=hand_slot, tile=(column, row))


def action_mask(match: Match, team: str) -> tuple[bool, ...]:
    """Return one legality flag for each discrete action."""
    if team not in match.players:
        raise ValueError("team must be 'blue' or 'red'")

    mask = [False] * ACTION_COUNT
    if getattr(match, "done", getattr(match, "match_finished", False)):
        return tuple(mask)

    # Waiting is always valid during a live match. Placement entries come from
    # the same authoritative validator used by human and scripted controllers.
    mask[WAIT_ACTION] = True
    for action in match.legal_actions_for(team):
        mask[encode_action(action)] = True
    return tuple(mask)


def apply_action(match: Match, team: str, action_id: int) -> bool:
    """Apply a discrete action after checking its current legality."""
    decoded = decode_action(action_id)
    if decoded is None:
        return not match.done
    if not action_mask(match, team)[action_id]:
        return False
    return match.play(team, decoded)


def _relation(owner_team: str, observer_team: str) -> int:
    """Encode own objects as zero and opponent objects as one."""
    return 0 if owner_team == observer_team else 1


def _done(match: Match) -> bool:
    return bool(getattr(match, "done", getattr(match, "match_finished", False)))


def _phase(match: Match) -> int:
    if _done(match):
        return PHASE_FINISHED
    if match.tiebreaker_active:
        return PHASE_TIEBREAKER
    if match.overtime_active:
        return PHASE_OVERTIME
    return PHASE_REGULATION


def _phase_time_remaining(match: Match) -> float:
    if _done(match) or match.tiebreaker_active:
        return 0.0
    if match.overtime_active:
        overtime_elapsed = max(
            0.0,
            match.match_elapsed - MATCH_DURATION_SECONDS,
        )
        return max(0.0, OVERTIME_DURATION_SECONDS - overtime_elapsed)
    return max(0.0, MATCH_DURATION_SECONDS - match.match_elapsed)


def observation(match: Match, team: str) -> dict[str, object]:
    """Build a detached numerical snapshot from one player's perspective.

    Coordinates are normalized to the playable arena. Health is kept as both
    current and maximum values so callers can choose raw or normalized input.
    Variable-length battlefield collections are deterministic and sorted by
    their stable runtime IDs.
    """
    if team not in match.players:
        raise ValueError("team must be 'blue' or 'red'")
    opponent = "red" if team == "blue" else "blue"
    player = match.players[team]
    scores = match.battle.crown_scores

    towers = []
    for entity in sorted(match.battle.entities, key=lambda item: item.entity_id):
        if entity.tower_kind is None:
            continue
        common = (
            ENTITY_TYPE_IDS.get(entity.name, UNKNOWN_ENTITY_TYPE),
            _relation(entity.team, team),
            (entity.position.x - ARENA_LEFT) / ARENA_WIDTH,
            entity.position.y / ARENA_HEIGHT,
            float(entity.health),
            float(entity.max_health),
        )
        towers.append(
            common
            + (
                0 if entity.tower_kind == "king" else 1,
                int(entity.active),
            )
        )

    entities = []
    for entity in sorted(match.battle.living_entities, key=lambda item: item.entity_id):
        if entity.tower_kind is not None:
            continue
        common = (
            ENTITY_TYPE_IDS.get(entity.name, UNKNOWN_ENTITY_TYPE),
            _relation(entity.team, team),
            (entity.position.x - ARENA_LEFT) / ARENA_WIDTH,
            entity.position.y / ARENA_HEIGHT,
            float(entity.health),
            float(entity.max_health),
        )
        lifetime_remaining = -1.0
        if entity.lifetime_seconds is not None:
            lifetime_remaining = max(
                0.0,
                entity.lifetime_seconds - entity.lifetime_elapsed,
            )
        entities.append(
            common
            + (
                int(entity.is_building),
                _ENTITY_STATE_IDS[entity.state],
                float(entity.attack_cooldown),
                float(entity.stun_remaining),
                float(entity.freeze_remaining),
                int(entity.is_jumping),
                lifetime_remaining,
            )
        )

    projectiles = tuple(
        (
            _relation(projectile.team, team),
            (projectile.position.x - ARENA_LEFT) / ARENA_WIDTH,
            projectile.position.y / ARENA_HEIGHT,
            float(projectile.damage),
            float(projectile.speed),
            float(projectile.splash_radius),
            float(projectile.freeze_duration),
        )
        for projectile in sorted(
            match.battle.projectiles,
            key=lambda item: item.projectile_id,
        )
    )
    pending_spells = tuple(
        (
            CARD_IDS.get(spell.card.name, 0),
            _relation(spell.team, team),
            (spell.position[0] - ARENA_LEFT) / ARENA_WIDTH,
            spell.position[1] / ARENA_HEIGHT,
            float(spell.remaining_seconds),
            float(spell.card.spell_stats.radius),
            float(spell.card.spell_stats.damage),
        )
        for spell in match.battle.pending_spells
    )

    return {
        "hand_card_ids": tuple(CARD_IDS[card.name] for card in player.card_cycle.hand),
        "hand_costs": tuple(card.elixir_cost for card in player.card_cycle.hand),
        "elixir": float(player.elixir.amount),
        "tower_state": tuple(towers),
        "crown_scores": (scores[team], scores[opponent]),
        "match_elapsed": float(match.match_elapsed),
        "phase_time_remaining": _phase_time_remaining(match),
        "match_phase": _phase(match),
        "entities": tuple(entities),
        "projectiles": projectiles,
        "pending_spells": pending_spells,
    }


__all__ = [
    "ACTION_COUNT",
    "CARD_IDS",
    "ENTITY_TYPE_IDS",
    "PHASE_FINISHED",
    "PHASE_OVERTIME",
    "PHASE_REGULATION",
    "PHASE_TIEBREAKER",
    "WAIT_ACTION",
    "action_mask",
    "apply_action",
    "decode_action",
    "encode_action",
    "observation",
]
