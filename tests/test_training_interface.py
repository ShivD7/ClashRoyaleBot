"""Contracts for numerical observations and the discrete action space."""

import pygame
import pytest

from arena_viewer import GRID_COLUMNS, GRID_ROWS, Match
from controllers import PlayCardAction, RandomController
from training_interface import (
    ACTION_COUNT,
    WAIT_ACTION,
    action_mask,
    decode_action,
    encode_action,
)


def test_action_encoding_reserves_zero_for_wait_and_round_trips() -> None:
    assert ACTION_COUNT == 2305
    assert encode_action(None) == WAIT_ACTION
    assert decode_action(WAIT_ACTION) is None

    actions = (
        PlayCardAction(0, (0, 0)),
        PlayCardAction(2, (9, 16)),
        PlayCardAction(3, (GRID_COLUMNS - 1, GRID_ROWS - 1)),
    )
    for action in actions:
        assert decode_action(encode_action(action)) == action

    assert encode_action(actions[0]) == 1
    assert encode_action(actions[-1]) == ACTION_COUNT - 1


@pytest.mark.parametrize("action_id", [-1, ACTION_COUNT])
def test_action_decoding_rejects_out_of_range_values(action_id: int) -> None:
    with pytest.raises(ValueError):
        decode_action(action_id)


def test_action_mask_matches_the_authoritative_legal_action_list() -> None:
    match = Match()
    mask = action_mask(match, "blue")
    legal_ids = {encode_action(action) for action in match.legal_actions_for("blue")}

    assert len(mask) == ACTION_COUNT
    assert mask[WAIT_ACTION]
    assert {index for index, legal in enumerate(mask) if legal} == {
        WAIT_ACTION,
        *legal_ids,
    }


def test_observation_is_numeric_and_does_not_expose_opponent_private_state() -> None:
    match = Match()
    match.players["red"].elixir.amount = 9.25
    match.players["red"].card_cycle.play(0)

    observed = match.observation("blue")

    assert observed["elixir"] == 5.0
    assert len(observed["hand_card_ids"]) == 4
    assert len(observed["hand_costs"]) == 4
    assert len(observed["tower_state"]) == 6
    assert observed["crown_scores"] == (0, 0)
    assert "opponent_elixir" not in observed
    assert "opponent_hand" not in observed
    assert "opponent_queue" not in observed
    assert "elixir" not in match.state
    assert "hands" not in match.state

    def assert_numeric(value: object) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                assert_numeric(nested)
            return
        if isinstance(value, tuple):
            for nested in value:
                assert_numeric(nested)
            return
        assert isinstance(value, (int, float))

    assert_numeric(observed)


def test_destroyed_towers_remain_in_the_fixed_tower_observation() -> None:
    match = Match()
    tower = next(
        entity
        for entity in match.battle.entities
        if entity.team == "red" and entity.tower_kind == "princess"
    )
    tower.take_damage(tower.max_health)

    tower_state = match.observation("blue")["tower_state"]

    assert len(tower_state) == 6
    assert any(record[4] == 0.0 for record in tower_state)


def test_discrete_wait_does_not_mutate_match_and_placement_uses_match_rules() -> None:
    match = Match()
    before = match.state

    assert match.apply_discrete_action("blue", WAIT_ACTION)
    assert match.state == before

    legal_action = match.legal_actions_for("blue")[0]
    action_id = encode_action(legal_action)
    assert match.action_mask("blue")[action_id]
    assert match.apply_discrete_action("blue", action_id)
    assert len(match.deployments) == 1


def test_random_controllers_finish_without_opening_a_window() -> None:
    display_before = pygame.display.get_surface()
    match = Match(
        blue_controller=RandomController(seed=1, play_probability=1.0),
        red_controller=RandomController(seed=2, play_probability=1.0),
    )

    winner = match.run_until_finished()

    assert match.done
    assert winner in {"blue", "red", None}
    assert pygame.display.get_surface() is display_before
