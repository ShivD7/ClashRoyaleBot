"""Executable Gymnasium contract for the single-agent training wrapper."""

import numpy as np
import pygame
import pytest
from gymnasium.utils.env_checker import check_env

from arena_viewer import CARD_CATALOG
from training_env import (
    CARD_DAMAGE_REWARD_SCALE,
    DECISION_SECONDS,
    DRAW_REWARD,
    LOSS_REWARD,
    NON_TERMINAL_REWARD,
    SIMULATION_STEPS_PER_DECISION,
    WIN_REWARD,
    ClashRoyaleEnv,
)
from training_interface import ACTION_COUNT, WAIT_ACTION


def card_named(name: str):
    """Return one immutable card definition used by the real match engine."""
    return next(card for card in CARD_CATALOG if card.name == name)


def damage_once_during_next_step(env, target, amount: float) -> None:
    """Inject one controlled hit while preserving the normal match update."""
    original_update = env.match.update_simulation
    damage_applied = False

    def update_with_damage() -> None:
        nonlocal damage_applied
        if not damage_applied:
            target.take_damage(amount)
            damage_applied = True
        original_update()

    env.match.update_simulation = update_with_damage


def test_reset_returns_valid_observation_and_action_mask_without_a_window() -> None:
    display_before = pygame.display.get_surface()
    env = ClashRoyaleEnv()

    observation, info = env.reset(seed=42)

    assert env.observation_space.contains(observation)
    assert info["action_mask"].shape == (ACTION_COUNT,)
    assert info["action_mask"].dtype == np.bool_
    assert info["action_mask"][WAIT_ACTION]
    assert info["action_accepted"] is None
    assert pygame.display.get_surface() is display_before


def test_step_advances_five_shared_physics_updates_and_schedules_opponent() -> None:
    env = ClashRoyaleEnv()
    env.reset(seed=42)

    observation, reward, terminated, truncated, info = env.step(WAIT_ACTION)

    assert env.observation_space.contains(observation)
    assert info["simulation_steps"] == SIMULATION_STEPS_PER_DECISION == 5
    assert info["match_elapsed"] == DECISION_SECONDS
    assert info["action_accepted"]
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert any(deployment.card for deployment in env.match.deployments)


def test_masked_illegal_action_behaves_as_a_rejected_wait() -> None:
    env = ClashRoyaleEnv()
    _, reset_info = env.reset(seed=1)
    illegal_action = next(
        index for index, legal in enumerate(reset_info["action_mask"]) if not legal
    )

    _, reward, terminated, truncated, info = env.step(illegal_action)

    assert info["action_accepted"] is False
    assert info["simulation_steps"] == 5
    assert reward == 0.0
    assert not terminated
    assert not truncated


def test_action_allowed_by_mask_is_accepted_without_a_play_card_reward() -> None:
    env = ClashRoyaleEnv()
    _, reset_info = env.reset(seed=2)
    legal_placement = next(
        index
        for index, legal in enumerate(reset_info["action_mask"])
        if index != WAIT_ACTION and legal
    )

    _, reward, terminated, truncated, info = env.step(legal_placement)

    assert info["action_accepted"] is True
    assert any(deployment.card for deployment in env.match.deployments)
    assert reward == NON_TERMINAL_REWARD
    assert not terminated
    assert not truncated


def test_decision_limit_uses_gymnasium_truncation_separately_from_termination() -> None:
    env = ClashRoyaleEnv(max_decisions=1)
    env.reset(seed=3)

    _, reward, terminated, truncated, _ = env.step(WAIT_ACTION)

    assert reward == 0.0
    assert not terminated
    assert truncated


def test_same_seed_and_actions_reproduce_the_same_trajectory() -> None:
    env = ClashRoyaleEnv()

    def rollout() -> list[tuple[dict[str, np.ndarray], float, bool, bool]]:
        initial, _ = env.reset(seed=123)
        trajectory = [(initial, 0.0, False, False)]
        for _ in range(4):
            observation, reward, terminated, truncated, _ = env.step(WAIT_ACTION)
            trajectory.append((observation, reward, terminated, truncated))
        return trajectory

    first = rollout()
    second = rollout()

    for first_step, second_step in zip(first, second, strict=True):
        first_observation, *first_result = first_step
        second_observation, *second_result = second_step
        assert first_result == second_result
        for key in first_observation:
            assert np.array_equal(first_observation[key], second_observation[key])


def test_terminal_reward_is_returned_from_the_learning_players_perspective() -> None:
    env = ClashRoyaleEnv()
    env.reset()
    red_king = next(
        entity
        for entity in env.match.battle.entities
        if entity.team == "red" and entity.tower_kind == "king"
    )
    red_king.take_damage(red_king.max_health)

    _, reward, terminated, truncated, info = env.step(WAIT_ACTION)

    assert reward == WIN_REWARD
    assert terminated
    assert not truncated
    assert info["winner"] == "blue"


def test_loss_has_negative_reward() -> None:
    env = ClashRoyaleEnv()
    env.reset()
    blue_king = next(
        entity
        for entity in env.match.battle.entities
        if entity.team == "blue" and entity.tower_kind == "king"
    )
    blue_king.take_damage(blue_king.max_health)

    _, reward, terminated, truncated, info = env.step(WAIT_ACTION)

    assert reward == LOSS_REWARD
    assert reward < 0
    assert terminated
    assert not truncated
    assert info["winner"] == "red"


def test_red_side_evaluation_uses_red_relative_observation_and_reward() -> None:
    env = ClashRoyaleEnv(learner_team="red", opponent="random")
    observation, info = env.reset(seed=9)
    blue_king = next(
        entity
        for entity in env.match.battle.entities
        if entity.team == "blue" and entity.tower_kind == "king"
    )
    blue_king.take_damage(blue_king.max_health)

    _, reward, terminated, truncated, result = env.step(WAIT_ACTION)

    assert env.observation_space.contains(observation)
    assert info["action_mask"][WAIT_ACTION]
    assert reward == WIN_REWARD
    assert terminated
    assert not truncated
    assert result["winner"] == "red"


def test_tiebreaker_draw_has_zero_reward() -> None:
    env = ClashRoyaleEnv()
    env.reset()
    env.match.tiebreaker_active = True
    env.match.tiebreaker_notice_remaining = 0.0
    for tower in env.match.battle.living_crown_towers:
        tower.health = 1.0

    _, reward, terminated, truncated, info = env.step(WAIT_ACTION)

    assert reward == DRAW_REWARD
    assert terminated
    assert not truncated
    assert info["winner"] is None


def test_automatic_tiebreaker_damage_has_no_intermediate_reward() -> None:
    env = ClashRoyaleEnv()
    env.reset()
    env.match.tiebreaker_active = True
    env.match.tiebreaker_notice_remaining = 0.0

    _, reward, terminated, truncated, _ = env.step(WAIT_ACTION)

    assert reward == NON_TERMINAL_REWARD
    assert not terminated
    assert not truncated


def test_damaging_an_opponent_card_gives_a_very_small_reward() -> None:
    env = ClashRoyaleEnv()
    env.reset(seed=4)
    enemy_knight = env.match.battle.deploy_card(
        card_named("Knight"),
        "red",
        (env.match.battle.arena_left + 50, 80),
    )[0]
    damage_once_during_next_step(env, enemy_knight, 100.0)

    _, reward, terminated, truncated, info = env.step(WAIT_ACTION)

    expected = CARD_DAMAGE_REWARD_SCALE * 100.0 / enemy_knight.max_health
    assert reward == pytest.approx(expected)
    assert info["reward_components"]["card_damage"] == pytest.approx(expected)
    assert 0.0 < reward < 0.001
    assert not terminated
    assert not truncated


def test_friendly_card_damage_has_the_equal_opposite_penalty() -> None:
    env = ClashRoyaleEnv()
    env.reset(seed=5)
    friendly_knight = env.match.battle.deploy_card(
        card_named("Knight"),
        "blue",
        (
            env.match.battle.arena_left + 50,
            env.match.battle.screen_height - 80,
        ),
    )[0]
    damage_once_during_next_step(env, friendly_knight, 100.0)

    _, reward, terminated, truncated, info = env.step(WAIT_ACTION)

    expected = -CARD_DAMAGE_REWARD_SCALE * 100.0 / friendly_knight.max_health
    assert reward == pytest.approx(expected)
    assert info["reward_components"]["card_damage"] == pytest.approx(expected)
    assert -0.001 < reward < 0.0
    assert not terminated
    assert not truncated


def test_building_lifetime_decay_does_not_count_as_card_damage() -> None:
    env = ClashRoyaleEnv()
    env.reset(seed=6)
    env.match.battle.deploy_card(
        card_named("Cannon"),
        "red",
        (env.match.battle.arena_left + 50, 80),
    )

    _, reward, terminated, truncated, info = env.step(WAIT_ACTION)

    assert reward == pytest.approx(NON_TERMINAL_REWARD, abs=1e-12)
    assert info["reward_components"]["card_damage"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert not terminated
    assert not truncated


def test_complete_episode_reaches_match_termination_before_safety_limit() -> None:
    env = ClashRoyaleEnv()
    env.reset(seed=42)
    terminated = False
    truncated = False
    final_reward = NON_TERMINAL_REWARD
    info = {}

    while not (terminated or truncated):
        _, final_reward, terminated, truncated, info = env.step(WAIT_ACTION)

    assert terminated
    assert not truncated
    assert info["winner"] == "red"
    assert info["reward_components"]["terminal"] == LOSS_REWARD
    assert final_reward < 0.0
    assert env.decision_count < env.max_decisions


def test_environment_satisfies_gymnasiums_api_checker() -> None:
    check_env(ClashRoyaleEnv(), skip_render_check=True)
