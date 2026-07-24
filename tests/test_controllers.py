import pytest

from arena_viewer import (
    FIXED_TIMESTEP_SECONDS,
    ArenaViewer,
    DEFAULT_DECK,
    parse_controller_arguments,
)
from controllers import (
    ControllerCard,
    ControllerContext,
    FixedSequenceController,
    HumanController,
    PlayCardAction,
    RLController,
    RandomController,
    ScriptedController,
    controller_names,
    create_controller,
)


def make_controller_viewer() -> ArenaViewer:
    """Create controller-facing match state without opening a Pygame window."""
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.battle = viewer.create_battle_engine()
    viewer.players = viewer.create_player_states()
    viewer.local_team = "blue"
    viewer.sync_local_player_aliases()
    viewer.controllers = {
        "blue": HumanController(),
        "red": ScriptedController(),
    }
    viewer.controller_decision_elapsed = {"blue": 0.0, "red": 0.0}
    viewer.deployments = []
    viewer.match_elapsed = 0.0
    viewer.match_finished = False
    return viewer


def test_all_controller_types_are_registered_and_constructible() -> None:
    assert controller_names() == (
        "human",
        "random",
        "scripted",
        "fixed",
        "rl",
    )
    assert isinstance(create_controller("human"), HumanController)
    assert isinstance(create_controller("random"), RandomController)
    assert isinstance(create_controller("scripted"), ScriptedController)
    assert isinstance(create_controller("fixed"), FixedSequenceController)
    assert isinstance(create_controller("rl"), RLController)


def test_command_line_selects_controllers_without_code_changes() -> None:
    settings = parse_controller_arguments(
        [
            "--blue-controller",
            "rl",
            "--red-controller",
            "fixed",
        ],
    )

    assert settings.blue_controller == "rl"
    assert settings.red_controller == "fixed"


def test_blue_and_red_have_independent_hands_and_elixir() -> None:
    viewer = make_controller_viewer()

    assert viewer.try_play_action("blue", PlayCardAction(2, (4, 25)))

    assert viewer.players["blue"].elixir.amount == 0
    assert viewer.players["red"].elixir.amount == 5
    assert viewer.players["blue"].card_cycle.hand[2] == DEFAULT_DECK[4]
    assert viewer.players["red"].card_cycle.hand[2] == DEFAULT_DECK[2]


def test_red_placement_rules_are_mirrored_and_authoritative() -> None:
    viewer = make_controller_viewer()

    assert not viewer.try_play_action("red", PlayCardAction(2, (4, 25)))
    assert viewer.try_play_action("red", PlayCardAction(2, (4, 6)))
    assert viewer.players["red"].elixir.amount == 0
    assert any(
        entity.team == "red" and entity.name == "Giant"
        for entity in viewer.battle.entities
    )


def test_scripted_controller_only_returns_a_legal_action() -> None:
    viewer = make_controller_viewer()
    context = viewer.controller_context("red")

    action = viewer.controllers["red"].choose_action(context)

    assert action is not None
    assert action in context.legal_actions


def test_match_loop_runs_non_human_controller_but_not_human_controller() -> None:
    viewer = make_controller_viewer()

    viewer.update_controllers(0.25)

    assert viewer.players["red"].elixir.amount == 0
    assert viewer.players["blue"].elixir.amount == 5
    assert any(
        entity.team == "red" and entity.name == "Giant"
        for entity in viewer.battle.entities
    )
    assert not any(
        entity.team == "blue" and not entity.is_building
        for entity in viewer.battle.entities
    )


def test_fixed_simulation_step_updates_both_players_and_controllers() -> None:
    viewer = make_controller_viewer()
    viewer.match_started_at = 0
    viewer.match_finished_at_ms = None
    viewer.match_winner = None
    viewer.overtime_active = False
    viewer.overtime_started_at_ms = None
    viewer.overtime_notice_remaining = 0.0
    viewer.elixir_multiplier_notice = None
    viewer.elixir_multiplier_notice_remaining = 0.0

    viewer.update_simulation()

    expected_elixir = 5 + FIXED_TIMESTEP_SECONDS / 2.8
    assert viewer.players["blue"].elixir.amount == pytest.approx(expected_elixir)
    assert viewer.players["red"].elixir.amount == pytest.approx(expected_elixir)
    assert viewer.controller_decision_elapsed["red"] == pytest.approx(
        FIXED_TIMESTEP_SECONDS,
    )


def test_rl_controller_uses_injected_policy_without_mutating_match() -> None:
    expected = PlayCardAction(0, (4, 25))
    policy_calls = []

    def policy(context: ControllerContext) -> PlayCardAction:
        policy_calls.append(context.team)
        return expected

    context = ControllerContext(
        team="blue",
        match_elapsed=0,
        elixir=5,
        hand=tuple(
            ControllerCard(
                card.name,
                card.elixir_cost,
                card.role,
                card.card_type,
                card.target_priority,
                card.target_types,
                card.movement_type,
                card.attack_style,
            )
            for card in DEFAULT_DECK[:4]
        ),
        legal_actions=(expected,),
        crown_scores={"red": 0, "blue": 0},
    )

    assert RLController(policy).choose_action(context) == expected
    assert policy_calls == ["blue"]
