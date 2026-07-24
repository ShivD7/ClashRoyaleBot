from types import SimpleNamespace

import pytest

from arena_viewer import (
    ARENA_HEIGHT,
    ARENA_LEFT,
    ARENA_RIGHT,
    ARENA_WIDTH,
    DEFAULT_DECK,
    DOUBLE_ELIXIR_START,
    ELIXIR_MULTIPLIER_NOTICE_SECONDS,
    ENEMY_DEPLOYMENT_UNLOCK_TOP,
    GRID_COLUMNS,
    GRID_ROWS,
    HUD_HEIGHT,
    LEFT_LANE_X,
    MATCH_DURATION_SECONDS,
    OVERTIME_DURATION_SECONDS,
    OVERTIME_NOTICE_SECONDS,
    RIGHT_LANE_X,
    ArenaViewer,
    CardCycle,
    ElixirMeter,
    PlacementRule,
    RIVER_TOP,
    RIVER_HEIGHT,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
    TOWERS,
    TRIPLE_ELIXIR_START,
    WINDOW_HEIGHT,
    WINDOW_SCALE,
    WINDOW_WIDTH,
    format_match_time,
    remaining_match_seconds,
)
from battle_engine import BattleEngine


def test_screen_positions_convert_to_expected_tiles() -> None:
    assert ArenaViewer.screen_to_tile((ARENA_LEFT, 0)) == (0, 0)
    assert ArenaViewer.screen_to_tile((ARENA_LEFT + 24, 24)) == (0, 0)
    assert ArenaViewer.screen_to_tile((ARENA_LEFT + 25, 25)) == (1, 1)
    assert ArenaViewer.screen_to_tile(
        (ARENA_RIGHT - 1, ARENA_HEIGHT - 1),
    ) == (
        17,
        (ARENA_HEIGHT - 1) // TILE_SIZE,
    )


def test_positions_outside_arena_are_rejected() -> None:
    assert ArenaViewer.screen_to_tile((-1, 0)) is None
    assert ArenaViewer.screen_to_tile((ARENA_LEFT, -1)) is None
    assert ArenaViewer.screen_to_tile((ARENA_LEFT - 1, 0)) is None
    assert ArenaViewer.screen_to_tile((ARENA_RIGHT, 0)) is None
    assert ArenaViewer.screen_to_tile((ARENA_LEFT, ARENA_HEIGHT)) is None


def test_stadium_buffers_do_not_change_playable_grid_width() -> None:
    assert ARENA_WIDTH == GRID_COLUMNS * TILE_SIZE
    assert SCREEN_WIDTH == ARENA_WIDTH + ARENA_LEFT * 2


def test_scaled_window_preserves_logical_arena_and_card_input() -> None:
    assert WINDOW_SCALE == 0.75
    assert WINDOW_WIDTH < SCREEN_WIDTH
    assert WINDOW_HEIGHT < SCREEN_HEIGHT

    logical_tile_center = (ARENA_LEFT + TILE_SIZE // 2, TILE_SIZE // 2)
    display_tile_center = (
        round(logical_tile_center[0] * WINDOW_WIDTH / SCREEN_WIDTH),
        round(logical_tile_center[1] * WINDOW_HEIGHT / SCREEN_HEIGHT),
    )
    remapped_tile_center = ArenaViewer.display_to_logical_position(
        display_tile_center,
    )
    assert ArenaViewer.screen_to_tile(remapped_tile_center) == (0, 0)

    first_card = ArenaViewer.hand_card_rectangles()[0]
    display_card_center = (
        round(first_card.centerx * WINDOW_WIDTH / SCREEN_WIDTH),
        round(first_card.centery * WINDOW_HEIGHT / SCREEN_HEIGHT),
    )
    remapped_card_center = ArenaViewer.display_to_logical_position(
        display_card_center,
    )
    assert ArenaViewer.hand_index_at(remapped_card_center) == 0


def test_arena_height_matches_all_grid_rows() -> None:
    assert ARENA_HEIGHT == GRID_ROWS * TILE_SIZE
    assert SCREEN_HEIGHT == ARENA_HEIGHT + HUD_HEIGHT


def test_each_team_has_two_princess_towers_and_one_king_tower() -> None:
    for team in ("red", "blue"):
        team_towers = [tower for tower in TOWERS if tower.team == team]

        assert [tower.kind for tower in team_towers].count("princess") == 2
        assert [tower.kind for tower in team_towers].count("king") == 1


def test_tower_layout_is_mirrored_across_the_arena() -> None:
    red_towers = {(tower.kind, tower.center) for tower in TOWERS if tower.team == "red"}
    mirrored_blue_towers = {
        (tower.kind, (tower.center[0], ARENA_HEIGHT - tower.center[1]))
        for tower in TOWERS
        if tower.team == "blue"
    }

    assert red_towers == mirrored_blue_towers


def test_towers_are_aligned_symmetrically_across_the_center_lane() -> None:
    red_king = next(
        tower for tower in TOWERS if tower.team == "red" and tower.kind == "king"
    )
    red_princess_towers = [
        tower
        for tower in TOWERS
        if tower.team == "red" and tower.kind == "princess"
    ]

    left_tower, right_tower = sorted(
        red_princess_towers,
        key=lambda tower: tower.center[0],
    )

    assert red_king.center[0] == SCREEN_WIDTH // 2
    assert left_tower.center[1] == right_tower.center[1]
    assert left_tower.center[0] + right_tower.center[0] == SCREEN_WIDTH


def test_bridges_align_with_princess_tower_lanes_and_span_river() -> None:
    red_princess_x_positions = sorted(
        tower.center[0]
        for tower in TOWERS
        if tower.team == "red" and tower.kind == "princess"
    )
    river = ArenaViewer.river_rectangle()
    bridges = ArenaViewer.bridge_rectangles()

    assert [bridge.centerx for bridge in bridges] == red_princess_x_positions

    for bridge in bridges:
        assert bridge.centery == river.centery
        assert bridge.top < river.top
        assert bridge.bottom > river.bottom


def test_ground_units_can_cross_only_at_bridges() -> None:
    river = ArenaViewer.river_rectangle()
    left_bridge, right_bridge = ArenaViewer.bridge_rectangles()

    assert ArenaViewer.is_walkable_position(left_bridge.center)
    assert ArenaViewer.is_walkable_position(right_bridge.center)
    assert not ArenaViewer.is_walkable_position((SCREEN_WIDTH // 2, river.centery))
    assert ArenaViewer.is_walkable_position((SCREEN_WIDTH // 2, river.top - 1))


def test_card_cycle_starts_with_four_cards_and_a_next_card() -> None:
    cycle = CardCycle(DEFAULT_DECK)

    assert cycle.hand == list(DEFAULT_DECK[:4])
    assert cycle.next_card == DEFAULT_DECK[4]


def test_default_deck_has_explicit_behavior_categories() -> None:
    cards = {card.name: card for card in DEFAULT_DECK}
    expected_behaviors = {
        "Knight": (
            "mini_tank",
            PlacementRule.FRIENDLY_TERRITORY,
            "nearest_enemy",
            "ground",
            "melee",
            1,
        ),
        "Archers": (
            "ranged_support",
            PlacementRule.FRIENDLY_TERRITORY,
            "nearest_enemy",
            "air_and_ground",
            "ranged",
            2,
        ),
        "Giant": (
            "win_condition",
            PlacementRule.FRIENDLY_TERRITORY,
            "buildings_only",
            "ground",
            "melee",
            1,
        ),
        "Fireball": (
            "big_spell",
            PlacementRule.ANYWHERE,
            "targeted_area",
            "air_and_ground",
            "area",
            0,
        ),
        "Mini P.E.K.K.A": (
            "tank_killer",
            PlacementRule.FRIENDLY_TERRITORY,
            "nearest_enemy",
            "ground",
            "melee",
            1,
        ),
        "Musketeer": (
            "ranged_support",
            PlacementRule.FRIENDLY_TERRITORY,
            "nearest_enemy",
            "air_and_ground",
            "ranged",
            1,
        ),
        "Skeletons": (
            "cycle_swarm",
            PlacementRule.FRIENDLY_TERRITORY,
            "nearest_enemy",
            "ground",
            "melee",
            3,
        ),
        "Zap": (
            "small_spell",
            PlacementRule.ANYWHERE,
            "targeted_area",
            "air_and_ground",
            "area",
            0,
        ),
    }

    for card_name, expected in expected_behaviors.items():
        card = cards[card_name]
        actual = (
            card.role,
            card.placement_rule,
            card.target_priority,
            card.target_types,
            card.attack_style,
            card.unit_count,
        )
        assert actual == expected


def test_fireball_and_zap_have_current_damage_radii() -> None:
    cards = {card.name: card for card in DEFAULT_DECK}

    assert cards["Fireball"].spell_stats is not None
    assert cards["Fireball"].spell_stats.radius == 2.5
    assert cards["Zap"].spell_stats is not None
    assert cards["Zap"].spell_stats.radius == 2.5


def test_played_card_moves_to_back_and_next_card_fills_same_slot() -> None:
    cycle = CardCycle(DEFAULT_DECK)
    played_card = cycle.play(1)

    assert played_card == DEFAULT_DECK[1]
    assert cycle.hand == [
        DEFAULT_DECK[0],
        DEFAULT_DECK[4],
        DEFAULT_DECK[2],
        DEFAULT_DECK[3],
    ]
    assert cycle.queue == [
        DEFAULT_DECK[5],
        DEFAULT_DECK[6],
        DEFAULT_DECK[7],
        DEFAULT_DECK[1],
    ]


def test_card_cycle_requires_exactly_eight_cards() -> None:
    with pytest.raises(ValueError):
        CardCycle(DEFAULT_DECK[:7])


def test_deployment_tiles_are_only_allowed_before_blue_bridges() -> None:
    first_player_row = (RIVER_TOP + RIVER_HEIGHT) // TILE_SIZE

    assert not ArenaViewer.is_valid_deployment_tile((5, first_player_row - 1))
    assert ArenaViewer.is_valid_deployment_tile((5, first_player_row))
    assert ArenaViewer.is_valid_deployment_tile((5, 31))
    assert not ArenaViewer.is_valid_deployment_tile((18, first_player_row))


def test_spells_can_target_any_arena_tile_while_troops_cannot() -> None:
    cards = {card.name: card for card in DEFAULT_DECK}
    enemy_tile = (5, 5)
    river_tile = (5, RIVER_TOP // TILE_SIZE)

    assert not ArenaViewer.is_valid_deployment_tile(enemy_tile, cards["Giant"])
    assert not ArenaViewer.is_valid_deployment_tile(river_tile, cards["Knight"])
    assert ArenaViewer.is_valid_deployment_tile(enemy_tile, cards["Fireball"])
    assert ArenaViewer.is_valid_deployment_tile(river_tile, cards["Zap"])
    assert ArenaViewer.restricted_deployment_tiles(cards["Fireball"]) == ()


def test_destroyed_princess_towers_unlock_their_enemy_lane_halves() -> None:
    knight = next(card for card in DEFAULT_DECK if card.name == "Knight")
    unlocked_row = ENEMY_DEPLOYMENT_UNLOCK_TOP // TILE_SIZE
    left_tile = (GRID_COLUMNS // 4, unlocked_row)
    right_tile = (GRID_COLUMNS * 3 // 4, unlocked_row)
    too_deep_tile = (GRID_COLUMNS // 4, unlocked_row - 1)

    assert ArenaViewer.is_valid_deployment_tile(
        left_tile,
        knight,
        frozenset({"left"}),
    )
    assert not ArenaViewer.is_valid_deployment_tile(
        right_tile,
        knight,
        frozenset({"left"}),
    )
    assert not ArenaViewer.is_valid_deployment_tile(
        too_deep_tile,
        knight,
        frozenset({"left"}),
    )

    assert ArenaViewer.is_valid_deployment_tile(
        right_tile,
        knight,
        frozenset({"right"}),
    )
    assert ArenaViewer.is_valid_deployment_tile(
        left_tile,
        knight,
        frozenset({"left", "right"}),
    )
    assert ArenaViewer.is_valid_deployment_tile(
        right_tile,
        knight,
        frozenset({"left", "right"}),
    )


def test_live_destroyed_tower_immediately_updates_ground_card_placement() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.selected_card_index = 0
    viewer.elixir = ElixirMeter(amount=10)
    viewer.card_cycle = CardCycle(DEFAULT_DECK)
    viewer.deployments = []
    viewer.battle = BattleEngine(
        tile_size=TILE_SIZE,
        screen_height=ARENA_HEIGHT,
        river_top=RIVER_TOP,
        river_height=RIVER_HEIGHT,
        bridge_x_positions=(LEFT_LANE_X, RIGHT_LANE_X),
        tower_layout=tuple(
            (tower.kind, tower.team, tower.center)
            for tower in TOWERS
        ),
    )
    red_left_princess = next(
        entity
        for entity in viewer.battle.entities
        if (
            entity.team == "red"
            and entity.tower_kind == "princess"
            and entity.position.x == LEFT_LANE_X
        )
    )
    red_left_princess.take_damage(red_left_princess.max_health)
    unlocked_left_tile = (
        GRID_COLUMNS // 4,
        ENEMY_DEPLOYMENT_UNLOCK_TOP // TILE_SIZE,
    )

    assert viewer.destroyed_enemy_princess_lanes() == frozenset({"left"})
    assert viewer.try_play_selected_card(unlocked_left_tile)


def test_king_tower_winner_finishes_viewer_and_clears_card_interaction() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.match_finished = False
    viewer.match_winner = None
    viewer.match_finished_at_ms = None
    viewer.match_started_at = 1_000
    viewer.battle = SimpleNamespace(winning_team="blue")
    viewer.selected_card_index = 2
    viewer.dragged_card_index = 2
    viewer.drag_position = (100, 100)

    viewer.update_match_state(now_ms=2_000)

    assert viewer.match_finished
    assert viewer.match_winner == "blue"
    assert viewer.match_finished_at_ms == 2_000
    assert viewer.selected_card_index is None
    assert viewer.dragged_card_index is None
    assert viewer.drag_position is None


def test_match_result_reports_winner_or_draw_with_final_crown_score() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.battle = SimpleNamespace(crown_scores={"red": 1, "blue": 2})
    viewer.match_winner = "blue"

    assert viewer.match_result_text() == ("BLUE WINS", "RED 1  -  2 BLUE")

    viewer.battle.crown_scores = {"red": 1, "blue": 1}
    viewer.match_winner = None
    assert viewer.match_result_text() == ("DRAW", "RED 1  -  1 BLUE")


def test_play_again_resets_every_mutable_match_value() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    old_battle = viewer.create_battle_engine()
    old_battle.entities[0].take_damage(old_battle.entities[0].max_health)
    viewer.battle = old_battle
    viewer.elixir = ElixirMeter(amount=1)
    viewer.card_cycle = CardCycle(DEFAULT_DECK)
    viewer.card_cycle.play(0)
    viewer.deployments = [SimpleNamespace()]
    viewer.match_started_at = 1
    viewer.match_elapsed = 250.0
    viewer.match_finished = True
    viewer.match_winner = "blue"
    viewer.match_finished_at_ms = 251_000
    viewer.overtime_active = True
    viewer.overtime_started_at_ms = 181_000
    viewer.selected_tile = (4, 20)
    viewer.selected_card_index = 2
    viewer.dragged_card_index = 2
    viewer.drag_position = (200, 500)
    viewer.elixir_multiplier_notice = 3
    viewer.elixir_multiplier_notice_remaining = 1.0
    viewer.overtime_notice_remaining = 1.0

    viewer.reset_match(now_ms=999_000)

    assert viewer.battle is not old_battle
    assert len(viewer.battle.entities) == 6
    assert all(entity.is_alive for entity in viewer.battle.entities)
    assert viewer.battle.projectiles == []
    assert viewer.elixir.amount == 5
    assert viewer.card_cycle.hand == list(DEFAULT_DECK[:4])
    assert viewer.card_cycle.queue == list(DEFAULT_DECK[4:])
    assert viewer.deployments == []
    assert viewer.match_started_at == 999_000
    assert viewer.match_elapsed == 0
    assert not viewer.match_finished
    assert viewer.match_winner is None
    assert viewer.match_finished_at_ms is None
    assert not viewer.overtime_active
    assert viewer.overtime_started_at_ms is None
    assert viewer.selected_tile is None
    assert viewer.selected_card_index is None
    assert viewer.dragged_card_index is None
    assert viewer.drag_position is None
    assert viewer.elixir_multiplier_notice is None
    assert viewer.elixir_multiplier_notice_remaining == 0
    assert viewer.overtime_notice_remaining == 0


def test_fireball_can_be_deployed_in_enemy_territory() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.selected_card_index = 3
    viewer.elixir = ElixirMeter(amount=4)
    viewer.card_cycle = CardCycle(DEFAULT_DECK)
    viewer.deployments = []
    enemy_tile = (5, 5)

    assert viewer.try_play_selected_card(enemy_tile)
    assert viewer.elixir.amount == 0
    assert viewer.deployments[0].card.name == "Fireball"
    assert viewer.deployments[0].tile == enemy_tile


def test_restricted_tiles_match_deployment_validation() -> None:
    restricted_tiles = set(ArenaViewer.restricted_deployment_tiles())

    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            tile = (column, row)
            assert (tile in restricted_tiles) is (
                not ArenaViewer.is_valid_deployment_tile(tile)
            )


def test_failed_deployment_does_not_spend_or_cycle() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.selected_card_index = 0
    viewer.elixir = ElixirMeter(amount=10)
    viewer.card_cycle = CardCycle(DEFAULT_DECK)
    viewer.deployments = []
    original_hand = viewer.card_cycle.hand.copy()

    assert not viewer.try_play_selected_card((5, 5))
    assert viewer.elixir.amount == 10
    assert viewer.card_cycle.hand == original_hand
    assert viewer.deployments == []


def test_unaffordable_deployment_does_not_cycle() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.selected_card_index = 2  # Giant costs five Elixir.
    viewer.elixir = ElixirMeter(amount=4)
    viewer.card_cycle = CardCycle(DEFAULT_DECK)
    viewer.deployments = []
    original_hand = viewer.card_cycle.hand.copy()

    assert not viewer.try_play_selected_card((5, 20))
    assert viewer.elixir.amount == 4
    assert viewer.card_cycle.hand == original_hand
    assert viewer.deployments == []


def test_successful_deployment_spends_elixir_records_and_cycles() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.selected_card_index = 0
    viewer.elixir = ElixirMeter(amount=5)
    viewer.card_cycle = CardCycle(DEFAULT_DECK)
    viewer.deployments = []
    tile = (5, 20)

    assert viewer.try_play_selected_card(tile)
    assert viewer.elixir.amount == 2
    assert viewer.deployments[0].card == DEFAULT_DECK[0]
    assert viewer.deployments[0].tile == tile
    assert viewer.card_cycle.hand[0] == DEFAULT_DECK[4]
    assert viewer.selected_card_index is None


def test_dragging_card_to_valid_tile_deploys_and_clears_drag_state() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.selected_card_index = None
    viewer.dragged_card_index = None
    viewer.drag_position = None
    viewer.elixir = ElixirMeter(amount=5)
    viewer.card_cycle = CardCycle(DEFAULT_DECK)
    viewer.deployments = []

    viewer.begin_card_drag(0, (44, 700))

    assert viewer.selected_card_index == 0
    assert viewer.dragged_card_index == 0
    assert viewer.finish_card_drag(
        (ARENA_LEFT + 5 * TILE_SIZE + 12, 20 * TILE_SIZE + 12),
    )
    assert viewer.deployments[0].tile == (5, 20)
    assert viewer.dragged_card_index is None
    assert viewer.drag_position is None


def test_dragged_spell_preview_converts_tile_radius_to_pixels() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.card_cycle = CardCycle(DEFAULT_DECK)
    viewer.dragged_card_index = 3  # Fireball is the fourth starting card.
    viewer.drag_position = (ARENA_LEFT + 200, 300)

    preview = viewer.dragged_spell_preview()

    assert preview is not None
    card, center, radius_pixels = preview
    assert card.name == "Fireball"
    assert center == viewer.drag_position
    assert radius_pixels == 63


def test_dragged_troop_has_no_spell_radius_preview() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.card_cycle = CardCycle(DEFAULT_DECK)
    viewer.dragged_card_index = 0  # Knight is a troop.
    viewer.drag_position = (ARENA_LEFT + 200, 300)

    assert viewer.dragged_spell_preview() is None


def test_dragging_card_to_restricted_tile_does_not_spend_elixir() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.selected_card_index = None
    viewer.dragged_card_index = None
    viewer.drag_position = None
    viewer.elixir = ElixirMeter(amount=5)
    viewer.card_cycle = CardCycle(DEFAULT_DECK)
    viewer.deployments = []

    viewer.begin_card_drag(0, (44, 700))

    assert not viewer.finish_card_drag(
        (ARENA_LEFT + 5 * TILE_SIZE + 12, 5 * TILE_SIZE + 12),
    )
    assert viewer.elixir.amount == 5
    assert viewer.deployments == []
    assert viewer.dragged_card_index is None


def test_river_is_centered_vertically() -> None:
    river = ArenaViewer.river_rectangle()

    assert river.centery == ARENA_HEIGHT // 2


def test_match_duration_is_three_minutes() -> None:
    assert MATCH_DURATION_SECONDS == 180


def make_match_state_viewer(
    *,
    red_crowns: int,
    blue_crowns: int,
) -> ArenaViewer:
    """Build the state needed to test match phases without opening a window."""
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.match_finished = False
    viewer.match_winner = None
    viewer.match_finished_at_ms = None
    viewer.match_started_at = 1_000
    viewer.overtime_active = False
    viewer.overtime_started_at_ms = None
    viewer.overtime_notice_remaining = 0.0
    viewer.battle = SimpleNamespace(
        winning_team=None,
        crown_scores={"red": red_crowns, "blue": blue_crowns},
    )
    viewer.selected_card_index = None
    viewer.dragged_card_index = None
    viewer.drag_position = None
    viewer.elixir_multiplier_notice = None
    viewer.elixir_multiplier_notice_remaining = 0.0
    return viewer


def test_tied_regulation_score_starts_two_minute_overtime() -> None:
    viewer = make_match_state_viewer(red_crowns=1, blue_crowns=1)

    viewer.update_match_state(now_ms=181_000)

    assert viewer.overtime_active
    assert viewer.overtime_started_at_ms == 181_000
    assert viewer.overtime_notice_remaining == OVERTIME_NOTICE_SECONDS
    assert not viewer.match_finished
    assert remaining_match_seconds(
        viewer.overtime_started_at_ms,
        181_000,
        OVERTIME_DURATION_SECONDS,
    ) == 120


def test_overtime_announcement_counts_down_and_disappears() -> None:
    viewer = make_match_state_viewer(red_crowns=0, blue_crowns=0)
    viewer.update_match_state(now_ms=181_000)

    viewer.update_overtime_notice(1.25)
    assert viewer.overtime_notice_remaining == pytest.approx(
        OVERTIME_NOTICE_SECONDS - 1.25,
    )

    viewer.update_overtime_notice(10.0)
    assert viewer.overtime_notice_remaining == 0


def test_regulation_crown_leader_wins_without_overtime() -> None:
    viewer = make_match_state_viewer(red_crowns=1, blue_crowns=2)

    viewer.update_match_state(now_ms=181_000)

    assert viewer.match_finished
    assert viewer.match_winner == "blue"
    assert not viewer.overtime_active


def test_overtime_is_sudden_death_when_a_team_takes_crown_lead() -> None:
    viewer = make_match_state_viewer(red_crowns=1, blue_crowns=1)
    viewer.update_match_state(now_ms=181_000)
    viewer.battle.crown_scores = {"red": 1, "blue": 2}

    viewer.update_match_state(now_ms=200_000)

    assert viewer.match_finished
    assert viewer.match_winner == "blue"


def test_tied_overtime_ends_after_exactly_two_minutes() -> None:
    viewer = make_match_state_viewer(red_crowns=0, blue_crowns=0)
    viewer.update_match_state(now_ms=181_000)

    viewer.update_match_state(now_ms=301_000)

    assert viewer.match_finished
    assert viewer.match_winner is None
    assert viewer.match_finished_at_ms == 301_000


def test_match_timer_counts_down_and_stops_at_zero() -> None:
    start_ms = 1_000

    assert remaining_match_seconds(start_ms, start_ms) == 180
    assert remaining_match_seconds(start_ms, start_ms + 1_000) == 179
    assert remaining_match_seconds(start_ms, start_ms + 179_001) == 1
    assert remaining_match_seconds(start_ms, start_ms + 180_000) == 0
    assert remaining_match_seconds(start_ms, start_ms + 999_999) == 0


def test_match_timer_uses_minutes_and_zero_padded_seconds() -> None:
    assert format_match_time(180) == "3:00"
    assert format_match_time(125) == "2:05"
    assert format_match_time(9) == "0:09"
    assert format_match_time(0) == "0:00"


def test_elixir_starts_at_five_and_generates_continuously() -> None:
    meter = ElixirMeter()

    meter.update(1.4, match_elapsed=0.0)

    assert meter.amount == pytest.approx(5.5)


def test_elixir_generation_uses_double_and_triple_rates() -> None:
    double_meter = ElixirMeter(amount=0)
    triple_meter = ElixirMeter(amount=0)

    double_meter.update(1.4, match_elapsed=120.0)
    triple_meter.update(2.8, match_elapsed=240.0)

    assert double_meter.amount == pytest.approx(1.0)
    assert triple_meter.amount == pytest.approx(3.0)


def test_triple_elixir_starts_one_minute_into_overtime_with_notice() -> None:
    assert TRIPLE_ELIXIR_START == MATCH_DURATION_SECONDS + 60

    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.elixir = ElixirMeter()
    viewer.match_elapsed = TRIPLE_ELIXIR_START - 0.25
    viewer.elixir_multiplier_notice = None
    viewer.elixir_multiplier_notice_remaining = 0.0

    viewer.update_elixir_multiplier_notice(0.5)

    assert viewer.elixir_multiplier_notice == 3
    assert (
        viewer.elixir_multiplier_notice_remaining
        == ELIXIR_MULTIPLIER_NOTICE_SECONDS
    )


def test_elixir_generation_accounts_for_rate_boundary_in_same_tick() -> None:
    meter = ElixirMeter(amount=0)

    meter.update(2.0, match_elapsed=119.0)

    assert meter.amount == pytest.approx(3 / 2.8)


def test_crossing_double_elixir_starts_timed_notice() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.elixir = ElixirMeter()
    viewer.match_elapsed = DOUBLE_ELIXIR_START - 0.25
    viewer.elixir_multiplier_notice = None
    viewer.elixir_multiplier_notice_remaining = 0.0

    viewer.update_elixir_multiplier_notice(0.5)

    assert viewer.elixir_multiplier_notice == 2
    assert (
        viewer.elixir_multiplier_notice_remaining
        == ELIXIR_MULTIPLIER_NOTICE_SECONDS
    )


def test_double_elixir_notice_counts_down_without_retriggering() -> None:
    viewer = ArenaViewer.__new__(ArenaViewer)
    viewer.elixir = ElixirMeter()
    viewer.match_elapsed = DOUBLE_ELIXIR_START + 1.0
    viewer.elixir_multiplier_notice = 2
    viewer.elixir_multiplier_notice_remaining = 1.0

    viewer.update_elixir_multiplier_notice(0.25)

    assert viewer.elixir_multiplier_notice == 2
    assert viewer.elixir_multiplier_notice_remaining == pytest.approx(0.75)

    viewer.update_elixir_multiplier_notice(1.0)

    assert viewer.elixir_multiplier_notice is None
    assert viewer.elixir_multiplier_notice_remaining == 0


def test_elixir_caps_at_ten_and_does_not_bank_overflow() -> None:
    meter = ElixirMeter(amount=9.75)

    meter.update(10.0, match_elapsed=0.0)
    assert meter.amount == 10

    assert meter.spend(4)
    assert meter.amount == 6


def test_elixir_can_only_be_spent_when_affordable() -> None:
    meter = ElixirMeter(amount=3.5)

    assert not meter.spend(4)
    assert meter.amount == 3.5

    assert meter.spend(3)
    assert meter.amount == pytest.approx(0.5)
