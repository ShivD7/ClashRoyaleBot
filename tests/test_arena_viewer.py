import pytest

from arena_viewer import (
    MATCH_DURATION_SECONDS,
    ArenaViewer,
    ElixirMeter,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
    TOWERS,
    format_match_time,
    remaining_match_seconds,
)


def test_screen_positions_convert_to_expected_tiles() -> None:
    assert ArenaViewer.screen_to_tile((0, 0)) == (0, 0)
    assert ArenaViewer.screen_to_tile((24, 24)) == (0, 0)
    assert ArenaViewer.screen_to_tile((25, 25)) == (1, 1)
    assert ArenaViewer.screen_to_tile((449, SCREEN_HEIGHT - 1)) == (
        17,
        (SCREEN_HEIGHT - 1) // TILE_SIZE,
    )


def test_positions_outside_arena_are_rejected() -> None:
    assert ArenaViewer.screen_to_tile((-1, 0)) is None
    assert ArenaViewer.screen_to_tile((0, -1)) is None
    assert ArenaViewer.screen_to_tile((450, 0)) is None
    assert ArenaViewer.screen_to_tile((0, SCREEN_HEIGHT)) is None


def test_arena_height_matches_all_grid_rows() -> None:
    assert SCREEN_HEIGHT == 800


def test_each_team_has_two_princess_towers_and_one_king_tower() -> None:
    for team in ("red", "blue"):
        team_towers = [tower for tower in TOWERS if tower.team == team]

        assert [tower.kind for tower in team_towers].count("princess") == 2
        assert [tower.kind for tower in team_towers].count("king") == 1


def test_tower_layout_is_mirrored_across_the_arena() -> None:
    red_towers = {(tower.kind, tower.center) for tower in TOWERS if tower.team == "red"}
    mirrored_blue_towers = {
        (tower.kind, (tower.center[0], SCREEN_HEIGHT - tower.center[1]))
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


def test_river_is_centered_vertically() -> None:
    river = ArenaViewer.river_rectangle()

    assert river.centery == SCREEN_HEIGHT // 2


def test_match_duration_is_three_minutes() -> None:
    assert MATCH_DURATION_SECONDS == 180


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


def test_elixir_generation_accounts_for_rate_boundary_in_same_tick() -> None:
    meter = ElixirMeter(amount=0)

    meter.update(2.0, match_elapsed=119.0)

    assert meter.amount == pytest.approx(3 / 2.8)


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
