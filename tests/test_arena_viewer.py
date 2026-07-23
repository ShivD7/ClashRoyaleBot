from arena_viewer import (
    MATCH_DURATION_SECONDS,
    ArenaViewer,
    SCREEN_HEIGHT,
    TOWERS,
    format_match_time,
    remaining_match_seconds,
)


def test_screen_positions_convert_to_expected_tiles() -> None:
    assert ArenaViewer.screen_to_tile((0, 0)) == (0, 0)
    assert ArenaViewer.screen_to_tile((24, 24)) == (0, 0)
    assert ArenaViewer.screen_to_tile((25, 25)) == (1, 1)
    assert ArenaViewer.screen_to_tile((449, 799)) == (17, 31)


def test_positions_outside_arena_are_rejected() -> None:
    assert ArenaViewer.screen_to_tile((-1, 0)) is None
    assert ArenaViewer.screen_to_tile((0, -1)) is None
    assert ArenaViewer.screen_to_tile((450, 0)) is None
    assert ArenaViewer.screen_to_tile((0, 800)) is None


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
