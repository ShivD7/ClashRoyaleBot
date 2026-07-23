from arena_viewer import ArenaViewer, SCREEN_HEIGHT, TOWERS


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
