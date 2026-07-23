from arena_viewer import ArenaViewer


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
