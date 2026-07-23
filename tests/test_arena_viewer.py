import pytest

from arena_viewer import ArenaViewer, ElixirMeter, SCREEN_HEIGHT, TOWERS


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
