import pytest

from arena_viewer import (
    ARENA_HEIGHT,
    DEFAULT_DECK,
    LEFT_LANE_X,
    RIGHT_LANE_X,
    RIVER_HEIGHT,
    RIVER_TOP,
    TILE_SIZE,
    TOWERS,
)
from battle_engine import BattleEngine, EntityState


def make_engine() -> BattleEngine:
    """Create the same deterministic level-11 engine used by the viewer."""
    return BattleEngine(
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


def card_named(name: str):
    return next(card for card in DEFAULT_DECK if card.name == name)


def test_giant_ignores_troops_and_locks_nearest_enemy_building() -> None:
    engine = make_engine()
    giant = engine.deploy_card(
        card_named("Giant"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    engine.deploy_card(
        card_named("Knight"),
        "red",
        (LEFT_LANE_X, 500),
    )

    engine.update(0.01)
    target = engine.entity_by_id(giant.target_id)

    assert target is not None
    assert target.is_building
    assert target.team == "red"
    assert target.tower_kind == "princess"


def test_moving_unit_switches_to_closer_enemy_inside_sight_range() -> None:
    engine = make_engine()
    blue_knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    first_enemy = engine.deploy_card(
        card_named("Knight"),
        "red",
        (LEFT_LANE_X, 400),
    )[0]
    second_enemy = engine.deploy_card(
        card_named("Knight"),
        "red",
        (RIGHT_LANE_X, 550),
    )[0]

    engine.update(0.01)
    assert blue_knight.target_id == first_enemy.entity_id
    assert blue_knight.state is EntityState.MOVING

    second_enemy.position = blue_knight.position + (0, -30)
    engine.update(0.01)

    assert blue_knight.target_id == second_enemy.entity_id


def test_attacking_unit_keeps_target_when_another_enemy_becomes_closer() -> None:
    engine = make_engine()
    blue_knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    first_enemy = engine.deploy_card(
        card_named("Knight"),
        "red",
        (LEFT_LANE_X, 510),
    )[0]
    second_enemy = engine.deploy_card(
        card_named("Knight"),
        "red",
        (RIGHT_LANE_X, 550),
    )[0]

    engine.update(0.01)
    assert blue_knight.target_id == first_enemy.entity_id
    assert blue_knight.state is EntityState.ATTACKING

    second_enemy.position = blue_knight.position.copy()
    engine.update(0.01)

    assert blue_knight.target_id == first_enemy.entity_id


def test_unit_retargets_after_locked_target_dies() -> None:
    engine = make_engine()
    attacker = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    first_target = engine.deploy_card(
        card_named("Knight"),
        "red",
        (LEFT_LANE_X, 500),
    )[0]
    second_target = engine.deploy_card(
        card_named("Knight"),
        "red",
        (LEFT_LANE_X, 450),
    )[0]

    engine.update(0.01)
    assert attacker.target_id == first_target.entity_id

    first_target.take_damage(first_target.max_health)
    engine.update(0.01)

    assert attacker.target_id == second_target.entity_id
    assert attacker.state is not EntityState.DEAD


def test_enemy_outside_sight_range_does_not_distract_walking_unit() -> None:
    engine = make_engine()
    blue_knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 600),
    )[0]
    engine.deploy_card(
        card_named("Knight"),
        "red",
        (LEFT_LANE_X, 400),
    )

    engine.update(0.01)
    target = engine.entity_by_id(blue_knight.target_id)

    assert target is not None
    assert target.tower_kind == "princess"


def test_fast_unit_moves_farther_than_slow_unit_in_same_time() -> None:
    slow_engine = make_engine()
    fast_engine = make_engine()
    giant = slow_engine.deploy_card(
        card_named("Giant"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    mini_pekka = fast_engine.deploy_card(
        card_named("Mini P.E.K.K.A"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    giant_start = giant.position.copy()
    mini_pekka_start = mini_pekka.position.copy()

    slow_engine.update(1.0)
    fast_engine.update(1.0)

    assert giant_start.distance_to(giant.position) == pytest.approx(18.75)
    assert mini_pekka_start.distance_to(mini_pekka.position) == pytest.approx(37.5)


def test_melee_damage_sets_health_to_zero_and_marks_target_dead() -> None:
    engine = make_engine()
    mini_pekka = engine.deploy_card(
        card_named("Mini P.E.K.K.A"),
        "blue",
        (LEFT_LANE_X, 500),
    )[0]
    skeletons = engine.deploy_card(
        card_named("Skeletons"),
        "red",
        (LEFT_LANE_X, 490),
    )

    engine.update(0.01)
    locked_skeleton = engine.entity_by_id(mini_pekka.target_id)

    assert mini_pekka.state is EntityState.ATTACKING
    assert locked_skeleton in skeletons
    assert locked_skeleton.health == 0
    assert locked_skeleton.state is EntityState.DEAD


def test_princess_tower_fires_locked_projectile_at_nearest_troop() -> None:
    engine = make_engine()
    knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 300),
    )[0]
    red_left_princess = next(
        entity
        for entity in engine.entities
        if entity.team == "red"
        and entity.tower_kind == "princess"
        and entity.position.x == LEFT_LANE_X
    )
    starting_health = knight.health

    engine.update(0.01)

    assert red_left_princess.target_id == knight.entity_id
    assert len(engine.projectiles) == 1
    assert engine.projectiles[0].target_id == knight.entity_id

    engine.update(0.3)

    assert knight.health == starting_health - red_left_princess.damage


def test_king_tower_activates_after_allied_princess_tower_dies() -> None:
    engine = make_engine()
    red_king = next(
        entity
        for entity in engine.entities
        if entity.team == "red" and entity.tower_kind == "king"
    )
    red_princess = next(
        entity
        for entity in engine.entities
        if entity.team == "red" and entity.tower_kind == "princess"
    )

    assert not red_king.active

    red_princess.take_damage(red_princess.max_health)
    engine.update(0.01)

    assert red_king.active


def test_destroyed_king_tower_finishes_and_freezes_the_battle() -> None:
    engine = make_engine()
    knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    red_king = next(
        entity
        for entity in engine.entities
        if entity.team == "red" and entity.tower_kind == "king"
    )
    red_king.take_damage(red_king.max_health)
    position_at_match_end = knight.position.copy()

    engine.update(1.0)

    assert engine.winning_team == "blue"
    assert knight.position == position_at_match_end
    assert engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 500),
    ) == ()


def test_crown_scores_track_destroyed_towers_for_both_teams() -> None:
    engine = make_engine()
    red_princesses = [
        entity
        for entity in engine.entities
        if entity.team == "red" and entity.tower_kind == "princess"
    ]
    blue_princess = next(
        entity
        for entity in engine.entities
        if entity.team == "blue" and entity.tower_kind == "princess"
    )

    assert engine.crown_scores == {"red": 0, "blue": 0}

    red_princesses[0].take_damage(red_princesses[0].max_health)
    blue_princess.take_damage(blue_princess.max_health)
    assert engine.crown_scores == {"red": 1, "blue": 1}

    red_princesses[1].take_damage(red_princesses[1].max_health)
    assert engine.crown_scores == {"red": 1, "blue": 2}


def test_destroying_king_tower_completes_score_at_three_crowns() -> None:
    engine = make_engine()
    red_king = next(
        entity
        for entity in engine.entities
        if entity.team == "red" and entity.tower_kind == "king"
    )

    red_king.take_damage(red_king.max_health)

    assert engine.crowns_for("blue") == 3


def test_fireball_uses_reduced_damage_against_crown_towers() -> None:
    engine = make_engine()
    red_princess = next(
        entity
        for entity in engine.entities
        if entity.team == "red"
        and entity.tower_kind == "princess"
        and entity.position.x == LEFT_LANE_X
    )
    starting_health = red_princess.health

    engine.deploy_card(
        card_named("Fireball"),
        "blue",
        (round(red_princess.position.x), round(red_princess.position.y)),
    )

    assert red_princess.health == starting_health - 207


def test_every_default_card_has_complete_combat_stats() -> None:
    for card in DEFAULT_DECK:
        if card.spell_stats is not None:
            assert card.spell_stats.damage > 0
            assert card.spell_stats.radius > 0
        else:
            assert card.unit_stats is not None
            assert card.unit_stats.max_health > 0
            assert card.unit_stats.damage > 0
            assert card.unit_stats.hit_speed > 0
            assert card.unit_stats.movement_speed > 0
