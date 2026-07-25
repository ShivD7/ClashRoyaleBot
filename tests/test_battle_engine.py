import pytest

from arena_viewer import (
    ARENA_HEIGHT,
    CARD_CATALOG,
    LEFT_LANE_X,
    RIGHT_LANE_X,
    RIVER_HEIGHT,
    RIVER_TOP,
    TILE_SIZE,
    TOWERS,
    ArenaViewer,
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
    return next(card for card in CARD_CATALOG if card.name == name)


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


def test_ground_only_troop_cannot_target_flying_minions() -> None:
    engine = make_engine()
    knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    minions = engine.deploy_card(
        card_named("Minions"),
        "red",
        (LEFT_LANE_X, 530),
    )

    engine.update(0.01)
    target = engine.entity_by_id(knight.target_id)

    assert target is not None
    assert target not in minions
    assert target.tower_kind == "princess"


def test_air_capable_archers_can_target_flying_minions() -> None:
    engine = make_engine()
    archer = engine.deploy_card(
        card_named("Archers"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    minions = engine.deploy_card(
        card_named("Minions"),
        "red",
        (LEFT_LANE_X, 530),
    )

    engine.update(0.01)

    assert engine.entity_by_id(archer.target_id) in minions


def test_cannon_is_stationary_and_targets_ground_but_not_air() -> None:
    engine = make_engine()
    cannon = engine.deploy_card(
        card_named("Cannon"),
        "blue",
        (LEFT_LANE_X, 520),
    )[0]
    engine.deploy_card(
        card_named("Minions"),
        "red",
        (LEFT_LANE_X, 500),
    )
    knight = engine.deploy_card(
        card_named("Knight"),
        "red",
        (LEFT_LANE_X, 480),
    )[0]
    starting_position = cannon.position.copy()

    engine.update(0.01)

    assert cannon.is_building
    assert cannon.tower_kind is None
    assert cannon.position == starting_position
    assert cannon.target_id == knight.entity_id


def test_cannon_loses_health_and_expires_after_thirty_seconds() -> None:
    engine = make_engine()
    cannon = engine.deploy_card(
        card_named("Cannon"),
        "blue",
        (LEFT_LANE_X, 520),
    )[0]

    assert cannon.lifetime_seconds == 30.0
    assert cannon.lifetime_elapsed == 0.0

    engine.update(15.0)

    assert cannon.health == pytest.approx(cannon.max_health / 2)
    assert cannon.lifetime_elapsed == pytest.approx(15.0)
    assert cannon.is_alive

    engine.update(14.95)
    assert cannon.is_alive

    engine.update(0.05)

    assert cannon.health == 0
    assert cannon.lifetime_elapsed == pytest.approx(30.0)
    assert cannon.state is EntityState.DEAD


def test_tombstone_spawns_two_skeletons_every_three_point_three_seconds() -> None:
    engine = make_engine()
    tombstone = engine.deploy_card(
        card_named("Tombstone"),
        "blue",
        (LEFT_LANE_X, 520),
    )[0]

    engine.update(3.25)
    assert not any(entity.name.startswith("Skeleton ") for entity in engine.entities)

    engine.update(0.05)
    first_wave = [
        entity
        for entity in engine.entities
        if entity.name.startswith("Skeleton ")
    ]
    assert len(first_wave) == 2
    assert all(entity.team == "blue" for entity in first_wave)
    assert all(not entity.is_building for entity in first_wave)

    engine.update(3.3)
    all_spawned = [
        entity
        for entity in engine.entities
        if entity.name.startswith("Skeleton ")
    ]
    assert len(all_spawned) == 4
    assert tombstone.is_alive


def test_destroyed_tombstone_stops_spawning_waves() -> None:
    engine = make_engine()
    tombstone = engine.deploy_card(
        card_named("Tombstone"),
        "blue",
        (LEFT_LANE_X, 520),
    )[0]
    tombstone.take_damage(tombstone.max_health)

    engine.update(10.0)

    assert not any(entity.name.startswith("Skeleton ") for entity in engine.entities)


def test_combat_damage_makes_building_expire_before_full_lifetime() -> None:
    engine = make_engine()
    cannon = engine.deploy_card(
        card_named("Cannon"),
        "blue",
        (LEFT_LANE_X, 520),
    )[0]

    cannon.take_damage(100)
    engine.update(27.0)

    assert not cannon.is_alive
    assert cannon.lifetime_elapsed < cannon.lifetime_seconds


def test_crown_towers_do_not_lose_health_over_time() -> None:
    engine = make_engine()
    starting_health = {
        entity.entity_id: entity.health
        for entity in engine.entities
        if entity.tower_kind is not None
    }

    engine.update(30.0)

    assert {
        entity.entity_id: entity.health
        for entity in engine.entities
        if entity.tower_kind is not None
    } == starting_health


@pytest.mark.parametrize(
    "card_name",
    ("Knight", "Minions", "Cannon"),
)
def test_units_and_buildings_cannot_spawn_inside_crown_tower(
    card_name: str,
) -> None:
    engine = make_engine()
    blue_princess = next(
        entity
        for entity in engine.entities
        if entity.team == "blue" and entity.tower_kind == "princess"
    )

    assert not engine.can_deploy_card(
        card_named(card_name),
        blue_princess.position,
    )
    assert engine.can_deploy_card(
        card_named("Fireball"),
        blue_princess.position,
    )


def test_every_body_in_multi_unit_formation_must_clear_building() -> None:
    engine = make_engine()
    red_princess = next(
        entity
        for entity in engine.entities
        if entity.team == "red" and entity.tower_kind == "princess"
    )
    # The formation center clears the tower, but the left Archer spawns five
    # pixels closer and would overlap its physical footprint.
    placement = (
        red_princess.position.x + red_princess.radius + 11,
        red_princess.position.y,
    )

    assert not engine.can_deploy_card(card_named("Archers"), placement)


def test_building_cannot_stack_on_building_or_ground_troop() -> None:
    engine = make_engine()
    cannon = card_named("Cannon")
    open_position = (300, 500)

    assert engine.can_deploy_card(cannon, open_position)
    engine.deploy_card(cannon, "blue", open_position)
    assert not engine.can_deploy_card(cannon, open_position)

    troop_position = (400, 500)
    engine.deploy_card(card_named("Knight"), "blue", troop_position)
    assert not engine.can_deploy_card(cannon, troop_position)


def test_troops_may_share_spawn_area_and_movement_resolves_overlap() -> None:
    engine = make_engine()
    position = (300, 500)
    engine.deploy_card(card_named("Knight"), "blue", position)

    assert engine.can_deploy_card(card_named("Skeletons"), position)


def test_destroyed_building_no_longer_blocks_placement() -> None:
    engine = make_engine()
    red_princess = next(
        entity
        for entity in engine.entities
        if entity.team == "red" and entity.tower_kind == "princess"
    )
    red_princess.take_damage(red_princess.max_health)

    assert engine.can_deploy_card(
        card_named("Knight"),
        red_princess.position,
    )


def test_spawn_footprint_must_fit_completely_inside_arena() -> None:
    engine = make_engine()
    giant = card_named("Giant")
    radius = giant.unit_stats.body_radius

    assert not engine.can_deploy_card(
        giant,
        (engine.arena_left + radius - 0.1, 500),
    )
    assert engine.can_deploy_card(
        giant,
        (engine.arena_left + radius, 500),
    )
def test_flying_units_cross_the_river_without_using_a_bridge() -> None:
    engine = make_engine()
    minion = engine.deploy_card(
        card_named("Minions"),
        "blue",
        (300, 550),
    )[0]
    giant = engine.deploy_card(
        card_named("Giant"),
        "blue",
        (300, 550),
    )[0]
    red_king = next(
        entity
        for entity in engine.entities
        if entity.team == "red" and entity.tower_kind == "king"
    )

    assert engine._movement_destination(minion, red_king) == red_king.position
    assert (
        engine._movement_destination(giant, red_king).x
        in engine.bridge_x_positions
    )


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


@pytest.mark.parametrize(
    ("team", "start_tile"),
    (
        ("blue", (0, 17)),
        ("red", (0, 14)),
    ),
)
def test_hog_rider_at_far_left_approaches_bridge_without_teleporting(
    team: str,
    start_tile: tuple[int, int],
) -> None:
    engine = make_engine()
    hog = engine.deploy_card(
        card_named("Hog Rider"),
        team,
        ArenaViewer.tile_rectangle(start_tile).center,
    )[0]
    timestep = 0.05
    starting_x = hog.position.x

    for _ in range(20):
        previous_position = hog.position.copy()
        engine.update(timestep)

        assert previous_position.distance_to(hog.position) <= (
            hog.movement_speed * timestep + 1e-6
        )

    assert hog.lane_x == LEFT_LANE_X
    assert abs(hog.position.x - hog.lane_x) < abs(starting_x - hog.lane_x)


def test_overlapping_ground_units_are_separated_by_body_radius() -> None:
    engine = make_engine()
    first = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    second = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    first.movement_speed = 0
    second.movement_speed = 0

    engine.update(0.01)

    assert first.position.distance_to(second.position) == pytest.approx(
        first.radius + second.radius,
    )


def test_light_unit_yields_more_than_heavy_unit_during_collision() -> None:
    engine = make_engine()
    giant = engine.deploy_card(
        card_named("Giant"),
        "blue",
        (LEFT_LANE_X, 500),
    )[0]
    skeletons = engine.deploy_card(
        card_named("Skeletons"),
        "blue",
        (LEFT_LANE_X, 500),
    )
    skeleton = skeletons[1]
    skeletons[0].position.x -= 100
    skeletons[2].position.x += 100
    giant.movement_speed = 0
    for small_unit in skeletons:
        small_unit.movement_speed = 0
    shared_start = giant.position.copy()

    engine.update(0.01)

    assert skeleton.position.distance_to(shared_start) > (
        giant.position.distance_to(shared_start)
    )
    assert giant.position.distance_to(skeleton.position) == pytest.approx(
        giant.radius + skeleton.radius,
    )


@pytest.mark.parametrize(
    ("blue_start_y", "red_start_y"),
    (
        (350, 250),
        (470, 330),
    ),
)
def test_opposing_giants_slide_past_instead_of_deadlocking(
    blue_start_y: int,
    red_start_y: int,
) -> None:
    engine = make_engine()
    for entity in engine.entities:
        if entity.is_building:
            entity.damage = 0
    blue_giant = engine.deploy_card(
        card_named("Giant"),
        "blue",
        (LEFT_LANE_X, blue_start_y),
    )[0]
    red_giant = engine.deploy_card(
        card_named("Giant"),
        "red",
        (LEFT_LANE_X, red_start_y),
    )[0]
    maximum_lateral_separation = 0.0

    for _ in range(160):
        engine.update(0.05)
        maximum_lateral_separation = max(
            maximum_lateral_separation,
            abs(blue_giant.position.x - red_giant.position.x),
        )

    assert blue_giant.target_id != red_giant.entity_id
    assert red_giant.target_id != blue_giant.entity_id
    assert maximum_lateral_separation > 5
    assert blue_giant.position.y < red_giant.position.y


def test_opposing_troops_that_target_each_other_stop_and_fight() -> None:
    engine = make_engine()
    blue_knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 500),
    )[0]
    red_knight = engine.deploy_card(
        card_named("Knight"),
        "red",
        (LEFT_LANE_X, 480),
    )[0]
    blue_start = blue_knight.position.copy()
    red_start = red_knight.position.copy()

    engine.update(0.01)

    assert blue_knight.target_id == red_knight.entity_id
    assert red_knight.target_id == blue_knight.entity_id
    assert blue_knight.state is EntityState.ATTACKING
    assert red_knight.state is EntityState.ATTACKING
    assert blue_knight.position == blue_start
    assert red_knight.position == red_start


def test_ground_and_air_units_can_share_the_same_position() -> None:
    engine = make_engine()
    knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 500),
    )[0]
    minions = engine.deploy_card(
        card_named("Minions"),
        "blue",
        (LEFT_LANE_X, 500),
    )
    minion = minions[1]
    minions[0].position.x -= 100
    minions[2].position.x += 100
    knight.movement_speed = 0
    for flying_unit in minions:
        flying_unit.movement_speed = 0

    engine.update(0.01)

    assert knight.position == minion.position


def test_troop_steers_around_stationary_building() -> None:
    engine = make_engine()
    cannon = engine.deploy_card(
        card_named("Cannon"),
        "blue",
        (LEFT_LANE_X, 500),
    )[0]
    knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 560),
    )[0]

    for _ in range(80):
        engine.update(0.05)

    assert cannon.position == (LEFT_LANE_X, 500)
    assert abs(knight.position.x - LEFT_LANE_X) > 5
    assert knight.position.distance_to(cannon.position) >= (
        knight.radius + cannon.radius
    )


def test_selected_bridge_lane_remains_stable_while_crossing() -> None:
    engine = make_engine()
    giant = engine.deploy_card(
        card_named("Giant"),
        "blue",
        (LEFT_LANE_X, 550),
    )[0]
    red_left_tower = next(
        entity
        for entity in engine.entities
        if entity.team == "red"
        and entity.tower_kind == "princess"
        and entity.position.x == LEFT_LANE_X
    )

    first_destination = engine._movement_destination(giant, red_left_tower)
    chosen_lane = giant.lane_x
    giant.position.x = RIGHT_LANE_X
    second_destination = engine._movement_destination(giant, red_left_tower)

    assert chosen_lane == LEFT_LANE_X
    assert first_destination.x == chosen_lane
    assert second_destination.x == chosen_lane


@pytest.mark.parametrize(
    ("team", "start_y", "crossed_bank"),
    (
        ("blue", RIVER_TOP + TILE_SIZE // 2, "top"),
        (
            "red",
            RIVER_TOP + RIVER_HEIGHT - TILE_SIZE // 2,
            "bottom",
        ),
    ),
)
@pytest.mark.parametrize("lane_x", (LEFT_LANE_X, RIGHT_LANE_X))
def test_troop_deployed_on_bridge_walks_off_the_opposite_end(
    team: str,
    start_y: int,
    crossed_bank: str,
    lane_x: int,
) -> None:
    engine = make_engine()
    for entity in engine.entities:
        if entity.is_building:
            entity.damage = 0
    knight = engine.deploy_card(
        card_named("Knight"),
        team,
        (lane_x, start_y),
    )[0]

    for _ in range(80):
        engine.update(0.05)

    if crossed_bank == "top":
        assert knight.position.y + knight.radius < RIVER_TOP
    else:
        assert knight.position.y - knight.radius > RIVER_TOP + RIVER_HEIGHT


def test_bridge_congestion_keeps_units_from_stacking() -> None:
    engine = make_engine()
    for entity in engine.entities:
        if entity.is_building:
            entity.damage = 0
    units = [
        engine.deploy_card(
            card_named("Knight"),
            "blue",
            (LEFT_LANE_X, 600),
        )[0]
        for _ in range(5)
    ]

    for _ in range(400):
        engine.update(0.05)

    assert all(unit.position.y < RIVER_TOP for unit in units)
    for index, first in enumerate(units):
        for second in units[index + 1:]:
            assert first.position.distance_to(second.position) >= (
                first.radius + second.radius - 1e-6
            )


@pytest.mark.parametrize("column", (4, 13))
@pytest.mark.parametrize(
    ("team", "building_row", "troop_rows", "crossed_bank"),
    (
        ("blue", 17, (19, 20, 21), "top"),
        ("red", 14, (12, 11, 10), "bottom"),
    ),
)
def test_troops_route_around_building_on_last_tile_before_bridge(
    column: int,
    team: str,
    building_row: int,
    troop_rows: tuple[int, ...],
    crossed_bank: str,
) -> None:
    engine = make_engine()
    for entity in engine.entities:
        if entity.is_building:
            entity.damage = 0
    engine.deploy_card(
        card_named("Cannon"),
        team,
        ArenaViewer.tile_rectangle((column, building_row)).center,
    )
    troops = [
        engine.deploy_card(
            card_named("Knight"),
            team,
            ArenaViewer.tile_rectangle((column, row)).center,
        )[0]
        for row in troop_rows
    ]

    for _ in range(400):
        engine.update(0.05)

    if crossed_bank == "top":
        assert all(troop.position.y < RIVER_TOP for troop in troops)
    else:
        river_bottom = RIVER_TOP + RIVER_HEIGHT
        assert all(troop.position.y > river_bottom for troop in troops)


def test_knockback_interrupts_attack_and_respects_unit_resistance() -> None:
    engine = make_engine()
    knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 500),
    )[0]
    giant = engine.deploy_card(
        card_named("Giant"),
        "blue",
        (RIGHT_LANE_X, 500),
    )[0]
    knight_start = knight.position.copy()
    giant_start = giant.position.copy()

    assert engine.apply_knockback(
        knight.entity_id,
        (knight.position.x, knight.position.y + 20),
        distance_tiles=2,
    )
    assert engine.apply_knockback(
        giant.entity_id,
        (giant.position.x, giant.position.y + 20),
        distance_tiles=2,
    )
    engine.update(0.25)

    assert knight.target_id is None
    assert knight.position.distance_to(knight_start) > (
        giant.position.distance_to(giant_start)
    )


def test_ground_knockback_cannot_push_a_unit_across_open_water() -> None:
    engine = make_engine()
    river_bottom = RIVER_TOP + RIVER_HEIGHT
    between_bridges = (LEFT_LANE_X + RIGHT_LANE_X) / 2
    knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (between_bridges, river_bottom + 11),
    )[0]
    knight.movement_speed = 0

    engine.apply_knockback(
        knight.entity_id,
        (knight.position.x, knight.position.y + 20),
        distance_tiles=3,
    )
    engine.update(0.25)

    assert knight.position.y == pytest.approx(river_bottom + knight.radius)


def test_buildings_do_not_move_when_resolving_collisions() -> None:
    engine = make_engine()
    cannon = engine.deploy_card(
        card_named("Cannon"),
        "blue",
        (LEFT_LANE_X, 500),
    )[0]
    knight = engine.deploy_card(
        card_named("Knight"),
        "blue",
        (LEFT_LANE_X, 500),
    )[0]
    cannon_start = cannon.position.copy()
    knight.movement_speed = 0

    engine.update(0.01)

    assert cannon.position == cannon_start
    assert cannon.position.distance_to(knight.position) == pytest.approx(
        cannon.radius + knight.radius,
    )


def test_movement_and_collision_results_are_deterministic() -> None:
    def run_scenario() -> tuple[tuple[float, float], ...]:
        engine = make_engine()
        for _ in range(4):
            engine.deploy_card(
                card_named("Knight"),
                "blue",
                (LEFT_LANE_X, 600),
            )
        engine.deploy_card(
            card_named("Cannon"),
            "blue",
            (LEFT_LANE_X, 500),
        )
        for _ in range(80):
            engine.update(0.05)
        return tuple(
            (entity.position.x, entity.position.y)
            for entity in engine.entities
        )

    assert run_scenario() == run_scenario()


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


def test_skeleton_army_spawns_fifteen_units_in_scatter_formation() -> None:
    engine = make_engine()

    army = engine.deploy_card(
        card_named("Skeleton Army"),
        "blue",
        (LEFT_LANE_X, 500),
    )
    positions = {
        (round(skeleton.position.x, 4), round(skeleton.position.y, 4))
        for skeleton in army
    }

    assert len(army) == 15
    assert len(positions) == 15
    assert len({position[1] for position in positions}) == 3
    minimum_separation = min(
        first.position.distance_to(second.position)
        for index, first in enumerate(army)
        for second in army[index + 1 :]
    )
    ordinary_group_spacing = min(12.0, TILE_SIZE * 0.4)
    assert minimum_separation == pytest.approx(
        ordinary_group_spacing * 1.25,
    )


def test_wizard_projectile_splashes_clustered_skeleton_army() -> None:
    engine = make_engine()
    for tower in engine.entities:
        tower.damage = 0
    wizard = engine.deploy_card(
        card_named("Wizard"),
        "blue",
        (LEFT_LANE_X, 500),
    )[0]
    army = engine.deploy_card(
        card_named("Skeleton Army"),
        "red",
        (LEFT_LANE_X, 400),
    )
    wizard.movement_speed = 0
    for skeleton in army:
        skeleton.movement_speed = 0

    engine.update(0.01)
    engine.update(0.5)

    assert not any(skeleton.is_alive for skeleton in army)


def test_wizard_splash_uses_impact_radius_without_hitting_distant_enemy() -> None:
    engine = make_engine()
    for tower in engine.entities:
        tower.damage = 0
    wizard = engine.deploy_card(
        card_named("Wizard"),
        "blue",
        (100, 500),
    )[0]
    primary = engine.deploy_card(
        card_named("Knight"),
        "red",
        (200, 500),
    )[0]
    nearby = engine.deploy_card(
        card_named("Knight"),
        "red",
        (244, 500),
    )[0]
    distant = engine.deploy_card(
        card_named("Knight"),
        "red",
        (280, 500),
    )[0]
    for entity in (wizard, primary, nearby, distant):
        entity.movement_speed = 0

    engine.update(0.01)
    engine.update(0.5)

    assert primary.health == primary.max_health - wizard.damage
    assert nearby.health == nearby.max_health - wizard.damage
    assert distant.health == distant.max_health


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


@pytest.mark.parametrize(
    ("king_team", "spell_team"),
    (("red", "blue"), ("blue", "red")),
)
def test_spell_damage_activates_king_tower_early(
    king_team: str,
    spell_team: str,
) -> None:
    engine = make_engine()
    king = next(
        entity
        for entity in engine.entities
        if entity.team == king_team and entity.tower_kind == "king"
    )
    allied_princesses = [
        entity
        for entity in engine.entities
        if entity.team == king_team and entity.tower_kind == "princess"
    ]
    starting_health = king.health

    assert not king.active
    assert all(tower.is_alive for tower in allied_princesses)

    engine.deploy_card(
        card_named("Fireball"),
        spell_team,
        (round(king.position.x), round(king.position.y)),
    )

    assert king.health < starting_health
    assert king.active
    assert all(tower.is_alive for tower in allied_princesses)


def test_spell_that_misses_king_tower_does_not_activate_it() -> None:
    engine = make_engine()
    red_king = next(
        entity
        for entity in engine.entities
        if entity.team == "red" and entity.tower_kind == "king"
    )

    engine.deploy_card(
        card_named("Fireball"),
        "blue",
        (round(red_king.position.x), round(red_king.position.y + 250)),
    )

    assert not red_king.active


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


def test_tiebreaker_drain_damages_all_living_crown_towers_equally() -> None:
    engine = make_engine()
    towers = engine.living_crown_towers
    starting_health = {
        tower.entity_id: tower.health
        for tower in towers
    }

    destroyed_teams = engine.drain_crown_towers(40)

    assert destroyed_teams == frozenset()
    assert all(
        tower.health == starting_health[tower.entity_id] - 40
        for tower in towers
    )


def test_tiebreaker_drain_stops_exactly_at_first_tower_destruction() -> None:
    engine = make_engine()
    red_princess = next(
        tower
        for tower in engine.living_crown_towers
        if tower.team == "red" and tower.tower_kind == "princess"
    )
    blue_princess = next(
        tower
        for tower in engine.living_crown_towers
        if tower.team == "blue" and tower.tower_kind == "princess"
    )
    red_princess.health = 25
    blue_starting_health = blue_princess.health

    destroyed_teams = engine.drain_crown_towers(100)

    assert destroyed_teams == frozenset({"red"})
    assert red_princess.health == 0
    assert blue_princess.health == blue_starting_health - 25


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


def test_fireball_knocks_surviving_troop_away_from_blast() -> None:
    engine = make_engine()
    knight = engine.deploy_card(
        card_named("Knight"),
        "red",
        (LEFT_LANE_X, 500),
    )[0]
    starting_position = knight.position.copy()
    blast_center = (LEFT_LANE_X - 20, 500)

    engine.deploy_card(
        card_named("Fireball"),
        "blue",
        blast_center,
    )
    assert knight.target_id is None

    engine.update(0.25)

    assert knight.position.x > starting_position.x
    assert knight.position.y == pytest.approx(starting_position.y)


def test_every_catalog_card_has_complete_combat_stats() -> None:
    for card in CARD_CATALOG:
        if card.spell_stats is not None:
            assert card.spell_stats.damage > 0
            assert card.spell_stats.radius > 0
        else:
            assert card.unit_stats is not None
            assert card.unit_stats.max_health > 0
            assert card.unit_stats.hit_speed > 0
            if card.unit_stats.spawner is None:
                assert card.unit_stats.damage > 0
            else:
                assert card.unit_stats.damage >= 0
                assert card.unit_stats.spawner.interval_seconds > 0
                assert card.unit_stats.spawner.card.unit_stats.damage > 0
            if card.card_type == "building":
                assert card.unit_stats.movement_speed == 0
                assert card.unit_stats.lifetime_seconds is not None
                assert card.unit_stats.lifetime_seconds > 0
            else:
                assert card.unit_stats.movement_speed > 0
                assert card.unit_stats.lifetime_seconds is None
            assert card.unit_stats.body_radius > 0
            assert card.unit_stats.mass > 0
            assert 0 <= card.unit_stats.knockback_resistance <= 1
