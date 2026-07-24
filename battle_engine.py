"""Deterministic combat rules for the arena simulator.

The engine owns mutable battle state but knows nothing about buttons, fonts, or
card dragging. Keeping it separate from Pygame drawing lets tests advance a
battle by exact time steps and makes the same engine reusable by an RL agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import pygame


class EntityState(Enum):
    """The high-level action currently performed by a battle entity."""

    RETARGETING = "retargeting"
    MOVING = "moving"
    ATTACKING = "attacking"
    DEAD = "dead"


@dataclass(frozen=True)
class UnitStats:
    """Level-11 combat statistics shared by every unit spawned from a card."""

    max_health: int
    damage: int
    hit_speed: float
    movement_speed: float
    attack_range: float
    projectile_speed: float = 0.0
    sight_range: float = 5.5
    body_radius: float = 9.0
    mass: float = 1.0
    knockback_resistance: float = 0.0


@dataclass(frozen=True)
class SpellStats:
    """Level-11 direct spell damage and affected radius."""

    damage: int
    crown_tower_damage: int
    radius: float
    knockback_distance: float = 0.0


class CardLike(Protocol):
    """The card attributes required by the engine without importing the UI."""

    name: str
    card_type: str
    target_priority: str
    target_types: str
    movement_type: str
    attack_style: str
    unit_count: int
    unit_stats: UnitStats | None
    spell_stats: SpellStats | None


@dataclass
class BattleEntity:
    """A troop, deployable building, or Crown Tower participating in combat."""

    entity_id: int
    name: str
    team: str
    position: pygame.Vector2
    max_health: int
    health: float
    damage: int
    hit_speed: float
    movement_speed: float
    attack_range: float
    sight_range: float
    projectile_speed: float
    target_priority: str
    target_types: str
    movement_type: str
    attack_style: str
    radius: float
    mass: float
    knockback_resistance: float
    is_building: bool = False
    tower_kind: str | None = None
    active: bool = True
    state: EntityState = EntityState.RETARGETING
    target_id: int | None = None
    attack_cooldown: float = 0.0
    lane_x: float | None = None
    velocity: pygame.Vector2 = field(default_factory=pygame.Vector2)
    knockback_velocity: pygame.Vector2 = field(default_factory=pygame.Vector2)
    knockback_remaining: float = 0.0

    @property
    def is_alive(self) -> bool:
        """Return whether this entity can move, attack, or be targeted."""
        return self.state is not EntityState.DEAD and self.health > 0

    def take_damage(self, amount: float) -> None:
        """Apply damage once and enter the permanent dead state at zero health."""
        if not self.is_alive:
            return

        previous_health = self.health
        self.health = max(0.0, self.health - amount)
        # A sleeping King Tower wakes as soon as an enemy damages it. Keeping
        # this rule in the shared damage method covers spells, troops, and
        # projectiles without requiring separate activation checks for each.
        if self.tower_kind == "king" and self.health < previous_health:
            self.active = True

        if self.health == 0:
            self.state = EntityState.DEAD
            self.target_id = None


@dataclass
class Projectile:
    """A visible attack travelling toward one permanently locked target."""

    projectile_id: int
    source_id: int
    target_id: int
    team: str
    position: pygame.Vector2
    damage: int
    speed: float
    color: tuple[int, int, int]


class BattleEngine:
    """Advance targeting, movement, attacks, projectiles, health, and death."""

    PRINCESS_TOWER_HEALTH = 3052
    KING_TOWER_HEALTH = 4824
    TOWER_DAMAGE = 109
    PRINCESS_HIT_SPEED = 0.8
    KING_HIT_SPEED = 1.0
    PRINCESS_RANGE = 7.5
    KING_RANGE = 7.0
    TOWER_PROJECTILE_SPEED = 1000 / 60
    COLLISION_ITERATIONS = 4
    LOCAL_AVOIDANCE_TILES = 0.85
    BUILDING_AVOIDANCE_TILES = 1.4
    BRIDGE_HALF_WIDTH_TILES = 1.0
    BRIDGE_CONGESTION_PENALTY_TILES = 0.35

    def __init__(
        self,
        *,
        tile_size: int,
        screen_height: int,
        river_top: int,
        river_height: int,
        bridge_x_positions: tuple[int, int],
        tower_layout: tuple[tuple[str, str, tuple[int, int]], ...],
    ) -> None:
        self.tile_size = tile_size
        self.screen_height = screen_height
        self.river_top = river_top
        self.river_bottom = river_top + river_height
        self.bridge_x_positions = bridge_x_positions
        lane_span = abs(bridge_x_positions[1] - bridge_x_positions[0])
        self.arena_left = min(bridge_x_positions) - lane_span / 2
        self.arena_right = max(bridge_x_positions) + lane_span / 2
        self.entities: list[BattleEntity] = []
        self.projectiles: list[Projectile] = []
        self._next_entity_id = 1
        self._next_projectile_id = 1

        for tower_kind, team, center in tower_layout:
            self._create_tower(tower_kind, team, center)

    @property
    def living_entities(self) -> tuple[BattleEntity, ...]:
        """Return a stable snapshot of all currently targetable entities."""
        return tuple(entity for entity in self.entities if entity.is_alive)

    @property
    def living_crown_towers(self) -> tuple[BattleEntity, ...]:
        """Return surviving Princess and King Towers for match rules."""
        return tuple(
            entity
            for entity in self.entities
            if entity.is_alive and entity.tower_kind is not None
        )

    @property
    def winning_team(self) -> str | None:
        """Return the winner after a King Tower dies, otherwise ``None``."""
        destroyed_king = next(
            (
                entity
                for entity in self.entities
                if entity.tower_kind == "king" and not entity.is_alive
            ),
            None,
        )
        if destroyed_king is None:
            return None
        return "blue" if destroyed_king.team == "red" else "red"

    def crowns_for(self, team: str) -> int:
        """Return the team's live battle score from destroyed enemy towers.

        Each Princess Tower is worth one crown. Destroying the King Tower ends
        the battle and completes the attacker's score at three crowns, even if
        one or both Princess Towers were still standing.
        """
        if team not in {"red", "blue"}:
            raise ValueError("Team must be 'red' or 'blue'")

        enemy_team = "red" if team == "blue" else "blue"
        enemy_towers = [
            entity
            for entity in self.entities
            if entity.team == enemy_team and entity.is_building
        ]
        enemy_king = next(
            tower
            for tower in enemy_towers
            if tower.tower_kind == "king"
        )
        if not enemy_king.is_alive:
            return 3

        return sum(
            not tower.is_alive
            for tower in enemy_towers
            if tower.tower_kind == "princess"
        )

    @property
    def crown_scores(self) -> dict[str, int]:
        """Return both team scores for UI, observations, and rewards."""
        return {
            "red": self.crowns_for("red"),
            "blue": self.crowns_for("blue"),
        }

    def entity_by_id(self, entity_id: int | None) -> BattleEntity | None:
        """Find an entity by its stable ID."""
        if entity_id is None:
            return None

        return next(
            (entity for entity in self.entities if entity.entity_id == entity_id),
            None,
        )

    def _new_entity_id(self) -> int:
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        return entity_id

    def _create_tower(
        self,
        tower_kind: str,
        team: str,
        center: tuple[int, int],
    ) -> BattleEntity:
        """Create a level-11 Crown Tower from the immutable arena layout."""
        is_king = tower_kind == "king"
        entity = BattleEntity(
            entity_id=self._new_entity_id(),
            name=f"{team.title()} {tower_kind.title()} Tower",
            team=team,
            position=pygame.Vector2(center),
            max_health=(
                self.KING_TOWER_HEALTH
                if is_king
                else self.PRINCESS_TOWER_HEALTH
            ),
            health=float(
                self.KING_TOWER_HEALTH
                if is_king
                else self.PRINCESS_TOWER_HEALTH
            ),
            damage=self.TOWER_DAMAGE,
            hit_speed=(
                self.KING_HIT_SPEED
                if is_king
                else self.PRINCESS_HIT_SPEED
            ),
            movement_speed=0.0,
            attack_range=self.KING_RANGE if is_king else self.PRINCESS_RANGE,
            sight_range=self.KING_RANGE if is_king else self.PRINCESS_RANGE,
            projectile_speed=self.TOWER_PROJECTILE_SPEED,
            target_priority="nearest_enemy",
            target_types="air_and_ground",
            movement_type="ground",
            attack_style="ranged",
            radius=30.0 if is_king else 23.0,
            mass=float("inf"),
            knockback_resistance=1.0,
            is_building=True,
            tower_kind=tower_kind,
            # King Towers wake after taking damage or losing an allied Princess
            # Tower. Princess Towers begin active.
            active=not is_king,
        )
        self.entities.append(entity)
        return entity

    def deploy_card(
        self,
        card: CardLike,
        team: str,
        position: tuple[int, int],
    ) -> tuple[BattleEntity, ...]:
        """Spawn all units from a troop card or resolve a spell immediately."""
        # A destroyed King Tower is a terminal state. Training code and the UI
        # cannot accidentally add actions to a battle that has already ended.
        if self.winning_team is not None:
            return ()

        if card.card_type == "spell":
            self.cast_spell(card, team, position)
            return ()

        if card.unit_stats is None:
            raise ValueError(
                f"{card.card_type.title()} card {card.name} "
                "has no unit statistics",
            )

        spawned = []
        spacing = min(12.0, self.tile_size * 0.4)
        middle = (card.unit_count - 1) / 2

        for index in range(card.unit_count):
            offset_x = (index - middle) * spacing
            spawned.append(
                self._spawn_unit(
                    card,
                    team,
                    (position[0] + offset_x, position[1]),
                    index,
                )
            )

        return tuple(spawned)

    def _spawn_unit(
        self,
        card: CardLike,
        team: str,
        position: tuple[float, float],
        index: int,
    ) -> BattleEntity:
        """Create one runtime troop or building from immutable card statistics."""
        stats = card.unit_stats
        if stats is None:
            raise ValueError(f"Troop card {card.name} has no unit statistics")

        suffix = f" {index + 1}" if card.unit_count > 1 else ""
        entity = BattleEntity(
            entity_id=self._new_entity_id(),
            name=f"{card.name}{suffix}",
            team=team,
            position=pygame.Vector2(position),
            max_health=stats.max_health,
            health=float(stats.max_health),
            damage=stats.damage,
            hit_speed=stats.hit_speed,
            # Speeds in the source data use 60 as Medium, or one tile/second.
            movement_speed=stats.movement_speed * self.tile_size,
            attack_range=stats.attack_range,
            sight_range=stats.sight_range,
            projectile_speed=stats.projectile_speed,
            target_priority=card.target_priority,
            target_types=card.target_types,
            movement_type=card.movement_type,
            attack_style=card.attack_style,
            radius=stats.body_radius,
            mass=float("inf") if card.card_type == "building" else stats.mass,
            knockback_resistance=(
                1.0
                if card.card_type == "building"
                else stats.knockback_resistance
            ),
            is_building=card.card_type == "building",
        )
        self.entities.append(entity)
        return entity

    def cast_spell(
        self,
        card: CardLike,
        team: str,
        position: tuple[int, int],
    ) -> None:
        """Apply consistent area damage at a chosen arena position."""
        stats = card.spell_stats
        if stats is None:
            raise ValueError(f"Spell card {card.name} has no spell statistics")

        center = pygame.Vector2(position)
        radius_pixels = stats.radius * self.tile_size

        for target in self.living_entities:
            if target.team == team:
                continue
            if center.distance_to(target.position) > radius_pixels + target.radius:
                continue

            damage = (
                stats.crown_tower_damage
                if target.is_building and target.tower_kind is not None
                else stats.damage
            )
            target.take_damage(damage)
            if stats.knockback_distance > 0 and target.is_alive:
                self.apply_knockback(
                    target.entity_id,
                    center,
                    distance_tiles=stats.knockback_distance,
                )

        self._activate_king_towers()

    def drain_crown_towers(self, requested_damage: float) -> frozenset[str]:
        """Damage every standing Crown Tower equally for a tiebreaker tick.

        Damage is capped at the lowest remaining tower health. This prevents a
        large simulation step from skipping over the true first destruction.
        The returned teams are exactly those that lost a tower on this tick.
        """
        if requested_damage <= 0:
            return frozenset()

        towers = self.living_crown_towers
        if not towers:
            return frozenset()

        damage = min(requested_damage, min(tower.health for tower in towers))
        destroyed_teams = {
            tower.team
            for tower in towers
            if tower.health <= damage
        }
        for tower in towers:
            tower.take_damage(damage)

        self._activate_king_towers()
        return frozenset(destroyed_teams)

    def update(self, delta_seconds: float) -> None:
        """Advance a deterministic slice of battle time."""
        if delta_seconds <= 0 or self.winning_team is not None:
            return

        self._activate_king_towers()
        movement_displacements: dict[int, pygame.Vector2] = {}

        # Decide every entity's action before changing any position. Movement
        # therefore uses a shared snapshot instead of depending on spawn order.
        for entity in sorted(self.entities, key=lambda item: item.entity_id):
            if not entity.is_alive or not entity.active:
                continue

            entity.velocity.update(0, 0)
            entity.attack_cooldown = max(
                0.0,
                entity.attack_cooldown - delta_seconds,
            )

            if entity.knockback_remaining > 0 and not entity.is_building:
                forced_time = min(delta_seconds, entity.knockback_remaining)
                movement_displacements[entity.entity_id] = (
                    entity.knockback_velocity * forced_time
                )
                entity.velocity = entity.knockback_velocity.copy()
                entity.knockback_remaining -= forced_time
                if entity.knockback_remaining <= 0:
                    entity.knockback_remaining = 0.0
                    entity.knockback_velocity.update(0, 0)
                entity.state = EntityState.MOVING
                continue

            target = self.entity_by_id(entity.target_id)

            if target is None or not target.is_alive:
                entity.target_id = None
                entity.state = EntityState.RETARGETING
                target = self._acquire_target(entity)
                if target is not None:
                    entity.target_id = target.entity_id
            elif (
                not entity.is_building
                and entity.state is not EntityState.ATTACKING
            ):
                # A walking troop may be distracted by a closer eligible enemy
                # inside its sight range. Once an attack begins, the target lock
                # remains until that target dies or leaves the engagement.
                closer_target = self._closer_visible_target(entity, target)
                if closer_target is not None:
                    target = closer_target
                    entity.target_id = target.entity_id

            if target is None:
                continue

            if self._is_in_attack_range(entity, target):
                entity.state = EntityState.ATTACKING
                self._attack_if_ready(entity, target)
                if self.winning_team is not None:
                    break
            elif entity.is_building:
                # Towers cannot move and keep their lock while the target lives.
                entity.state = EntityState.ATTACKING
            else:
                entity.state = EntityState.MOVING
                velocity = self._desired_velocity(
                    entity,
                    target,
                    delta_seconds,
                )
                entity.velocity = velocity
                movement_displacements[entity.entity_id] = (
                    velocity * delta_seconds
                )

        if self.winning_team is None:
            self._apply_movement(movement_displacements)
            self._update_projectiles(delta_seconds)
        self._activate_king_towers()

    def _acquire_target(self, entity: BattleEntity) -> BattleEntity | None:
        """Choose a visible enemy, falling back to the nearest Crown Tower."""
        visible_candidates = []
        crown_tower_candidates = []

        for candidate in self.living_entities:
            if not self._is_eligible_target(entity, candidate):
                continue

            distance = entity.position.distance_to(candidate.position)
            if entity.is_building:
                maximum = entity.attack_range * self.tile_size
                maximum += entity.radius + candidate.radius
                if distance > maximum:
                    continue
                visible_candidates.append(
                    (distance, candidate.entity_id, candidate)
                )
                continue

            sight_distance = entity.sight_range * self.tile_size
            sight_distance += entity.radius + candidate.radius
            if distance <= sight_distance:
                visible_candidates.append(
                    (distance, candidate.entity_id, candidate)
                )

            if candidate.tower_kind is not None:
                crown_tower_candidates.append(
                    (distance, candidate.entity_id, candidate)
                )

        # Entity ID breaks equal-distance ties deterministically.
        if visible_candidates:
            return min(
                visible_candidates,
                key=lambda item: (item[0], item[1]),
            )[2]
        if crown_tower_candidates:
            return min(
                crown_tower_candidates,
                key=lambda item: (item[0], item[1]),
            )[2]
        return None

    @staticmethod
    def _is_eligible_target(
        entity: BattleEntity,
        candidate: BattleEntity,
    ) -> bool:
        """Return whether an entity is allowed to target this candidate."""
        if candidate.team == entity.team:
            return False
        if entity.target_priority == "buildings_only" and not candidate.is_building:
            return False
        # Crown Towers defend only against troops, not other buildings.
        if entity.is_building and candidate.is_building:
            return False
        if entity.target_types == "ground" and candidate.movement_type == "air":
            return False
        return True

    def _closer_visible_target(
        self,
        entity: BattleEntity,
        current_target: BattleEntity,
    ) -> BattleEntity | None:
        """Find a closer eligible target seen by a troop while it is walking."""
        current_distance = entity.position.distance_to(current_target.position)
        candidates = []

        for candidate in self.living_entities:
            if candidate.entity_id == current_target.entity_id:
                continue
            if not self._is_eligible_target(entity, candidate):
                continue

            distance = entity.position.distance_to(candidate.position)
            sight_distance = entity.sight_range * self.tile_size
            sight_distance += entity.radius + candidate.radius
            if distance > sight_distance or distance >= current_distance:
                continue

            candidates.append((distance, candidate.entity_id, candidate))

        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], item[1]))[2]

    def _is_in_attack_range(
        self,
        attacker: BattleEntity,
        target: BattleEntity,
    ) -> bool:
        distance = attacker.position.distance_to(target.position)
        allowed = attacker.attack_range * self.tile_size
        allowed += attacker.radius + target.radius
        return distance <= allowed

    def _movement_destination(
        self,
        entity: BattleEntity,
        target: BattleEntity,
    ) -> pygame.Vector2:
        """Return the next lane waypoint toward a target."""
        if entity.movement_type == "air":
            return target.position.copy()

        moving_up = entity.team == "blue"
        entity_below = entity.position.y >= self.river_bottom
        target_above = target.position.y < self.river_top
        entity_above = entity.position.y < self.river_top
        target_below = target.position.y >= self.river_bottom
        crosses_river = (
            moving_up and entity_below and target_above
        ) or (
            not moving_up and entity_above and target_below
        )
        inside_river = self.river_top <= entity.position.y < self.river_bottom

        if not crosses_river and not inside_river:
            return target.position.copy()

        if entity.lane_x is None:
            entity.lane_x = self._select_bridge(entity, target)
        bridge_x = entity.lane_x

        if moving_up:
            entry = pygame.Vector2(
                bridge_x,
                self.river_bottom + entity.radius,
            )
            exit_point = pygame.Vector2(
                bridge_x,
                self.river_top - entity.radius,
            )
        else:
            entry = pygame.Vector2(
                bridge_x,
                self.river_top - entity.radius,
            )
            exit_point = pygame.Vector2(
                bridge_x,
                self.river_bottom + entity.radius,
            )

        entry_tolerance = max(1.0, entity.movement_speed / 20)
        if not inside_river and entity.position.distance_to(entry) > entry_tolerance:
            return entry
        return exit_point

    def _select_bridge(
        self,
        entity: BattleEntity,
        target: BattleEntity,
    ) -> float:
        """Choose a stable lane using travel distance and current congestion."""
        moving_up = entity.team == "blue"
        entry_y = (
            self.river_bottom + entity.radius
            if moving_up
            else self.river_top - entity.radius
        )
        exit_y = (
            self.river_top - entity.radius
            if moving_up
            else self.river_bottom + entity.radius
        )
        choices = []

        for bridge_x in self.bridge_x_positions:
            entry = pygame.Vector2(bridge_x, entry_y)
            exit_point = pygame.Vector2(bridge_x, exit_y)
            travel_distance = entity.position.distance_to(entry)
            travel_distance += exit_point.distance_to(target.position)
            congestion = sum(
                other.entity_id != entity.entity_id
                and other.is_alive
                and not other.is_building
                and other.movement_type == "ground"
                and other.lane_x == bridge_x
                and self.river_top - self.tile_size * 3
                <= other.position.y
                <= self.river_bottom + self.tile_size * 3
                for other in self.entities
            )
            cost = travel_distance + (
                congestion
                * self.BRIDGE_CONGESTION_PENALTY_TILES
                * self.tile_size
            )
            choices.append((cost, bridge_x))

        return min(choices, key=lambda item: (item[0], item[1]))[1]

    def _desired_velocity(
        self,
        entity: BattleEntity,
        target: BattleEntity,
        delta_seconds: float,
    ) -> pygame.Vector2:
        """Blend path following with short-range unit and obstacle avoidance."""
        destination = self._movement_destination(entity, target)
        displacement = destination - entity.position
        distance = displacement.length()
        if distance <= 1e-9:
            return pygame.Vector2()

        maximum_speed = entity.movement_speed
        speed = min(maximum_speed, distance / delta_seconds)
        direction = displacement.normalize()
        velocity = direction * speed
        velocity += self._avoidance_velocity(entity, target, direction)

        if velocity.length_squared() > maximum_speed * maximum_speed:
            velocity.scale_to_length(maximum_speed)
        return velocity

    def _avoidance_velocity(
        self,
        entity: BattleEntity,
        target: BattleEntity,
        travel_direction: pygame.Vector2,
    ) -> pygame.Vector2:
        """Steer around nearby bodies before collision actually occurs."""
        steering = pygame.Vector2()

        for other in sorted(self.living_entities, key=lambda item: item.entity_id):
            if other.entity_id in {entity.entity_id, target.entity_id}:
                continue
            if not self._shares_collision_layer(entity, other):
                continue

            offset = other.position - entity.position
            distance = offset.length()
            extra_clearance = self.LOCAL_AVOIDANCE_TILES * self.tile_size
            if other.is_building:
                extra_clearance = self.BUILDING_AVOIDANCE_TILES * self.tile_size
            influence_distance = entity.radius + other.radius + extra_clearance
            if distance >= influence_distance:
                continue

            if distance <= 1e-9:
                away = self._stable_pair_normal(entity, other) * -1
            else:
                away = -offset / distance

            ahead = offset.dot(travel_direction) > 0
            influence = 1.0 - distance / influence_distance

            # Overlap correction begins immediately. Approaching units also
            # receive a deterministic side-step so queues can flow around one
            # another instead of repeatedly walking straight into a body.
            body_distance = entity.radius + other.radius
            if distance < body_distance:
                steering += away * entity.movement_speed * influence
            if ahead:
                lateral = pygame.Vector2(
                    -travel_direction.y,
                    travel_direction.x,
                )
                cross = travel_direction.cross(offset)
                if abs(cross) <= 1e-6:
                    side = -1.0 if entity.entity_id < other.entity_id else 1.0
                else:
                    side = -1.0 if cross > 0 else 1.0
                strength = 0.8 if other.is_building else 0.45
                steering += (
                    lateral
                    * side
                    * entity.movement_speed
                    * influence
                    * strength
                )
                steering += away * entity.movement_speed * influence * 0.35

        return steering

    @staticmethod
    def _shares_collision_layer(
        first: BattleEntity,
        second: BattleEntity,
    ) -> bool:
        """Return whether two bodies physically collide while moving."""
        if first.movement_type == "air" or second.movement_type == "air":
            return (
                first.movement_type == "air"
                and second.movement_type == "air"
            )
        return True

    @staticmethod
    def _stable_pair_normal(
        first: BattleEntity,
        second: BattleEntity,
    ) -> pygame.Vector2:
        """Return a reproducible separation direction for coincident bodies."""
        lower_id, higher_id = sorted((first.entity_id, second.entity_id))
        axis = (lower_id * 37 + higher_id * 17) % 4
        normals = (
            pygame.Vector2(1, 0),
            pygame.Vector2(0, 1),
            pygame.Vector2(-1, 0),
            pygame.Vector2(0, -1),
        )
        normal = normals[axis]
        return normal if first.entity_id == lower_id else -normal

    def _apply_movement(
        self,
        displacements: dict[int, pygame.Vector2],
    ) -> None:
        """Apply simultaneous movement and solve any remaining body overlaps."""
        previous_positions = {
            entity.entity_id: entity.position.copy()
            for entity in self.living_entities
        }

        for entity in sorted(self.living_entities, key=lambda item: item.entity_id):
            displacement = displacements.get(entity.entity_id)
            if displacement is None or entity.is_building:
                continue
            entity.position += displacement
            self._constrain_entity(
                entity,
                previous_positions[entity.entity_id],
            )

        for _ in range(self.COLLISION_ITERATIONS):
            changed = self._resolve_collisions(previous_positions)
            if not changed:
                break

    def _resolve_collisions(
        self,
        previous_positions: dict[int, pygame.Vector2],
    ) -> bool:
        """Separate overlapping circular bodies using mass-weighted correction."""
        entities = sorted(self.living_entities, key=lambda item: item.entity_id)
        changed = False

        for index, first in enumerate(entities):
            for second in entities[index + 1:]:
                if first.is_building and second.is_building:
                    continue
                if not self._shares_collision_layer(first, second):
                    continue

                offset = second.position - first.position
                distance = offset.length()
                required_distance = first.radius + second.radius
                if distance >= required_distance - 1e-6:
                    continue

                normal = (
                    offset / distance
                    if distance > 1e-9
                    else self._stable_pair_normal(first, second)
                )
                penetration = required_distance - distance
                first_inverse_mass = (
                    0.0 if first.is_building else 1.0 / max(first.mass, 0.01)
                )
                second_inverse_mass = (
                    0.0 if second.is_building else 1.0 / max(second.mass, 0.01)
                )
                inverse_mass_total = first_inverse_mass + second_inverse_mass
                if inverse_mass_total <= 0:
                    continue

                first.position -= (
                    normal
                    * penetration
                    * first_inverse_mass
                    / inverse_mass_total
                )
                second.position += (
                    normal
                    * penetration
                    * second_inverse_mass
                    / inverse_mass_total
                )
                self._constrain_entity(
                    first,
                    previous_positions[first.entity_id],
                )
                self._constrain_entity(
                    second,
                    previous_positions[second.entity_id],
                )
                changed = True

        return changed

    def _constrain_entity(
        self,
        entity: BattleEntity,
        previous_position: pygame.Vector2,
    ) -> None:
        """Keep bodies inside the arena and ground troops out of open water."""
        if entity.is_building:
            return

        entity.position.x = min(
            self.arena_right - entity.radius,
            max(self.arena_left + entity.radius, entity.position.x),
        )
        entity.position.y = min(
            self.screen_height - entity.radius,
            max(entity.radius, entity.position.y),
        )
        if entity.movement_type == "air":
            return

        entered_from_above = (
            previous_position.y + entity.radius <= self.river_top
            and entity.position.y + entity.radius > self.river_top
        )
        entered_from_below = (
            previous_position.y - entity.radius >= self.river_bottom
            and entity.position.y - entity.radius < self.river_bottom
        )
        if (
            (entered_from_above or entered_from_below)
            and self._bridge_under_entity(entity) is None
        ):
            entity.position.y = (
                self.river_top - entity.radius
                if entered_from_above
                else self.river_bottom + entity.radius
            )
            return

        overlaps_river = (
            entity.position.y + entity.radius > self.river_top
            and entity.position.y - entity.radius < self.river_bottom
        )
        if not overlaps_river:
            return

        bridge_x = self._bridge_under_entity(entity)
        if bridge_x is not None:
            half_width = self.BRIDGE_HALF_WIDTH_TILES * self.tile_size
            entity.position.x = min(
                bridge_x + half_width - entity.radius,
                max(
                    bridge_x - half_width + entity.radius,
                    entity.position.x,
                ),
            )
            return

        previous_inside_river = (
            self.river_top <= previous_position.y <= self.river_bottom
        )
        if previous_inside_river and entity.lane_x is not None:
            entity.position.x = entity.lane_x
            return

        if previous_position.y < self.river_top:
            entity.position.y = self.river_top - entity.radius
        elif previous_position.y > self.river_bottom:
            entity.position.y = self.river_bottom + entity.radius
        elif entity.position.y < (self.river_top + self.river_bottom) / 2:
            entity.position.y = self.river_top - entity.radius
        else:
            entity.position.y = self.river_bottom + entity.radius

    def _bridge_under_entity(self, entity: BattleEntity) -> float | None:
        """Return a bridge that fully supports this ground unit's body."""
        half_width = self.BRIDGE_HALF_WIDTH_TILES * self.tile_size
        usable_half_width = max(0.0, half_width - entity.radius)
        candidates = [
            bridge_x
            for bridge_x in self.bridge_x_positions
            if abs(entity.position.x - bridge_x) <= usable_half_width + 1e-6
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda bridge_x: abs(entity.position.x - bridge_x),
        )

    def apply_knockback(
        self,
        entity_id: int,
        source_position: tuple[float, float] | pygame.Vector2,
        *,
        distance_tiles: float,
        duration: float = 0.25,
    ) -> bool:
        """Interrupt and push a movable troop away from a source position."""
        if duration <= 0:
            raise ValueError("Knockback duration must be positive")
        if distance_tiles < 0:
            raise ValueError("Knockback distance cannot be negative")

        entity = self.entity_by_id(entity_id)
        if entity is None or not entity.is_alive or entity.is_building:
            return False

        direction = entity.position - pygame.Vector2(source_position)
        if direction.length_squared() <= 1e-9:
            direction = pygame.Vector2(
                -1 if entity.entity_id % 2 else 1,
                0,
            )
        else:
            direction = direction.normalize()

        effective_distance = (
            distance_tiles
            * self.tile_size
            * max(0.0, 1.0 - entity.knockback_resistance)
        )
        if effective_distance <= 0:
            return False

        entity.knockback_velocity = (
            direction * effective_distance / duration
        )
        entity.knockback_remaining = duration
        entity.target_id = None
        entity.state = EntityState.RETARGETING
        return True

    def _attack_if_ready(
        self,
        attacker: BattleEntity,
        target: BattleEntity,
    ) -> None:
        if attacker.attack_cooldown > 0:
            return

        attacker.attack_cooldown = attacker.hit_speed
        if attacker.projectile_speed > 0 or attacker.attack_style == "ranged":
            self._create_projectile(attacker, target)
        else:
            target.take_damage(attacker.damage)

    def _create_projectile(
        self,
        attacker: BattleEntity,
        target: BattleEntity,
    ) -> None:
        color = (245, 222, 115) if attacker.is_building else (218, 234, 255)
        speed_tiles = (
            attacker.projectile_speed
            if attacker.projectile_speed > 0
            else 10.0
        )
        self.projectiles.append(
            Projectile(
                projectile_id=self._next_projectile_id,
                source_id=attacker.entity_id,
                target_id=target.entity_id,
                team=attacker.team,
                position=attacker.position.copy(),
                damage=attacker.damage,
                speed=speed_tiles * self.tile_size,
                color=color,
            )
        )
        self._next_projectile_id += 1

    def _update_projectiles(self, delta_seconds: float) -> None:
        survivors = []

        for projectile in self.projectiles:
            target = self.entity_by_id(projectile.target_id)
            if target is None or not target.is_alive:
                continue

            displacement = target.position - projectile.position
            distance = displacement.length()
            step = projectile.speed * delta_seconds

            if distance <= step + target.radius:
                target.take_damage(projectile.damage)
                if self.winning_team is not None:
                    break
                continue

            if distance > 0:
                projectile.position += displacement.normalize() * step
            survivors.append(projectile)

        self.projectiles = survivors

    def _activate_king_towers(self) -> None:
        """Wake each King Tower after either allied Princess Tower dies.

        Direct damage wakes a King Tower inside ``take_damage``. This method
        handles the other activation rule, which depends on allied tower state.
        """
        for king in (
            entity
            for entity in self.entities
            if entity.tower_kind == "king" and entity.is_alive
        ):
            allied_princesses = [
                entity
                for entity in self.entities
                if entity.team == king.team
                and entity.tower_kind == "princess"
            ]
            if any(not tower.is_alive for tower in allied_princesses):
                king.active = True
