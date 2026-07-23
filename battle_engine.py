"""Deterministic combat rules for the arena simulator.

The engine owns mutable battle state but knows nothing about buttons, fonts, or
card dragging. Keeping it separate from Pygame drawing lets tests advance a
battle by exact time steps and makes the same engine reusable by an RL agent.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SpellStats:
    """Level-11 direct spell damage and affected radius."""

    damage: int
    crown_tower_damage: int
    radius: float


class CardLike(Protocol):
    """The card attributes required by the engine without importing the UI."""

    name: str
    card_type: str
    target_priority: str
    target_types: str
    attack_style: str
    unit_count: int
    unit_stats: UnitStats | None
    spell_stats: SpellStats | None


@dataclass
class BattleEntity:
    """A troop or Crown Tower participating in combat."""

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
    projectile_speed: float
    target_priority: str
    target_types: str
    attack_style: str
    radius: float
    is_building: bool = False
    tower_kind: str | None = None
    active: bool = True
    state: EntityState = EntityState.RETARGETING
    target_id: int | None = None
    attack_cooldown: float = 0.0

    @property
    def is_alive(self) -> bool:
        """Return whether this entity can move, attack, or be targeted."""
        return self.state is not EntityState.DEAD and self.health > 0

    def take_damage(self, amount: float) -> None:
        """Apply damage once and enter the permanent dead state at zero health."""
        if not self.is_alive:
            return

        self.health = max(0.0, self.health - amount)
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
            projectile_speed=self.TOWER_PROJECTILE_SPEED,
            target_priority="nearest_enemy",
            target_types="air_and_ground",
            attack_style="ranged",
            radius=30.0 if is_king else 23.0,
            is_building=True,
            tower_kind=tower_kind,
            # King Towers wake only after one allied Princess Tower is destroyed.
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
        if card.card_type == "spell":
            self.cast_spell(card, team, position)
            return ()

        if card.unit_stats is None:
            raise ValueError(f"Troop card {card.name} has no unit statistics")

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
        """Create one runtime entity from immutable card statistics."""
        stats = card.unit_stats
        if stats is None:
            raise ValueError(f"Troop card {card.name} has no unit statistics")

        radius_by_name = {
            "Skeletons": 5.0,
            "Archers": 7.0,
            "Musketeer": 9.0,
            "Knight": 10.0,
            "Mini P.E.K.K.A": 10.0,
            "Giant": 15.0,
        }
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
            projectile_speed=stats.projectile_speed,
            target_priority=card.target_priority,
            target_types=card.target_types,
            attack_style=card.attack_style,
            radius=radius_by_name.get(card.name, 9.0),
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

        self._activate_king_towers()

    def update(self, delta_seconds: float) -> None:
        """Advance a deterministic slice of battle time."""
        if delta_seconds <= 0:
            return

        self._activate_king_towers()

        # Entity IDs and locked target IDs remain stable even if another entity
        # dies during this loop.
        for entity in tuple(self.entities):
            if not entity.is_alive or not entity.active:
                continue

            entity.attack_cooldown = max(
                0.0,
                entity.attack_cooldown - delta_seconds,
            )
            target = self.entity_by_id(entity.target_id)

            # A target lock ends only when that exact target dies or disappears.
            if target is None or not target.is_alive:
                entity.target_id = None
                entity.state = EntityState.RETARGETING
                target = self._acquire_target(entity)
                if target is not None:
                    entity.target_id = target.entity_id

            if target is None:
                continue

            if self._is_in_attack_range(entity, target):
                entity.state = EntityState.ATTACKING
                self._attack_if_ready(entity, target)
            elif entity.is_building:
                # Towers cannot move and keep their lock while the target lives.
                entity.state = EntityState.ATTACKING
            else:
                entity.state = EntityState.MOVING
                self._move_toward_target(entity, target, delta_seconds)

        self._update_projectiles(delta_seconds)
        self._activate_king_towers()

    def _acquire_target(self, entity: BattleEntity) -> BattleEntity | None:
        """Choose the nearest eligible target when no lock currently exists."""
        candidates = []

        for candidate in self.living_entities:
            if candidate.team == entity.team:
                continue
            if entity.target_priority == "buildings_only" and not candidate.is_building:
                continue
            # Crown Towers defend only against troops, not other buildings.
            if entity.is_building and candidate.is_building:
                continue
            if (
                entity.target_types == "ground"
                and candidate.target_types == "air"
            ):
                continue

            distance = entity.position.distance_to(candidate.position)
            if entity.is_building:
                maximum = entity.attack_range * self.tile_size
                maximum += entity.radius + candidate.radius
                if distance > maximum:
                    continue

            candidates.append((distance, candidate.entity_id, candidate))

        if not candidates:
            return None

        # Entity ID breaks equal-distance ties deterministically.
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
        """Route cross-river ground movement through the nearest bridge."""
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
            return target.position

        bridge_x = min(
            self.bridge_x_positions,
            key=lambda x: abs(x - entity.position.x),
        )

        if moving_up:
            destination_y = (
                self.river_top - 1
                if inside_river
                else self.river_bottom - 1
            )
        else:
            destination_y = (
                self.river_bottom + 1
                if inside_river
                else self.river_top + 1
            )

        return pygame.Vector2(bridge_x, destination_y)

    def _move_toward_target(
        self,
        entity: BattleEntity,
        target: BattleEntity,
        delta_seconds: float,
    ) -> None:
        destination = self._movement_destination(entity, target)
        displacement = destination - entity.position
        distance = displacement.length()
        if distance == 0:
            return

        step = min(distance, entity.movement_speed * delta_seconds)
        entity.position += displacement.normalize() * step

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
                continue

            if distance > 0:
                projectile.position += displacement.normalize() * step
            survivors.append(projectile)

        self.projectiles = survivors

    def _activate_king_towers(self) -> None:
        """Wake each King Tower after either allied Princess Tower dies."""
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
