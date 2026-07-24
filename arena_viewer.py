"""Show and run the Clash Royale-style arena.

This file is split into five main parts:

1. Settings for the screen, game rules, and colors.
2. Classes that store tower and Elixir data.
3. Helper functions for the match timer.
4. The ``ArenaViewer`` class, which handles input, updates, and drawing.
5. The ``main`` function, which starts the game.

For each screen frame, the game checks input, runs any ready 50-millisecond
game updates, and redraws the screen.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import math

import pygame

from battle_engine import (
    BattleEngine,
    BattleEntity,
    SpellStats,
    UnitStats,
)
from controllers import (
    ControllerCard,
    ControllerContext,
    HumanController,
    PlayCardAction,
    PlayerController,
    controller_names,
    create_controller,
)


# ---------------------------------------------------------------------------
# Screen and arena settings
# ---------------------------------------------------------------------------
# The playable arena remains an 18-by-32 grid. Non-playable stadium sidelines
# widen the window without changing any tile or battle dimensions.
ARENA_WIDTH = 450
STADIUM_BUFFER_WIDTH = 70
ARENA_LEFT = STADIUM_BUFFER_WIDTH
ARENA_RIGHT = ARENA_LEFT + ARENA_WIDTH
SCREEN_WIDTH = ARENA_WIDTH + STADIUM_BUFFER_WIDTH * 2
ARENA_HEIGHT = 800
HUD_HEIGHT = 153
SCREEN_HEIGHT = ARENA_HEIGHT + HUD_HEIGHT
# Draw at the full logical resolution, then uniformly shrink the finished frame
# for the actual desktop window. Game coordinates and RL observations therefore
# remain unchanged while the entire arena fits comfortably on smaller screens.
WINDOW_SCALE = 0.75
WINDOW_WIDTH = int(SCREEN_WIDTH * WINDOW_SCALE + 0.5)
WINDOW_HEIGHT = int(SCREEN_HEIGHT * WINDOW_SCALE + 0.5)
GRID_COLUMNS = 18
GRID_ROWS = 32
TILE_SIZE = 25
RIVER_HEIGHT = TILE_SIZE * 2
RIVER_TOP = (ARENA_HEIGHT - RIVER_HEIGHT) // 2
# Destroying an enemy Princess Tower unlocks that lane from the river to one
# grid row below the midpoint of the enemy half. The small offset keeps the
# forward placement boundary from reaching too deeply into enemy territory.
ENEMY_DEPLOYMENT_UNLOCK_TOP = math.ceil(
    (RIVER_TOP / 2) / TILE_SIZE,
) * TILE_SIZE + TILE_SIZE
FPS = 60
# Draw up to 60 frames per second, but always update the battle in exact
# 50-millisecond steps. This gives the simulator 20 updates per second.
FIXED_TIMESTEP_MS = 50
FIXED_TIMESTEP_SECONDS = FIXED_TIMESTEP_MS / 1000
MATCH_DURATION_SECONDS = 3 * 60
OVERTIME_DURATION_SECONDS = 2 * 60

# All colors are kept here so they are easy to find and change.
ARENA_COLOR = (74, 145, 82)
ALTERNATE_TILE_COLOR = (78, 151, 86)
RIVER_COLOR = (56, 144, 201)
RIVER_HIGHLIGHT_COLOR = (91, 177, 224)
BRIDGE_PLANK_COLOR = (174, 112, 62)
BRIDGE_PLANK_LIGHT_COLOR = (213, 151, 87)
BRIDGE_EDGE_COLOR = (101, 64, 42)
GRID_COLOR = (42, 91, 51)
HOVER_COLOR = (255, 255, 255, 75)
RESTRICTED_TILE_COLOR = (255, 110, 110, 78)
SELECTED_COLOR = (255, 218, 71)
TEXT_COLOR = (245, 245, 245)
STONE_COLOR = (207, 196, 170)
STONE_SHADOW_COLOR = (135, 125, 107)
STONE_HIGHLIGHT_COLOR = (235, 226, 204)
TOWER_OPENING_COLOR = (50, 43, 39)
RED_TEAM_COLOR = (205, 57, 61)
RED_TEAM_LIGHT_COLOR = (244, 93, 87)
BLUE_TEAM_COLOR = (45, 111, 196)
BLUE_TEAM_LIGHT_COLOR = (74, 157, 229)
CROWN_COLOR = (255, 205, 54)
CROWN_SHADOW_COLOR = (193, 128, 28)
TIMER_PANEL_COLOR = (16, 18, 20)
TIMER_BORDER_COLOR = (72, 77, 82)
TIMER_SHADOW_COLOR = (0, 0, 0, 125)
TIMER_URGENT_COLOR = (255, 92, 84)
STADIUM_FLOOR_COLOR = (82, 67, 53)
STADIUM_TIER_COLOR = (103, 84, 63)
STADIUM_RAIL_COLOR = (154, 145, 124)
SPECTATOR_SKIN_COLOR = (221, 171, 121)

# ---------------------------------------------------------------------------
# Arena layout
# ---------------------------------------------------------------------------
# These positions place the towers and bridges in the middle of their tiles.
# Reusing them keeps the towers and bridges lined up.
LEFT_LANE_X = (
    ARENA_LEFT
    + (ARENA_WIDTH // 4) // TILE_SIZE * TILE_SIZE
    + TILE_SIZE // 2
)
RIGHT_LANE_X = ARENA_RIGHT - (LEFT_LANE_X - ARENA_LEFT)
CENTER_LANE_X = SCREEN_WIDTH // 2
TOP_KING_Y = TILE_SIZE * 4
TOP_PRINCESS_Y = TILE_SIZE * 8
BRIDGE_WIDTH = TILE_SIZE * 2
BRIDGE_HEIGHT = RIVER_HEIGHT + TILE_SIZE

# ---------------------------------------------------------------------------
# Elixir settings and colors
# ---------------------------------------------------------------------------
# Elixir uses decimal values so it can fill smoothly. The bar only shows the
# whole number. Triple Elixir is included in case overtime is added later.
ELIXIR_MAX = 10.0
ELIXIR_START = 5.0
ELIXIR_SECONDS_PER_UNIT = 2.8
DOUBLE_ELIXIR_START = 120.0
# Overtime starts at 3:00; Triple Elixir begins one minute into that phase.
TRIPLE_ELIXIR_START = MATCH_DURATION_SECONDS + 60.0
ELIXIR_COLOR = (220, 52, 213)
ELIXIR_HIGHLIGHT_COLOR = (255, 115, 241)
ELIXIR_DARK_COLOR = (85, 24, 105)
ELIXIR_EMPTY_COLOR = (48, 27, 64)
ELIXIR_FRAME_COLOR = (30, 18, 40)
# Show a large arena announcement briefly when Elixir generation speeds up.
# Keeping the duration as a named constant makes it easy to tune later.
ELIXIR_MULTIPLIER_NOTICE_SECONDS = 2.5
ELIXIR_NOTICE_PANEL_COLOR = (31, 13, 45, 225)
ELIXIR_NOTICE_BORDER_COLOR = (245, 92, 232)
# Regulation-to-overtime feedback uses its own announcement so it remains
# visually distinct from Double and Triple Elixir changes.
OVERTIME_NOTICE_SECONDS = 1.5
OVERTIME_NOTICE_PANEL_COLOR = (8, 9, 12, 235)
OVERTIME_NOTICE_TEXT_COLOR = (244, 70, 70)

# The temporary card HUD sits immediately above the Elixir bar. Keeping these
# measurements together makes both drawing and mouse hit-testing use the same
# rectangles.
HAND_HUD_TOP = ARENA_HEIGHT
CARD_WIDTH = 72
CARD_HEIGHT = 84
CARD_GAP = 6
CARD_START_X = ARENA_LEFT + 8
NEXT_CARD_X = ARENA_LEFT + 360
# Affordable cards use this normal dark-blue face.
CARD_BACKGROUND_COLOR = (45, 53, 66)
# Unaffordable cards start darker and receive the transparent overlay below.
CARD_DISABLED_COLOR = (27, 30, 37)
CARD_BORDER_COLOR = (188, 198, 214)
# The fourth number is alpha: 0 is invisible and 255 is completely opaque.
# This layer darkens every part of an unaffordable card, not only its border.
CARD_DISABLED_OVERLAY_COLOR = (5, 7, 10, 145)
DEPLOYMENT_COLOR = (59, 135, 224)
# Spell previews use transparent fills so troops, towers, and arena tiles remain
# visible underneath the affected area. The outline makes the exact edge clear.
SPELL_RADIUS_VALID_FILL = (61, 170, 255, 65)
SPELL_RADIUS_VALID_BORDER = (113, 207, 255, 220)
SPELL_RADIUS_INVALID_FILL = (238, 66, 74, 55)
SPELL_RADIUS_INVALID_BORDER = (255, 113, 119, 220)


# ---------------------------------------------------------------------------
# Game data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tower:
    """Store the type, team, and position of one tower.

    ``kind`` is either king or princess. ``team`` is red or blue. ``center`` is
    the tower's center point on the screen. Towers cannot be changed after they
    are created because they do not move.
    """

    kind: str
    team: str
    center: tuple[int, int]


class PlacementRule(Enum):
    """Describe which arena region accepts a card deployment.

    Keeping placement policy separate from card type matters because future
    cards may be exceptions. For example, a troop such as Miner can be allowed
    anywhere even though ordinary troops must start in friendly territory.
    """

    FRIENDLY_TERRITORY = "friendly_territory"
    ANYWHERE = "anywhere"


@dataclass(frozen=True)
class Card:
    """Describe one reusable card and the rules its future entity will follow.

    ``role`` is a strategic label, while the remaining fields are simulator
    rules. Ground targeting includes ground troops and buildings. Spells have
    no spawned units because they apply an effect directly to a chosen area.
    """

    name: str
    elixir_cost: int
    card_type: str
    role: str
    placement_rule: PlacementRule
    target_priority: str
    target_types: str
    attack_style: str
    unit_count: int
    unit_stats: UnitStats | None
    spell_stats: SpellStats | None


@dataclass
class CardCycle:
    """Store the four-card hand and the four cards waiting behind it.

    The original deck tuple never changes. ``hand`` and ``queue`` are the two
    moving parts of the current match:

    * ``hand`` contains the four cards the player can select.
    * ``queue`` contains the next four cards in their exact cycle order.
    """

    deck: tuple[Card, ...]

    def __post_init__(self) -> None:
        """Build the starting hand and queue from an eight-card deck."""
        if len(self.deck) != 8:
            raise ValueError("A deck must contain exactly eight cards")

        # The first four deck positions are visible immediately.
        self.hand = list(self.deck[:4])
        # The remaining four wait in order; queue[0] is always the next preview.
        self.queue = list(self.deck[4:])

    @property
    def next_card(self) -> Card:
        """Return the card that will enter the hand after the next play."""
        return self.queue[0]

    def play(self, hand_index: int) -> Card:
        """Cycle a successfully played card to the back of the queue."""
        if not 0 <= hand_index < len(self.hand):
            raise IndexError("Hand index must be between 0 and 3")

        # Save the card before overwriting its hand position.
        played_card = self.hand[hand_index]
        # The front queued card fills the exact hand slot that was just played.
        self.hand[hand_index] = self.queue.pop(0)
        # A played card must wait for all other queued cards before returning.
        self.queue.append(played_card)
        return played_card


@dataclass(frozen=True)
class Deployment:
    """Record a temporary card marker placed on one arena tile."""

    card: Card
    tile: tuple[int, int]


@dataclass
class ElixirMeter:
    """Store the player's Elixir and handle adding or spending it.

    This class does not draw anything. The game rules stay separate from the
    Elixir bar, which makes them easier to test and reuse.
    """

    amount: float = ELIXIR_START
    full_notice_remaining: float = 0.0

    @staticmethod
    def multiplier_at(match_elapsed: float) -> int:
        """Return the Elixir speed for the given point in the match."""
        if match_elapsed >= TRIPLE_ELIXIR_START:
            return 3
        if match_elapsed >= DOUBLE_ELIXIR_START:
            return 2
        return 1

    def update(self, delta_seconds: float, match_elapsed: float) -> None:
        """Add the Elixir earned during one game update.

        An update may cross from normal speed into Double Elixir. If that
        happens, the code splits the update so each part uses the correct speed.
        """
        if delta_seconds <= 0:
            return

        self.full_notice_remaining = max(
            0.0,
            self.full_notice_remaining - delta_seconds,
        )
        if self.amount >= ELIXIR_MAX:
            return

        remaining = delta_seconds
        cursor = match_elapsed
        generated = 0.0

        # Most updates use this loop once. An update that crosses a speed change
        # uses it more than once.
        while remaining > 0:
            if cursor < DOUBLE_ELIXIR_START:
                boundary = DOUBLE_ELIXIR_START
            elif cursor < TRIPLE_ELIXIR_START:
                boundary = TRIPLE_ELIXIR_START
            else:
                boundary = math.inf

            step = min(remaining, boundary - cursor)
            generated += (
                step
                * self.multiplier_at(cursor)
                / ELIXIR_SECONDS_PER_UNIT
            )
            cursor += step
            remaining -= step

        previous_amount = self.amount
        self.amount = min(ELIXIR_MAX, self.amount + generated)
        # Show the full message only when the bar first reaches 10.
        if previous_amount < ELIXIR_MAX and self.amount >= ELIXIR_MAX:
            self.full_notice_remaining = 1.5

    def spend(self, cost: int) -> bool:
        """Spend whole elixir if the player can afford the requested cost."""
        if cost < 0 or self.amount + 1e-9 < cost:
            return False

        self.amount -= cost
        self.full_notice_remaining = 0.0
        return True


@dataclass
class PlayerState:
    """Mutable card-cycle and Elixir state independently owned by one team."""

    team: str
    card_cycle: CardCycle
    elixir: ElixirMeter


@dataclass
class FixedTimestepClock:
    """Turn changing frame times into equal simulation steps.

    Drawing may take a different amount of time on each frame. This clock saves
    that time until there is enough for one or more 50-millisecond game updates.
    Any smaller amount is kept for the next frame.
    """

    waiting_ms: int = 0

    def add_frame_time(self, frame_ms: int) -> int:
        """Add one frame's time and return how many game updates should run."""
        if frame_ms < 0:
            raise ValueError("Frame time cannot be negative")

        self.waiting_ms += frame_ms
        step_count, self.waiting_ms = divmod(
            self.waiting_ms,
            FIXED_TIMESTEP_MS,
        )
        return step_count

    def reset(self) -> None:
        """Remove time left over from the previous match."""
        self.waiting_ms = 0


# These are data-only placeholder cards. Adding sprites or unit behavior later
# will not require changing CardCycle because it only cares about card order.
DEFAULT_DECK = (
    Card(
        "Knight",
        3,
        "troop",
        "mini_tank",
        PlacementRule.FRIENDLY_TERRITORY,
        "nearest_enemy",
        "ground",
        "melee",
        1,
        UnitStats(1766, 202, 1.2, 1.0, 1.2),
        None,
    ),
    Card(
        "Archers",
        3,
        "troop",
        "ranged_support",
        PlacementRule.FRIENDLY_TERRITORY,
        "nearest_enemy",
        "air_and_ground",
        "ranged",
        2,
        UnitStats(304, 112, 0.9, 1.0, 5.0, 10.0),
        None,
    ),
    Card(
        "Giant",
        5,
        "troop",
        "win_condition",
        PlacementRule.FRIENDLY_TERRITORY,
        "buildings_only",
        "ground",
        "melee",
        1,
        UnitStats(4090, 253, 1.5, 0.75, 1.2),
        None,
    ),
    Card(
        "Fireball",
        4,
        "spell",
        "big_spell",
        PlacementRule.ANYWHERE,
        "targeted_area",
        "air_and_ground",
        "area",
        0,
        None,
        SpellStats(688, 207, 2.5),
    ),
    Card(
        "Mini P.E.K.K.A",
        4,
        "troop",
        "tank_killer",
        PlacementRule.FRIENDLY_TERRITORY,
        "nearest_enemy",
        "ground",
        "melee",
        1,
        UnitStats(1390, 755, 1.6, 1.5, 0.8),
        None,
    ),
    Card(
        "Musketeer",
        4,
        "troop",
        "ranged_support",
        PlacementRule.FRIENDLY_TERRITORY,
        "nearest_enemy",
        "air_and_ground",
        "ranged",
        1,
        UnitStats(721, 217, 1.0, 1.0, 6.0, 1000 / 60),
        None,
    ),
    Card(
        "Skeletons",
        1,
        "troop",
        "cycle_swarm",
        PlacementRule.FRIENDLY_TERRITORY,
        "nearest_enemy",
        "ground",
        "melee",
        3,
        UnitStats(81, 81, 1.1, 1.5, 0.5),
        None,
    ),
    Card(
        "Zap",
        2,
        "spell",
        "small_spell",
        PlacementRule.ANYWHERE,
        "targeted_area",
        "air_and_ground",
        "area",
        0,
        None,
        SpellStats(192, 58, 2.5),
    ),
)


# Define every tower in one place. The blue tower positions mirror the red
# tower positions.
TOWERS = (
    Tower("king", "red", (CENTER_LANE_X, TOP_KING_Y)),
    Tower("princess", "red", (LEFT_LANE_X, TOP_PRINCESS_Y)),
    Tower("princess", "red", (RIGHT_LANE_X, TOP_PRINCESS_Y)),
    Tower(
        "princess",
        "blue",
        (LEFT_LANE_X, ARENA_HEIGHT - TOP_PRINCESS_Y),
    ),
    Tower(
        "princess",
        "blue",
        (RIGHT_LANE_X, ARENA_HEIGHT - TOP_PRINCESS_Y),
    ),
    Tower("king", "blue", (CENTER_LANE_X, ARENA_HEIGHT - TOP_KING_Y)),
)


# ---------------------------------------------------------------------------
# Match timer helpers
# ---------------------------------------------------------------------------
def remaining_match_seconds(
    start_ms: int,
    now_ms: int,
    duration_seconds: int = MATCH_DURATION_SECONDS,
) -> int:
    """Return the number of full seconds left in a timed match phase.

    This keeps ``3:00`` on screen for the first second. It does not show
    ``0:00`` until the full three minutes have passed.
    """
    elapsed_ms = max(0, now_ms - start_ms)
    remaining_ms = max(0, duration_seconds * 1000 - elapsed_ms)
    return (remaining_ms + 999) // 1000


def format_match_time(seconds: int) -> str:
    """Format a countdown value as M:SS."""
    minutes, seconds_part = divmod(max(0, seconds), 60)
    return f"{minutes}:{seconds_part:02d}"


class ArenaViewer:
    """Handle the game window, input, updates, and drawing.

    Each drawing method adds one part of the screen. Parts drawn later appear
    on top of parts drawn earlier.
    """

    def __init__(
        self,
        blue_controller: str | PlayerController = "human",
        red_controller: str | PlayerController = "scripted",
    ) -> None:
        """Set up Pygame and create the starting game data."""
        pygame.init()
        pygame.display.set_caption("Royale Simulator - Grid Arena")

        # All drawing uses the full logical canvas. Only the final frame is
        # scaled into the smaller desktop window.
        self.display_surface = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT),
        )
        self.screen = pygame.Surface(
            (SCREEN_WIDTH, SCREEN_HEIGHT),
        ).convert()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.timer_label_font = pygame.font.Font(None, 19)
        self.timer_label_font.set_bold(True)
        self.timer_font = pygame.font.Font(None, 42)
        self.timer_font.set_bold(True)
        self.match_over_font = pygame.font.Font(None, 52)
        self.match_over_font.set_bold(True)

        # Save the start time for the timer. Elixir tracks its own elapsed time.
        self.match_started_at = pygame.time.get_ticks()
        self.elixir_font = pygame.font.Font(None, 31)
        self.elixir_notice_font = pygame.font.Font(None, 25)
        self.elixir_multiplier_font = pygame.font.Font(None, 52)
        self.elixir_multiplier_font.set_bold(True)
        self.card_font = pygame.font.Font(None, 19)
        self.card_cost_font = pygame.font.Font(None, 23)
        self.running = True
        # selected_tile is visual feedback for the player's last arena click.
        self.selected_tile: tuple[int, int] | None = None
        # None means that an arena click should not attempt to play a card.
        self.selected_card_index: int | None = None
        # Drag state is separate from selection so click-to-place still works.
        self.dragged_card_index: int | None = None
        self.drag_position: tuple[int, int] | None = None
        self.controllers = {
            "blue": (
                create_controller(blue_controller, "blue")
                if isinstance(blue_controller, str)
                else blue_controller
            ),
            "red": (
                create_controller(red_controller, "red")
                if isinstance(red_controller, str)
                else red_controller
            ),
        }
        self.local_team = next(
            (
                team
                for team in ("blue", "red")
                if isinstance(self.controllers[team], HumanController)
            ),
            "blue",
        )
        self.players = self.create_player_states()
        self.sync_local_player_aliases()
        self.controller_decision_elapsed = {"blue": 0.0, "red": 0.0}
        pygame.display.set_caption(
            "Royale Simulator - "
            f"Blue: {self.controllers['blue'].name} vs "
            f"Red: {self.controllers['red'].name}",
        )
        # Zero means no large multiplier announcement is currently visible.
        # The multiplier value is stored separately so the same system can also
        # announce Triple Elixir if overtime is added later.
        self.elixir_multiplier_notice: int | None = None
        self.elixir_multiplier_notice_remaining = 0.0
        # Deployment history is useful for replay/debugging. Mutable combat
        # entities themselves live in BattleEngine.
        self.deployments: list[Deployment] = []
        self.battle = self.create_battle_engine()
        self.fixed_timestep = FixedTimestepClock()
        self.match_elapsed = 0.0
        self.match_finished = False
        self.match_winner: str | None = None
        self.match_finished_at_ms: int | None = None
        self.overtime_active = False
        self.overtime_started_at_ms: int | None = None
        self.overtime_notice_remaining = 0.0

    @staticmethod
    def create_battle_engine() -> BattleEngine:
        """Create a fresh combat world shared by startup and Play Again."""
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

    @staticmethod
    def create_player_states() -> dict[str, PlayerState]:
        """Create equal independent hand, queue, and Elixir state for each side."""
        return {
            team: PlayerState(
                team=team,
                card_cycle=CardCycle(DEFAULT_DECK),
                elixir=ElixirMeter(),
            )
            for team in ("blue", "red")
        }

    def sync_local_player_aliases(self) -> None:
        """Point the existing HUD helpers at the selected local player."""
        local_player = self.players[self.local_team]
        self.card_cycle = local_player.card_cycle
        self.elixir = local_player.elixir

    def reset_match(self, now_ms: int | None = None) -> None:
        """Reset every mutable episode value for a completely fresh rematch."""
        self.battle = self.create_battle_engine()
        if not hasattr(self, "controllers"):
            self.controllers = {
                "blue": create_controller("human", "blue"),
                "red": create_controller("scripted", "red"),
            }
        if not hasattr(self, "local_team"):
            self.local_team = "blue"
        self.players = self.create_player_states()
        self.sync_local_player_aliases()
        self.controller_decision_elapsed = {"blue": 0.0, "red": 0.0}
        for controller in self.controllers.values():
            controller.reset()
        self.fixed_timestep = FixedTimestepClock()
        self.deployments = []

        self.match_started_at = (
            pygame.time.get_ticks()
            if now_ms is None
            else now_ms
        )
        self.match_elapsed = 0.0
        self.match_finished = False
        self.match_winner = None
        self.match_finished_at_ms = None
        self.overtime_active = False
        self.overtime_started_at_ms = None

        self.selected_tile = None
        self.selected_card_index = None
        self.dragged_card_index = None
        self.drag_position = None

        self.elixir_multiplier_notice = None
        self.elixir_multiplier_notice_remaining = 0.0
        self.overtime_notice_remaining = 0.0

    @staticmethod
    def play_again_button_rectangle() -> pygame.Rect:
        """Return the logical-space button used for drawing and hit testing."""
        button = pygame.Rect(0, 0, 190, 44)
        button.center = (SCREEN_WIDTH // 2, ARENA_HEIGHT // 2 + 70)
        return button

    def match_result_text(self) -> tuple[str, str]:
        """Return the final outcome title and Crown score."""
        scores = self.battle.crown_scores
        title = (
            f"{self.match_winner.upper()} WINS"
            if self.match_winner is not None
            else "DRAW"
        )
        score = f"RED {scores['red']}  -  {scores['blue']} BLUE"
        return title, score

    # ------------------------------------------------------------------
    # Position and movement helpers
    # ------------------------------------------------------------------
    @staticmethod
    def display_to_logical_position(
        position: tuple[int, int],
    ) -> tuple[int, int]:
        """Map a point in the scaled window back onto the logical canvas."""
        display_x, display_y = position
        return (
            int(display_x * SCREEN_WIDTH / WINDOW_WIDTH),
            int(display_y * SCREEN_HEIGHT / WINDOW_HEIGHT),
        )

    @staticmethod
    def screen_to_tile(position: tuple[int, int]) -> tuple[int, int] | None:
        """Change a screen position into a tile position.

        Return ``None`` when the position is outside the arena.
        """
        mouse_x, mouse_y = position

        if not (
            ARENA_LEFT <= mouse_x < ARENA_RIGHT
            and 0 <= mouse_y < ARENA_HEIGHT
        ):
            return None

        column = (mouse_x - ARENA_LEFT) // TILE_SIZE
        row = mouse_y // TILE_SIZE
        return column, row

    @staticmethod
    def tile_rectangle(tile: tuple[int, int]) -> pygame.Rect:
        """Return the screen rectangle occupied by a tile."""
        column, row = tile
        return pygame.Rect(
            ARENA_LEFT + column * TILE_SIZE,
            row * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        )

    @staticmethod
    def river_rectangle() -> pygame.Rect:
        """Return a horizontal river centered exactly within the arena."""
        return pygame.Rect(
            ARENA_LEFT,
            RIVER_TOP,
            ARENA_WIDTH,
            RIVER_HEIGHT,
        )

    @staticmethod
    def bridge_rectangles() -> tuple[pygame.Rect, pygame.Rect]:
        """Return rectangles for the left and right bridges.

        A new pair is made each time, so callers can safely change their copies.
        """
        bridges = []

        for center_x in (LEFT_LANE_X, RIGHT_LANE_X):
            bridge = pygame.Rect(0, 0, BRIDGE_WIDTH, BRIDGE_HEIGHT)
            bridge.center = (center_x, ARENA_HEIGHT // 2)
            bridges.append(bridge)

        return bridges[0], bridges[1]

    @classmethod
    def is_walkable_position(cls, position: tuple[int, int]) -> bool:
        """Check if a ground unit can stand at a screen position.

        Units can move on grass and bridges. They cannot move through water.
        """
        if cls.screen_to_tile(position) is None:
            return False

        if not cls.river_rectangle().collidepoint(position):
            return True

        return any(
            bridge.collidepoint(position)
            for bridge in cls.bridge_rectangles()
        )

    @classmethod
    def is_valid_deployment_tile(
        cls,
        tile: tuple[int, int],
        card: Card | None = None,
        destroyed_enemy_lanes: frozenset[str] = frozenset(),
        team: str = "blue",
    ) -> bool:
        """Check a tile against the selected card's placement behavior.

        Area spells may target any arena tile. Troops use the blue player's
        territory. Each destroyed enemy Princess Tower also unlocks the half of
        enemy territory in that lane nearest the river. ``None`` keeps the
        troop rule for callers that only need the default arena boundary.
        """
        column, row = tile
        # Validate coordinates before creating a rectangle from them.
        if not (0 <= column < GRID_COLUMNS and 0 <= row < GRID_ROWS):
            return False

        # With no card, use the ordinary troop boundary for generic arena
        # checks. A selected card replaces that default with its own policy.
        placement_rule = (
            PlacementRule.FRIENDLY_TERRITORY
            if card is None
            else card.placement_rule
        )

        # Spells such as Fireball and Zap target a coordinate directly, so
        # enemy territory, bridges, and river tiles are all legal destinations.
        if placement_rule is PlacementRule.ANYWHERE:
            return True

        if placement_rule is PlacementRule.FRIENDLY_TERRITORY:
            tile_rectangle = cls.tile_rectangle(tile)
            river = cls.river_rectangle()

            # Each side may deploy throughout its own half of the arena.
            in_friendly_half = (
                tile_rectangle.top >= river.bottom
                if team == "blue"
                else tile_rectangle.bottom <= river.top
            )
            if in_friendly_half:
                return True

            # Water and bridge tiles are never valid troop deployment tiles.
            intersects_river = (
                tile_rectangle.bottom > river.top
                and tile_rectangle.top < river.bottom
            )
            if intersects_river:
                return False

            # Destroyed Princess Towers unlock a mirrored forward rectangle.
            in_forward_depth = (
                tile_rectangle.top >= ENEMY_DEPLOYMENT_UNLOCK_TOP
                if team == "blue"
                else tile_rectangle.bottom
                <= ARENA_HEIGHT - ENEMY_DEPLOYMENT_UNLOCK_TOP
            )
            if not in_forward_depth:
                return False

            lane = (
                "left"
                if tile_rectangle.centerx < SCREEN_WIDTH // 2
                else "right"
            )
            return lane in destroyed_enemy_lanes

        # Every enum member should be handled above. Failing loudly makes newly
        # added placement policies impossible to forget in this central method.
        raise ValueError(f"Unsupported placement rule: {placement_rule}")

    @classmethod
    def restricted_deployment_tiles(
        cls,
        card: Card | None = None,
        destroyed_enemy_lanes: frozenset[str] = frozenset(),
        team: str = "blue",
    ) -> tuple[tuple[int, int], ...]:
        """Return every grid tile rejected for a specific selected card."""
        return tuple(
            (column, row)
            for row in range(GRID_ROWS)
            for column in range(GRID_COLUMNS)
            if not cls.is_valid_deployment_tile(
                (column, row),
                card,
                destroyed_enemy_lanes,
                team,
            )
        )

    def destroyed_enemy_princess_lanes(
        self,
        team: str = "blue",
    ) -> frozenset[str]:
        """Return lanes whose enemy Princess Tower has been destroyed.

        Placement reads this directly from combat health instead of maintaining
        a second flag, so the allowed tiles update on the same frame as death.
        """
        battle = getattr(self, "battle", None)
        if battle is None:
            return frozenset()

        destroyed_lanes = {
            (
                "left"
                if entity.position.x < SCREEN_WIDTH // 2
                else "right"
            )
            for entity in battle.entities
            if (
                entity.team != team
                and entity.tower_kind == "princess"
                and not entity.is_alive
            )
        }
        return frozenset(destroyed_lanes)

    @staticmethod
    def hand_card_rectangles() -> tuple[pygame.Rect, ...]:
        """Return the four card rectangles shared by drawing and mouse input."""
        return tuple(
            pygame.Rect(
                CARD_START_X + index * (CARD_WIDTH + CARD_GAP),
                HAND_HUD_TOP + 8,
                CARD_WIDTH,
                CARD_HEIGHT,
            )
            for index in range(4)
        )

    @classmethod
    def hand_index_at(cls, position: tuple[int, int]) -> int | None:
        """Return the hand slot clicked by the player, if there is one."""
        for index, card_rectangle in enumerate(cls.hand_card_rectangles()):
            if card_rectangle.collidepoint(position):
                return index
        return None

    def try_play_selected_card(self, tile: tuple[int, int]) -> bool:
        """Deploy the selected card and cycle only when every rule succeeds."""
        if getattr(self, "match_finished", False):
            return False

        if self.selected_card_index is None:
            return False

        team = getattr(self, "local_team", "blue")
        action = PlayCardAction(self.selected_card_index, tile)
        played = self.try_play_action(team, action)
        if played:
            self.selected_card_index = None
        return played

    def try_play_action(self, team: str, action: PlayCardAction) -> bool:
        """Validate and apply one controller request through shared game rules."""
        if getattr(self, "match_finished", False):
            return False
        if not 0 <= action.hand_slot < 4:
            return False

        if hasattr(self, "players"):
            player = self.players[team]
            card_cycle = player.card_cycle
            elixir = player.elixir
        else:
            # Backward-compatible path for small isolated unit-test viewers.
            card_cycle = self.card_cycle
            elixir = self.elixir

        card = card_cycle.hand[action.hand_slot]
        if not self.is_valid_deployment_tile(
            action.tile,
            card,
            self.destroyed_enemy_princess_lanes(team),
            team,
        ):
            return False

        if not elixir.spend(card.elixir_cost):
            return False

        self.deployments.append(Deployment(card, action.tile))
        battle = getattr(self, "battle", None)
        if battle is not None:
            battle.deploy_card(
                card,
                team,
                self.tile_rectangle(action.tile).center,
            )
        card_cycle.play(action.hand_slot)
        return True

    def legal_actions_for(self, team: str) -> tuple[PlayCardAction, ...]:
        """Enumerate every card/tile action currently legal for one team."""
        player = self.players[team]
        destroyed_lanes = self.destroyed_enemy_princess_lanes(team)
        actions = []

        for hand_slot, card in enumerate(player.card_cycle.hand):
            if player.elixir.amount + 1e-9 < card.elixir_cost:
                continue
            for row in range(GRID_ROWS):
                for column in range(GRID_COLUMNS):
                    tile = (column, row)
                    if self.is_valid_deployment_tile(
                        tile,
                        card,
                        destroyed_lanes,
                        team,
                    ):
                        actions.append(PlayCardAction(hand_slot, tile))

        return tuple(actions)

    def controller_context(self, team: str) -> ControllerContext:
        """Build the read-only snapshot supplied to a team's controller."""
        player = self.players[team]
        return ControllerContext(
            team=team,
            match_elapsed=self.match_elapsed,
            elixir=player.elixir.amount,
            hand=tuple(
                ControllerCard(
                    name=card.name,
                    elixir_cost=card.elixir_cost,
                    role=card.role,
                    card_type=card.card_type,
                )
                for card in player.card_cycle.hand
            ),
            legal_actions=self.legal_actions_for(team),
            crown_scores=self.battle.crown_scores,
        )

    def update_controllers(self, delta_seconds: float) -> None:
        """Ask non-human controllers for actions at a bounded decision rate."""
        decision_interval = 0.25
        for team, controller in self.controllers.items():
            if isinstance(controller, HumanController):
                continue

            self.controller_decision_elapsed[team] += delta_seconds
            if self.controller_decision_elapsed[team] < decision_interval:
                continue
            self.controller_decision_elapsed[team] %= decision_interval

            action = controller.choose_action(self.controller_context(team))
            if action is not None:
                self.try_play_action(team, action)

    def update_match_state(self, now_ms: int | None = None) -> None:
        """Advance regulation/overtime state and finish when a side wins."""
        if self.match_finished:
            return

        current_ms = self.simulation_now_ms() if now_ms is None else now_ms
        winner = self.battle.winning_team
        if winner is not None:
            self.finish_match(winner, current_ms)
            return

        scores = self.battle.crown_scores
        score_winner = None
        if scores["red"] > scores["blue"]:
            score_winner = "red"
        elif scores["blue"] > scores["red"]:
            score_winner = "blue"

        if self.overtime_active:
            # Overtime is sudden death: the first new Crown lead wins.
            if score_winner is not None:
                self.finish_match(score_winner, current_ms)
                return

            if self.overtime_started_at_ms is None:
                raise RuntimeError("Overtime is active without a start time")
            overtime_expired = remaining_match_seconds(
                self.overtime_started_at_ms,
                current_ms,
                OVERTIME_DURATION_SECONDS,
            ) == 0
            if overtime_expired:
                self.finish_match(None, current_ms)
            return

        regulation_expired = remaining_match_seconds(
            self.match_started_at,
            current_ms,
        ) == 0
        if not regulation_expired:
            return

        if score_winner is not None:
            self.finish_match(score_winner, current_ms)
            return

        # Tied regulation scores earn exactly two additional minutes. Anchor
        # the phase to the regulation deadline so a slow frame cannot shorten
        # or lengthen overtime.
        self.overtime_active = True
        self.overtime_started_at_ms = (
            self.match_started_at + MATCH_DURATION_SECONDS * 1000
        )
        self.overtime_notice_remaining = OVERTIME_NOTICE_SECONDS

    def simulation_now_ms(self) -> int:
        """Return the match time reached by completed simulation steps."""
        return self.match_started_at + round(self.match_elapsed * 1000)

    def update_simulation(self) -> None:
        """Advance every gameplay system by one fixed 50-millisecond step."""
        self.update_match_state()
        if self.match_finished:
            return

        if hasattr(self, "players"):
            for player in self.players.values():
                player.elixir.update(
                    FIXED_TIMESTEP_SECONDS,
                    self.match_elapsed,
                )
            self.update_controllers(FIXED_TIMESTEP_SECONDS)
        else:
            # Small unit-test viewers may only have the local Elixir object.
            self.elixir.update(FIXED_TIMESTEP_SECONDS, self.match_elapsed)
        self.update_elixir_multiplier_notice(FIXED_TIMESTEP_SECONDS)
        self.battle.update(FIXED_TIMESTEP_SECONDS)
        # Rounding stops tiny floating-point errors from moving time boundaries.
        self.match_elapsed = round(
            self.match_elapsed + FIXED_TIMESTEP_SECONDS,
            10,
        )
        self.update_match_state()
        self.update_overtime_notice(FIXED_TIMESTEP_SECONDS)

    def finish_match(self, winner: str | None, current_ms: int) -> None:
        """Store a terminal result and cancel all pending player interaction."""
        self.match_finished = True
        self.match_winner = winner
        self.match_finished_at_ms = current_ms
        # Clear interaction state so a held or selected card cannot be played
        # after the final tower-destroying attack lands.
        self.selected_card_index = None
        self.dragged_card_index = None
        self.drag_position = None
        # A match-ending attack can occur while the notice is active. Clear it
        # because finished matches freeze updates and would otherwise freeze the
        # temporary announcement on screen indefinitely.
        self.elixir_multiplier_notice = None
        self.elixir_multiplier_notice_remaining = 0.0
        self.overtime_notice_remaining = 0.0

    def update_overtime_notice(self, delta_seconds: float) -> None:
        """Count down the temporary regulation-to-overtime announcement."""
        if delta_seconds <= 0:
            return
        self.overtime_notice_remaining = max(
            0.0,
            self.overtime_notice_remaining - delta_seconds,
        )

    def update_elixir_multiplier_notice(self, delta_seconds: float) -> None:
        """Start or count down the announcement for an Elixir speed change.

        Comparing the multiplier before and after this update is safer than
        checking for an exact timestamp. A 60 FPS game will almost never land
        on exactly 120.000 seconds.
        """
        if delta_seconds <= 0:
            return

        # Existing notices lose the amount of time used by this update.
        self.elixir_multiplier_notice_remaining = max(
            0.0,
            self.elixir_multiplier_notice_remaining - delta_seconds,
        )
        if self.elixir_multiplier_notice_remaining == 0:
            self.elixir_multiplier_notice = None

        multiplier_before = self.elixir.multiplier_at(self.match_elapsed)
        multiplier_after = self.elixir.multiplier_at(
            self.match_elapsed + delta_seconds,
        )

        # A larger value means this update entered Double or Triple Elixir.
        if multiplier_after > multiplier_before:
            self.elixir_multiplier_notice = multiplier_after
            self.elixir_multiplier_notice_remaining = (
                ELIXIR_MULTIPLIER_NOTICE_SECONDS
            )

    def begin_card_drag(
        self,
        hand_index: int,
        position: tuple[int, int],
    ) -> None:
        """Select a hand card and begin following the pointer with it."""
        if not 0 <= hand_index < len(self.card_cycle.hand):
            raise IndexError("Hand index must be between 0 and 3")

        self.selected_card_index = hand_index
        self.dragged_card_index = hand_index
        self.drag_position = position

    def finish_card_drag(self, position: tuple[int, int]) -> bool:
        """Drop the dragged card, deploying it only on a valid arena tile."""
        if self.dragged_card_index is None:
            return False

        # Restore the dragged slot as the active selection before validation.
        self.selected_card_index = self.dragged_card_index
        self.dragged_card_index = None
        self.drag_position = None

        if position[1] >= HAND_HUD_TOP:
            return False

        tile = self.screen_to_tile(position)
        if tile is None:
            return False

        self.selected_tile = tile
        return self.try_play_selected_card(tile)

    def dragged_spell_preview(
        self,
    ) -> tuple[Card, tuple[int, int], int] | None:
        """Return the active spell and its pixel-space preview geometry.

        Spell statistics store their radius in arena tiles because that is how
        Clash Royale describes ranges. Drawing works in pixels, so this helper
        performs the conversion in one central place:

        ``pixel radius = spell tile radius * TILE_SIZE``

        ``None`` means there is no spell currently being dragged over the
        playable arena, so the drawing method has nothing to display.
        """
        if self.dragged_card_index is None or self.drag_position is None:
            return None

        card = self.card_cycle.hand[self.dragged_card_index]
        if card.spell_stats is None:
            # Troops use their normal dragged-card preview without an area circle.
            return None

        if self.screen_to_tile(self.drag_position) is None:
            # Do not paint a damage circle over the stadium sidelines or HUD.
            return None

        # Adding 0.5 gives ordinary half-up rounding: 2.5 tiles at 25 pixels
        # per tile becomes 63 pixels rather than Python's even-number rounding.
        radius_pixels = int(card.spell_stats.radius * TILE_SIZE + 0.5)
        return card, self.drag_position, radius_pixels

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def handle_events(self) -> None:
        """Handle keyboard and mouse input for one frame.

        Escape or Q quits. Space clears selections. Keys 1-4 or card clicks
        select a hand slot. An arena click then attempts to deploy that card.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif self.match_finished:
                    continue
                elif event.key == pygame.K_SPACE:
                    # Space cancels both selections without changing game state.
                    self.selected_tile = None
                    self.selected_card_index = None
                    self.dragged_card_index = None
                    self.drag_position = None
                elif (
                    isinstance(
                        self.controllers[self.local_team],
                        HumanController,
                    )
                    and pygame.K_1 <= event.key <= pygame.K_4
                ):
                    # K_1 maps to list index 0, K_2 to index 1, and so on.
                    self.selected_card_index = event.key - pygame.K_1
                    self.dragged_card_index = None
                    self.drag_position = None

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                logical_position = self.display_to_logical_position(event.pos)
                if self.match_finished:
                    if self.play_again_button_rectangle().collidepoint(
                        logical_position,
                    ):
                        self.reset_match()
                    continue

                if not isinstance(
                    self.controllers[self.local_team],
                    HumanController,
                ):
                    continue

                # Check the hand before converting the click into an arena tile.
                clicked_hand_index = self.hand_index_at(logical_position)
                if clicked_hand_index is not None:
                    self.begin_card_drag(clicked_hand_index, logical_position)
                    continue

                # The whole HUD is outside the playable arena, even though the
                # screen-to-grid conversion can mathematically produce a tile.
                if logical_position[1] >= HAND_HUD_TOP:
                    continue

                clicked_tile = self.screen_to_tile(logical_position)
                if clicked_tile is not None:
                    # Keep the outline even when the attempted play is invalid.
                    self.selected_tile = clicked_tile
                    self.try_play_selected_card(clicked_tile)

            elif event.type == pygame.MOUSEMOTION:
                if self.dragged_card_index is not None:
                    self.drag_position = self.display_to_logical_position(
                        event.pos,
                    )

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.dragged_card_index is not None:
                    self.finish_card_drag(
                        self.display_to_logical_position(event.pos),
                    )

    # ------------------------------------------------------------------
    # Draw the arena and buildings
    # ------------------------------------------------------------------
    def draw_arena(self) -> None:
        """Draw the grass, river, grid, and bridges.

        Bridges are drawn last so they appear on top of the water and grid.
        """
        self.screen.fill(STADIUM_FLOOR_COLOR)
        self.draw_stadium_sidelines()
        pygame.draw.rect(
            self.screen,
            ARENA_COLOR,
            (ARENA_LEFT, 0, ARENA_WIDTH, ARENA_HEIGHT),
        )

        for row in range(GRID_ROWS):
            for column in range(GRID_COLUMNS):
                if (column + row) % 2 == 0:
                    pygame.draw.rect(
                        self.screen,
                        ALTERNATE_TILE_COLOR,
                        self.tile_rectangle((column, row)),
                    )

        river = self.river_rectangle()
        pygame.draw.rect(self.screen, RIVER_COLOR, river)

        for y in range(river.top + 8, river.bottom, 14):
            pygame.draw.line(
                self.screen,
                RIVER_HIGHLIGHT_COLOR,
                (ARENA_LEFT, y),
                (ARENA_RIGHT, y),
                2,
            )

        for column in range(GRID_COLUMNS + 1):
            x = ARENA_LEFT + column * TILE_SIZE
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (x, 0),
                (x, ARENA_HEIGHT),
                1,
            )

        for row in range(GRID_ROWS + 1):
            y = row * TILE_SIZE
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (ARENA_LEFT, y),
                (ARENA_RIGHT, y),
                1,
            )

        self.draw_bridges()

    def draw_stadium_sidelines(self) -> None:
        """Draw decorative, non-playable stands outside both arena edges."""
        for side_left in (0, ARENA_RIGHT):
            side = pygame.Rect(
                side_left,
                0,
                STADIUM_BUFFER_WIDTH,
                SCREEN_HEIGHT,
            )
            pygame.draw.rect(self.screen, STADIUM_FLOOR_COLOR, side)

            # Horizontal bands suggest stepped wooden/stone spectator seating.
            for y in range(0, HAND_HUD_TOP, 32):
                pygame.draw.rect(
                    self.screen,
                    STADIUM_TIER_COLOR,
                    (side.left, y + 24, side.width, 5),
                )

            # Two columns of simple spectators keep the boundary lively without
            # introducing gameplay-like shapes inside the actual grid.
            for row, y in enumerate(range(18, HAND_HUD_TOP - 8, 32)):
                team_color = (
                    RED_TEAM_COLOR
                    if y < ARENA_HEIGHT // 2
                    else BLUE_TEAM_COLOR
                )
                for column, local_x in enumerate((18, 48)):
                    spectator_x = side.left + local_x
                    spectator_y = y + ((row + column) % 2) * 3
                    pygame.draw.circle(
                        self.screen,
                        (45, 39, 34),
                        (spectator_x + 1, spectator_y + 2),
                        7,
                    )
                    pygame.draw.rect(
                        self.screen,
                        team_color,
                        (spectator_x - 6, spectator_y + 2, 12, 8),
                        border_radius=3,
                    )
                    pygame.draw.circle(
                        self.screen,
                        SPECTATOR_SKIN_COLOR,
                        (spectator_x, spectator_y - 2),
                        4,
                    )

        # Raised rails clearly separate the decorative stands from legal tiles.
        pygame.draw.line(
            self.screen,
            STADIUM_RAIL_COLOR,
            (ARENA_LEFT - 3, 0),
            (ARENA_LEFT - 3, HAND_HUD_TOP),
            5,
        )
        pygame.draw.line(
            self.screen,
            STADIUM_RAIL_COLOR,
            (ARENA_RIGHT + 2, 0),
            (ARENA_RIGHT + 2, HAND_HUD_TOP),
            5,
        )

    def draw_bridges(self) -> None:
        """Draw a wooden bridge in each tower lane.

        Each bridge uses rectangles and lines for its shadow, wood, rails, and
        separate planks.
        """
        for bridge in self.bridge_rectangles():
            shadow = bridge.move(3, 4)
            pygame.draw.rect(
                self.screen,
                BRIDGE_EDGE_COLOR,
                shadow,
                border_radius=4,
            )
            pygame.draw.rect(
                self.screen,
                BRIDGE_PLANK_COLOR,
                bridge,
                border_radius=3,
            )

            pygame.draw.line(
                self.screen,
                BRIDGE_EDGE_COLOR,
                (bridge.left + 4, bridge.top),
                (bridge.left + 4, bridge.bottom),
                5,
            )
            pygame.draw.line(
                self.screen,
                BRIDGE_EDGE_COLOR,
                (bridge.right - 5, bridge.top),
                (bridge.right - 5, bridge.bottom),
                5,
            )

            for plank_y in range(bridge.top + 8, bridge.bottom, 12):
                pygame.draw.line(
                    self.screen,
                    BRIDGE_EDGE_COLOR,
                    (bridge.left + 5, plank_y),
                    (bridge.right - 5, plank_y),
                    2,
                )
                pygame.draw.line(
                    self.screen,
                    BRIDGE_PLANK_LIGHT_COLOR,
                    (bridge.left + 7, plank_y + 2),
                    (bridge.right - 7, plank_y + 2),
                    1,
                )

    @staticmethod
    def team_colors(team: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return the main and highlight colors for a tower's team."""
        if team == "red":
            return RED_TEAM_COLOR, RED_TEAM_LIGHT_COLOR
        return BLUE_TEAM_COLOR, BLUE_TEAM_LIGHT_COLOR

    def draw_princess_tower(self, tower: Tower) -> None:
        """Draw one princess tower.

        Every shape starts from ``tower.center``. The same code can therefore
        draw all four princess towers.
        """
        center_x, center_y = tower.center
        team_color, team_light = self.team_colors(tower.team)

        shadow = pygame.Rect(center_x - 36, center_y - 27, 72, 64)
        pygame.draw.rect(self.screen, STONE_SHADOW_COLOR, shadow, border_radius=9)

        base = pygame.Rect(center_x - 32, center_y - 31, 64, 60)
        pygame.draw.rect(
            self.screen,
            STONE_COLOR,
            base,
            border_radius=7,
        )

        # Chunky stone blocks keep the silhouette close to the in-game towers.
        for block_x in (base.x + 2, base.centerx - 8, base.right - 18):
            pygame.draw.rect(
                self.screen,
                STONE_HIGHLIGHT_COLOR,
                (block_x, base.y - 9, 16, 18),
                border_radius=3,
            )

        band = pygame.Rect(base.x - 2, center_y + 8, base.width + 4, 13)
        pygame.draw.rect(self.screen, team_color, band, border_radius=3)
        pygame.draw.line(
            self.screen,
            team_light,
            (band.left + 5, band.top + 3),
            (band.right - 5, band.top + 3),
            3,
        )

        opening = pygame.Rect(center_x - 11, center_y - 15, 22, 27)
        pygame.draw.ellipse(self.screen, TOWER_OPENING_COLOR, opening)
        pygame.draw.rect(
            self.screen,
            TOWER_OPENING_COLOR,
            (opening.x, opening.centery, opening.width, opening.height // 2),
        )

        roof = [
            (center_x - 27, center_y - 29),
            (center_x, center_y - 52),
            (center_x + 27, center_y - 29),
        ]
        pygame.draw.polygon(self.screen, team_color, roof)
        pygame.draw.line(
            self.screen,
            team_light,
            roof[0],
            roof[1],
            4,
        )
        pygame.draw.line(
            self.screen,
            team_light,
            roof[1],
            roof[2],
            4,
        )

        # Gold crown badge.
        pygame.draw.circle(
            self.screen,
            CROWN_SHADOW_COLOR,
            (center_x, center_y + 14),
            9,
        )
        pygame.draw.circle(self.screen, CROWN_COLOR, (center_x, center_y + 13), 8)
        pygame.draw.polygon(
            self.screen,
            team_color,
            [
                (center_x - 5, center_y + 15),
                (center_x - 4, center_y + 8),
                (center_x, center_y + 12),
                (center_x + 4, center_y + 8),
                (center_x + 5, center_y + 15),
            ],
        )

    def draw_king_tower(self, tower: Tower) -> None:
        """Draw one king tower.

        It is wider than a princess tower and has a bigger doorway and crown.
        """
        center_x, center_y = tower.center
        team_color, team_light = self.team_colors(tower.team)

        shadow = pygame.Rect(center_x - 46, center_y - 34, 92, 78)
        pygame.draw.rect(self.screen, STONE_SHADOW_COLOR, shadow, border_radius=11)
        body = pygame.Rect(center_x - 42, center_y - 37, 84, 70)
        pygame.draw.rect(self.screen, STONE_COLOR, body, border_radius=8)

        battlement_y = body.y - 11
        for block_x in (body.x, body.x + 23, body.right - 18):
            pygame.draw.rect(
                self.screen,
                STONE_HIGHLIGHT_COLOR,
                (block_x, battlement_y, 18, 22),
                border_radius=3,
            )

        pygame.draw.rect(
            self.screen,
            team_color,
            (body.x - 3, center_y + 9, body.width + 6, 16),
            border_radius=4,
        )
        pygame.draw.line(
            self.screen,
            team_light,
            (body.x + 3, center_y + 13),
            (body.right - 3, center_y + 13),
            4,
        )

        doorway = pygame.Rect(center_x - 14, center_y - 18, 28, 35)
        pygame.draw.ellipse(self.screen, TOWER_OPENING_COLOR, doorway)
        pygame.draw.rect(
            self.screen,
            TOWER_OPENING_COLOR,
            (doorway.x, doorway.centery, doorway.width, doorway.height // 2),
        )

        crown_y = center_y - 51
        crown = [
            (center_x - 19, crown_y + 19),
            (center_x - 19, crown_y),
            (center_x - 8, crown_y + 11),
            (center_x, crown_y - 3),
            (center_x + 8, crown_y + 11),
            (center_x + 19, crown_y),
            (center_x + 19, crown_y + 19),
        ]
        pygame.draw.polygon(self.screen, CROWN_SHADOW_COLOR, crown)
        pygame.draw.polygon(
            self.screen,
            CROWN_COLOR,
            [(x, y - 2) for x, y in crown],
        )
        pygame.draw.circle(self.screen, team_color, (center_x, crown_y + 12), 4)

    def draw_towers(self) -> None:
        """Draw living Crown Towers, rubble, activation state, and health."""
        for entity in self.battle.entities:
            if entity.tower_kind is None:
                continue

            center = (round(entity.position.x), round(entity.position.y))
            tower = Tower(entity.tower_kind, entity.team, center)

            if not entity.is_alive:
                pygame.draw.circle(
                    self.screen,
                    TOWER_OPENING_COLOR,
                    center,
                    round(entity.radius),
                )
                pygame.draw.line(
                    self.screen,
                    STONE_SHADOW_COLOR,
                    (center[0] - 16, center[1] - 12),
                    (center[0] + 17, center[1] + 13),
                    6,
                )
                continue

            if entity.tower_kind == "king":
                self.draw_king_tower(tower)
            else:
                self.draw_princess_tower(tower)

            self.draw_health_bar(entity, width=58)

    def draw_health_bar(
        self,
        entity: BattleEntity,
        *,
        width: int = 34,
    ) -> None:
        """Draw a compact health bar immediately above one living entity."""
        if not entity.is_alive:
            return

        height = 5
        center_x = round(entity.position.x)
        top = round(entity.position.y - entity.radius - 12)
        background = pygame.Rect(center_x - width // 2, top, width, height)
        ratio = max(0.0, min(1.0, entity.health / entity.max_health))
        remaining_width = round((width - 2) * ratio)

        pygame.draw.rect(
            self.screen,
            (37, 42, 44),
            background,
            border_radius=2,
        )
        if remaining_width > 0:
            health_color = (
                BLUE_TEAM_LIGHT_COLOR
                if entity.team == "blue"
                else RED_TEAM_LIGHT_COLOR
            )
            pygame.draw.rect(
                self.screen,
                health_color,
                (
                    background.x + 1,
                    background.y + 1,
                    remaining_width,
                    height - 2,
                ),
                border_radius=1,
            )

    def draw_units(self) -> None:
        """Draw each living troop as a distinct, state-driven arena marker."""
        unit_colors = {
            "Knight": (84, 132, 201),
            "Archers": (130, 212, 160),
            "Giant": (205, 145, 82),
            "Mini P.E.K.K.A": (83, 93, 119),
            "Musketeer": (135, 101, 188),
            "Skeletons": (226, 224, 211),
        }

        for entity in self.battle.entities:
            if entity.is_building or not entity.is_alive:
                continue

            base_name = entity.name.rsplit(" ", 1)[0]
            if base_name not in unit_colors:
                base_name = entity.name
            center = (round(entity.position.x), round(entity.position.y))
            team_outline = (
                BLUE_TEAM_COLOR if entity.team == "blue" else RED_TEAM_COLOR
            )

            pygame.draw.circle(
                self.screen,
                team_outline,
                center,
                round(entity.radius) + 3,
            )
            pygame.draw.circle(
                self.screen,
                unit_colors.get(base_name, STONE_COLOR),
                center,
                round(entity.radius),
            )

            initials = "".join(
                word[0]
                for word in base_name.replace(".", "").split()
            )[:2]
            label = self.card_font.render(initials, True, TOWER_OPENING_COLOR)
            self.screen.blit(label, label.get_rect(center=center))
            self.draw_health_bar(entity)

    def draw_projectiles(self) -> None:
        """Draw arrows and ranged shots as small moving rectangles."""
        for projectile in self.battle.projectiles:
            center = (
                round(projectile.position.x),
                round(projectile.position.y),
            )
            pygame.draw.rect(
                self.screen,
                projectile.color,
                pygame.Rect(center[0] - 5, center[1] - 2, 10, 4),
                border_radius=2,
            )

    # ------------------------------------------------------------------
    # Draw selection and game information
    # ------------------------------------------------------------------
    def draw_restricted_placement_tiles(self) -> None:
        """Tint unavailable placement tiles while a hand card is selected."""
        if self.selected_card_index is None:
            return

        selected_card = self.card_cycle.hand[self.selected_card_index]
        overlay = pygame.Surface(
            (SCREEN_WIDTH, SCREEN_HEIGHT),
            pygame.SRCALPHA,
        )

        for tile in self.restricted_deployment_tiles(
            selected_card,
            self.destroyed_enemy_princess_lanes(self.local_team),
            self.local_team,
        ):
            # A one-pixel inset preserves clear grid lines under the tint.
            restricted_area = self.tile_rectangle(tile).inflate(-2, -2)
            pygame.draw.rect(
                overlay,
                RESTRICTED_TILE_COLOR,
                restricted_area,
            )

        self.screen.blit(overlay, (0, 0))

    def draw_tile_highlights(self) -> None:
        """Show the tile under the mouse and the selected tile.

        The mouse highlight is see-through. The selected tile uses a solid
        border and stays selected until it is cleared.
        """
        hovered_tile = self.screen_to_tile(
            self.display_to_logical_position(pygame.mouse.get_pos()),
        )

        if hovered_tile is not None:
            hover_surface = pygame.Surface(
                (TILE_SIZE, TILE_SIZE),
                pygame.SRCALPHA,
            )
            hover_surface.fill(HOVER_COLOR)
            self.screen.blit(
                hover_surface,
                self.tile_rectangle(hovered_tile).topleft,
            )

        if self.selected_tile is not None:
            pygame.draw.rect(
                self.screen,
                SELECTED_COLOR,
                self.tile_rectangle(self.selected_tile),
                3,
            )

    def draw_match_timer(self) -> None:
        """Draw the match timer in the top-right corner.

        The timer uses completed game updates, so it always agrees with combat.
        The numbers turn red for the last ten seconds. At zero, the match-over
        message appears.
        """
        timer_now_ms = (
            self.match_finished_at_ms
            if self.match_finished_at_ms is not None
            else self.simulation_now_ms()
        )
        if self.overtime_active:
            if self.overtime_started_at_ms is None:
                raise RuntimeError("Overtime is active without a start time")
            seconds_left = remaining_match_seconds(
                self.overtime_started_at_ms,
                timer_now_ms,
                OVERTIME_DURATION_SECONDS,
            )
            timer_label = "Overtime"
        else:
            seconds_left = remaining_match_seconds(
                self.match_started_at,
                timer_now_ms,
            )
            timer_label = "Time left"

        panel = pygame.Rect(SCREEN_WIDTH - 111, 6, 105, 63)
        shadow = pygame.Surface((panel.width + 6, panel.height + 6), pygame.SRCALPHA)
        pygame.draw.rect(
            shadow,
            TIMER_SHADOW_COLOR,
            shadow.get_rect(),
            border_radius=7,
        )
        self.screen.blit(shadow, (panel.x + 3, panel.y + 4))

        pygame.draw.rect(self.screen, TIMER_BORDER_COLOR, panel, border_radius=6)
        pygame.draw.rect(
            self.screen,
            TIMER_PANEL_COLOR,
            panel.inflate(-6, -6),
            border_radius=4,
        )

        label = self.timer_label_font.render(timer_label, True, TEXT_COLOR)
        label_position = (
            panel.centerx - label.get_width() // 2,
            panel.y + 5,
        )
        self.screen.blit(label, label_position)

        timer_color = TIMER_URGENT_COLOR if seconds_left <= 10 else TEXT_COLOR
        time_text = self.timer_font.render(
            format_match_time(seconds_left),
            True,
            timer_color,
        )
        time_position = (
            panel.centerx - time_text.get_width() // 2,
            panel.y + 22,
        )
        self.screen.blit(time_text, time_position)

        if self.match_finished:
            self.draw_match_over()

    def draw_match_over(self) -> None:
        """Show the winner, final score, and Play Again action."""
        title_text, score_text = self.match_result_text()
        title_color = (
            BLUE_TEAM_LIGHT_COLOR
            if self.match_winner == "blue"
            else RED_TEAM_LIGHT_COLOR
            if self.match_winner == "red"
            else CROWN_COLOR
        )
        title = self.match_over_font.render(title_text, True, title_color)
        score = self.font.render(score_text, True, TEXT_COLOR)
        button_label = self.font.render("PLAY AGAIN", True, TEXT_COLOR)

        # Dim only the battlefield; the permanent HUD and stadium counters stay
        # visible behind the final result.
        dimmer = pygame.Surface((ARENA_WIDTH, ARENA_HEIGHT), pygame.SRCALPHA)
        dimmer.fill((0, 0, 0, 85))
        self.screen.blit(dimmer, (ARENA_LEFT, 0))

        panel = pygame.Rect(0, 0, 360, 220)
        panel.center = (SCREEN_WIDTH // 2, ARENA_HEIGHT // 2)
        overlay = pygame.Surface(panel.size, pygame.SRCALPHA)
        pygame.draw.rect(
            overlay,
            (8, 10, 14, 235),
            overlay.get_rect(),
            border_radius=12,
        )
        pygame.draw.rect(
            overlay,
            TIMER_BORDER_COLOR,
            overlay.get_rect(),
            3,
            border_radius=12,
        )
        overlay.blit(title, title.get_rect(center=(panel.width // 2, 54)))
        overlay.blit(score, score.get_rect(center=(panel.width // 2, 105)))
        self.screen.blit(overlay, panel.topleft)

        button = self.play_again_button_rectangle()
        pygame.draw.rect(
            self.screen,
            BLUE_TEAM_COLOR,
            button,
            border_radius=9,
        )
        pygame.draw.rect(
            self.screen,
            BLUE_TEAM_LIGHT_COLOR,
            button,
            3,
            border_radius=9,
        )
        self.screen.blit(
            button_label,
            button_label.get_rect(center=button.center),
        )

    def draw_crown_scores(self) -> None:
        """Draw compact red and blue Crown counters on the arena's left edge."""
        scores = self.battle.crown_scores
        counter_width = 54
        counter_height = 34
        counter_x = 6
        counter_positions = {
            "red": RIVER_TOP - counter_height - 8,
            "blue": RIVER_TOP + RIVER_HEIGHT + 8,
        }

        for team in ("red", "blue"):
            main_color, light_color = self.team_colors(team)
            panel = pygame.Rect(
                counter_x,
                counter_positions[team],
                counter_width,
                counter_height,
            )

            # A dark offset keeps this small HUD readable over grass and grid.
            shadow_surface = pygame.Surface(panel.size, pygame.SRCALPHA)
            pygame.draw.rect(
                shadow_surface,
                (20, 24, 28, 85),
                shadow_surface.get_rect(),
                border_radius=7,
            )
            self.screen.blit(shadow_surface, panel.move(2, 3).topleft)

            # Draw the team-colored box on an alpha surface so the stadium and
            # spectators remain visible beneath it. The crown and number are
            # drawn afterward at full opacity for legibility.
            panel_surface = pygame.Surface(panel.size, pygame.SRCALPHA)
            pygame.draw.rect(
                panel_surface,
                (*main_color, 155),
                panel_surface.get_rect(),
                border_radius=7,
            )
            pygame.draw.rect(
                panel_surface,
                (*light_color, 205),
                panel_surface.get_rect(),
                2,
                border_radius=7,
            )
            self.screen.blit(panel_surface, panel.topleft)

            # The three-point polygon mirrors Clash Royale's gold crown symbol.
            crown_x = panel.x + 15
            crown_y = panel.centery
            crown_points = [
                (crown_x - 9, crown_y - 7),
                (crown_x - 4, crown_y - 2),
                (crown_x, crown_y - 9),
                (crown_x + 4, crown_y - 2),
                (crown_x + 9, crown_y - 7),
                (crown_x + 7, crown_y + 7),
                (crown_x - 7, crown_y + 7),
            ]
            pygame.draw.polygon(self.screen, CROWN_COLOR, crown_points)
            pygame.draw.line(
                self.screen,
                CROWN_SHADOW_COLOR,
                (crown_x - 7, crown_y + 3),
                (crown_x + 7, crown_y + 3),
                2,
            )

            score_text = self.font.render(
                str(scores[team]),
                True,
                TEXT_COLOR,
            )
            self.screen.blit(
                score_text,
                score_text.get_rect(
                    center=(panel.x + 40, panel.centery + 1),
                ),
            )

    def draw_card(
        self,
        card: Card,
        rectangle: pygame.Rect,
        *,
        selected: bool = False,
        key_number: int | None = None,
        show_affordability: bool = True,
    ) -> None:
        """Draw one placeholder card with cost and affordability feedback.

        Only the four playable hand cards use affordability dimming. The next
        preview cannot be played yet, so dimming it would incorrectly suggest
        that clicking it should do something.
        """
        # The tiny tolerance matches ElixirMeter.spend and avoids a floating-point
        # value such as 2.999999999 making a three-Elixir card appear disabled.
        affordable = (
            not show_affordability
            or self.elixir.amount + 1e-9 >= card.elixir_cost
        )
        background = (
            CARD_BACKGROUND_COLOR if affordable else CARD_DISABLED_COLOR
        )
        border = SELECTED_COLOR if selected else CARD_BORDER_COLOR

        # Draw the card face first; labels and badges are layered on top of it.
        pygame.draw.rect(self.screen, background, rectangle, border_radius=7)

        if key_number is not None:
            # The printed number teaches the matching 1-4 keyboard shortcut.
            key_label = self.card_font.render(str(key_number), True, TEXT_COLOR)
            self.screen.blit(key_label, (rectangle.x + 6, rectangle.y + 5))

        # The cost badge is a temporary stand-in for the familiar Elixir icon.
        cost_center = (rectangle.right - 13, rectangle.y + 13)
        pygame.draw.circle(self.screen, ELIXIR_COLOR, cost_center, 11)
        cost_label = self.card_cost_font.render(
            str(card.elixir_cost),
            True,
            TEXT_COLOR,
        )
        self.screen.blit(cost_label, cost_label.get_rect(center=cost_center))

        # Split longer names over two centered lines instead of shrinking them.
        words = card.name.split()
        if len(words) > 1:
            name_lines = (words[0], " ".join(words[1:]))
        else:
            name_lines = (card.name,)

        first_line_y = rectangle.centery - 5 * len(name_lines)
        for line_index, line in enumerate(name_lines):
            name_label = self.card_font.render(line, True, TEXT_COLOR)
            name_rectangle = name_label.get_rect(
                center=(
                    rectangle.centerx,
                    first_line_y + line_index * 17,
                )
            )
            self.screen.blit(name_label, name_rectangle)

        type_label = self.card_font.render(
            card.card_type.title(),
            True,
            CARD_BORDER_COLOR,
        )
        self.screen.blit(
            type_label,
            type_label.get_rect(
                center=(rectangle.centerx, rectangle.bottom - 11),
            ),
        )

        if not affordable:
            # A per-card alpha surface dims the face, text, type, and Elixir badge
            # together. It is recreated here because each card rectangle may have
            # a different size (the next-card preview is shorter).
            dim_overlay = pygame.Surface(rectangle.size, pygame.SRCALPHA)
            dim_overlay.fill(CARD_DISABLED_OVERLAY_COLOR)
            self.screen.blit(dim_overlay, rectangle.topleft)

        # Draw the border last so a selected-but-unaffordable card remains easy to
        # identify even though its contents are deliberately darkened.
        pygame.draw.rect(
            self.screen,
            border,
            rectangle,
            4 if selected else 2,
            border_radius=7,
        )

    def draw_card_hand(self) -> None:
        """Draw four selectable cards and preview the next queued card."""
        # This translucent panel separates the hand from the active arena.
        hud = pygame.Surface((SCREEN_WIDTH, 100), pygame.SRCALPHA)
        hud.fill((18, 22, 30, 225))
        self.screen.blit(hud, (0, HAND_HUD_TOP))

        # zip pairs each live hand position with the matching clickable rectangle.
        for index, (card, rectangle) in enumerate(
            zip(self.card_cycle.hand, self.hand_card_rectangles())
        ):
            # The floating drag preview is the card itself, so do not leave a
            # duplicate copy behind in its hand slot.
            if index == self.dragged_card_index:
                pygame.draw.rect(
                    self.screen,
                    CARD_DISABLED_COLOR,
                    rectangle,
                    border_radius=7,
                )
                pygame.draw.rect(
                    self.screen,
                    CARD_BORDER_COLOR,
                    rectangle,
                    1,
                    border_radius=7,
                )
                continue

            self.draw_card(
                card,
                rectangle,
                selected=index == self.selected_card_index,
                key_number=index + 1,
            )

        # The next preview is deliberately smaller and has no keyboard number.
        next_label = self.card_font.render("NEXT", True, TEXT_COLOR)
        next_rectangle = pygame.Rect(
            NEXT_CARD_X,
            HAND_HUD_TOP + 25,
            CARD_WIDTH,
            CARD_HEIGHT - 17,
        )
        self.screen.blit(
            next_label,
            next_label.get_rect(
                center=(next_rectangle.centerx, HAND_HUD_TOP + 14),
            ),
        )
        self.draw_card(
            self.card_cycle.next_card,
            next_rectangle,
            show_affordability=False,
        )

    def draw_dragged_card(self) -> None:
        """Draw the active card above the pointer while it is being dragged."""
        if self.dragged_card_index is None or self.drag_position is None:
            return

        card = self.card_cycle.hand[self.dragged_card_index]
        preview = pygame.Rect(0, 0, CARD_WIDTH, CARD_HEIGHT)
        preview.center = (
            self.drag_position[0],
            self.drag_position[1] - CARD_HEIGHT // 2,
        )
        preview.clamp_ip(self.screen.get_rect())

        shadow = pygame.Surface(
            (preview.width + 10, preview.height + 10),
            pygame.SRCALPHA,
        )
        pygame.draw.rect(
            shadow,
            (0, 0, 0, 105),
            shadow.get_rect(),
            border_radius=10,
        )
        self.screen.blit(shadow, (preview.x + 5, preview.y + 7))

        self.draw_card(
            card,
            preview,
            selected=True,
            show_affordability=False,
        )

    def draw_spell_radius_preview(self) -> None:
        """Draw the damage area beneath a spell while its card is dragged."""
        preview = self.dragged_spell_preview()
        if preview is None:
            return

        card, center, radius_pixels = preview
        hovered_tile = self.screen_to_tile(center)
        if hovered_tile is None:
            return

        # Ask the same placement validator used by the actual drop. This keeps
        # preview color and deployment behavior synchronized if a future spell
        # receives a restricted placement rule.
        is_valid = self.is_valid_deployment_tile(
            hovered_tile,
            card,
            self.destroyed_enemy_princess_lanes(self.local_team),
            self.local_team,
        )
        fill_color = (
            SPELL_RADIUS_VALID_FILL
            if is_valid
            else SPELL_RADIUS_INVALID_FILL
        )
        border_color = (
            SPELL_RADIUS_VALID_BORDER
            if is_valid
            else SPELL_RADIUS_INVALID_BORDER
        )

        # Draw into an arena-sized transparent layer. Its dimensions clip the
        # circle at arena edges so it never darkens the HUD or stadium seating.
        radius_layer = pygame.Surface(
            (ARENA_WIDTH, ARENA_HEIGHT),
            pygame.SRCALPHA,
        )
        local_center = (center[0] - ARENA_LEFT, center[1])
        pygame.draw.circle(
            radius_layer,
            fill_color,
            local_center,
            radius_pixels,
        )
        pygame.draw.circle(
            radius_layer,
            border_color,
            local_center,
            radius_pixels,
            3,
        )
        # The center marker shows the exact point used when the mouse is released.
        pygame.draw.circle(radius_layer, border_color, local_center, 4)
        self.screen.blit(radius_layer, (ARENA_LEFT, 0))

    def draw_elixir_bar(self) -> None:
        """Draw the purple Elixir bar at the bottom of the screen.

        The ten boxes show the ten available Elixir units. The current box can
        be partly full because Elixir grows smoothly. The number on the left
        shows the amount the player can spend right now.
        """
        hud = pygame.Surface((SCREEN_WIDTH, 53), pygame.SRCALPHA)
        hud.fill((18, 13, 28, 215))
        self.screen.blit(hud, (0, SCREEN_HEIGHT - 53))

        bar_rect = pygame.Rect(
            ARENA_LEFT + 54,
            SCREEN_HEIGHT - 31,
            382,
            22,
        )
        pygame.draw.rect(
            self.screen,
            ELIXIR_FRAME_COLOR,
            bar_rect.inflate(6, 6),
            border_radius=8,
        )

        gap = 2
        cell_width = (bar_rect.width - gap * 9) / 10
        for index in range(10):
            # Round each box edge so all ten boxes fit the bar exactly.
            cell_x = round(bar_rect.x + index * (cell_width + gap))
            next_x = round(bar_rect.x + (index + 1) * cell_width + index * gap)
            cell = pygame.Rect(cell_x, bar_rect.y, next_x - cell_x, bar_rect.height)
            pygame.draw.rect(
                self.screen,
                ELIXIR_EMPTY_COLOR,
                cell,
                border_radius=3,
            )

            fill_fraction = max(0.0, min(1.0, self.elixir.amount - index))
            if fill_fraction > 0:
                fill = cell.copy()
                fill.width = max(1, round(cell.width * fill_fraction))
                pygame.draw.rect(
                    self.screen,
                    ELIXIR_COLOR,
                    fill,
                    border_radius=3,
                )
                if fill.width > 5:
                    pygame.draw.line(
                        self.screen,
                        ELIXIR_HIGHLIGHT_COLOR,
                        (fill.left + 3, fill.top + 4),
                        (fill.right - 3, fill.top + 4),
                        2,
                    )

        # The teardrop badge mirrors the large count at the left of the real HUD.
        badge_center = (ARENA_LEFT + 28, SCREEN_HEIGHT - 21)
        pygame.draw.polygon(
            self.screen,
            ELIXIR_FRAME_COLOR,
            [
                (badge_center[0], badge_center[1] - 28),
                (badge_center[0] - 22, badge_center[1] + 1),
                (badge_center[0] + 22, badge_center[1] + 1),
            ],
        )
        pygame.draw.circle(self.screen, ELIXIR_FRAME_COLOR, badge_center, 24)
        pygame.draw.polygon(
            self.screen,
            ELIXIR_COLOR,
            [
                (badge_center[0], badge_center[1] - 23),
                (badge_center[0] - 18, badge_center[1] + 1),
                (badge_center[0] + 18, badge_center[1] + 1),
            ],
        )
        pygame.draw.circle(self.screen, ELIXIR_COLOR, badge_center, 19)
        pygame.draw.arc(
            self.screen,
            ELIXIR_HIGHLIGHT_COLOR,
            pygame.Rect(ARENA_LEFT + 15, SCREEN_HEIGHT - 38, 26, 25),
            1.7,
            3.8,
            3,
        )

        value = self.elixir_font.render(
            str(math.floor(self.elixir.amount + 1e-9)),
            True,
            TEXT_COLOR,
        )
        value_shadow = self.elixir_font.render(
            str(math.floor(self.elixir.amount + 1e-9)),
            True,
            ELIXIR_DARK_COLOR,
        )
        value_rect = value.get_rect(center=(badge_center[0], badge_center[1] + 1))
        self.screen.blit(value_shadow, value_rect.move(2, 2))
        self.screen.blit(value, value_rect)

        multiplier = self.elixir.multiplier_at(self.match_elapsed)
        if multiplier > 1:
            multiplier_text = self.font.render(
                f"x{multiplier} Elixir",
                True,
                ELIXIR_HIGHLIGHT_COLOR,
            )
            self.screen.blit(
                multiplier_text,
                (SCREEN_WIDTH - multiplier_text.get_width() - 13, SCREEN_HEIGHT - 53),
            )

        if self.elixir.full_notice_remaining > 0:
            # The ElixirMeter decides how long this message stays visible.
            notice = self.elixir_notice_font.render(
                "Elixir bar is full!",
                True,
                TEXT_COLOR,
            )
            shadow = self.elixir_notice_font.render(
                "Elixir bar is full!",
                True,
                ELIXIR_DARK_COLOR,
            )
            notice_rect = notice.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 66),
            )
            self.screen.blit(shadow, notice_rect.move(2, 2))
            self.screen.blit(notice, notice_rect)

    def draw_elixir_multiplier_notice(self) -> None:
        """Draw a temporary center-screen Double/Triple Elixir announcement."""
        if (
            self.elixir_multiplier_notice is None
            or self.elixir_multiplier_notice_remaining <= 0
        ):
            return

        # Fade only during the final half-second. The message stays fully legible
        # for most of its lifetime instead of fading immediately after appearing.
        fade_seconds = 0.5
        alpha_fraction = min(
            1.0,
            self.elixir_multiplier_notice_remaining / fade_seconds,
        )
        panel_alpha = round(255 * alpha_fraction)

        title = self.elixir_multiplier_font.render(
            f"{self.elixir_multiplier_notice}x ELIXIR",
            True,
            TEXT_COLOR,
        )
        subtitle = self.font.render(
            "ELIXIR GENERATION INCREASED",
            True,
            ELIXIR_HIGHLIGHT_COLOR,
        )

        panel = pygame.Surface((330, 92), pygame.SRCALPHA)
        pygame.draw.rect(
            panel,
            (*ELIXIR_NOTICE_PANEL_COLOR[:3], min(
                ELIXIR_NOTICE_PANEL_COLOR[3],
                panel_alpha,
            )),
            panel.get_rect(),
            border_radius=12,
        )
        pygame.draw.rect(
            panel,
            (*ELIXIR_NOTICE_BORDER_COLOR, panel_alpha),
            panel.get_rect(),
            4,
            border_radius=12,
        )

        # Applying alpha to the rendered text lets the complete announcement
        # disappear together during the last half-second.
        title.set_alpha(panel_alpha)
        subtitle.set_alpha(panel_alpha)
        panel.blit(title, title.get_rect(center=(panel.get_width() // 2, 33)))
        panel.blit(
            subtitle,
            subtitle.get_rect(center=(panel.get_width() // 2, 68)),
        )
        panel_rectangle = panel.get_rect(
            center=(SCREEN_WIDTH // 2, ARENA_HEIGHT // 2),
        )
        self.screen.blit(panel, panel_rectangle)

    def draw_overtime_notice(self) -> None:
        """Announce the start of overtime in a centered black rectangle."""
        if self.overtime_notice_remaining <= 0:
            return

        fade_seconds = 0.5
        alpha_fraction = min(
            1.0,
            self.overtime_notice_remaining / fade_seconds,
        )
        panel_alpha = round(255 * alpha_fraction)

        title = self.elixir_multiplier_font.render(
            "OVERTIME!",
            True,
            OVERTIME_NOTICE_TEXT_COLOR,
        )
        subtitle = self.font.render(
            "SUDDEN DEATH - 2:00",
            True,
            OVERTIME_NOTICE_TEXT_COLOR,
        )
        title.set_alpha(panel_alpha)
        subtitle.set_alpha(panel_alpha)

        panel = pygame.Surface((330, 92), pygame.SRCALPHA)
        pygame.draw.rect(
            panel,
            (
                *OVERTIME_NOTICE_PANEL_COLOR[:3],
                min(OVERTIME_NOTICE_PANEL_COLOR[3], panel_alpha),
            ),
            panel.get_rect(),
            border_radius=10,
        )
        pygame.draw.rect(
            panel,
            (*OVERTIME_NOTICE_TEXT_COLOR, panel_alpha),
            panel.get_rect(),
            3,
            border_radius=10,
        )
        panel.blit(title, title.get_rect(center=(panel.get_width() // 2, 33)))
        panel.blit(
            subtitle,
            subtitle.get_rect(center=(panel.get_width() // 2, 68)),
        )
        self.screen.blit(
            panel,
            panel.get_rect(center=(SCREEN_WIDTH // 2, ARENA_HEIGHT // 2)),
        )

    # ------------------------------------------------------------------
    # Draw frames and run the game
    # ------------------------------------------------------------------
    def draw(self) -> None:
        """Draw one complete frame.

        The arena is drawn first. Highlights, towers, and game information are
        drawn afterward so they appear on top.
        """
        self.draw_arena()
        self.draw_restricted_placement_tiles()
        self.draw_towers()
        self.draw_units()
        self.draw_projectiles()
        self.draw_tile_highlights()
        # Draw after combat objects so the full affected area remains readable.
        # The HUD is drawn later and therefore stays visually above this circle.
        self.draw_spell_radius_preview()
        self.draw_match_timer()
        self.draw_crown_scores()
        self.draw_overtime_notice()
        # The temporary announcement belongs above arena action but below the
        # draggable card, which should always remain attached to the pointer.
        self.draw_elixir_multiplier_notice()
        self.draw_card_hand()
        self.draw_elixir_bar()
        self.draw_dragged_card()
        pygame.transform.smoothscale(
            self.screen,
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            self.display_surface,
        )
        pygame.display.flip()

    def run(self) -> None:
        """Keep updating and drawing until the window closes.

        Drawing can run at different speeds, but gameplay always moves forward
        in exact 50-millisecond steps. Slow frames run several updates to catch
        up. Fast frames save their extra time until the next update is ready.
        """
        while self.running:
            frame_ms = self.clock.tick(FPS)
            self.handle_events()

            # A finished match remains visible and responsive to quit input,
            # but combat, Elixir generation, and the match clock are frozen.
            if not self.match_finished:
                self.update_match_state()
                step_count = self.fixed_timestep.add_frame_time(frame_ms)
                for _ in range(step_count):
                    self.update_simulation()
                    if self.match_finished:
                        break
            else:
                # Do not carry time spent on the result screen into a rematch.
                self.fixed_timestep.reset()
            self.draw()

        pygame.quit()


def parse_controller_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Parse controller choices without coupling them to match logic."""
    parser = argparse.ArgumentParser(description="Run the Royale simulator")
    parser.add_argument(
        "--blue-controller",
        choices=controller_names(),
        default="human",
        help="decision maker for the blue team (default: human)",
    )
    parser.add_argument(
        "--red-controller",
        choices=controller_names(),
        default="scripted",
        help="decision maker for the red team (default: scripted)",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    """Create controllers from command-line configuration and start the game."""
    settings = parse_controller_arguments(arguments)
    ArenaViewer(
        blue_controller=settings.blue_controller,
        red_controller=settings.red_controller,
    ).run()


# Start the game only when this file is run directly.
if __name__ == "__main__":
    main()
