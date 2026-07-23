"""Basic phone-sized grid arena for the simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pygame


SCREEN_WIDTH = 450
SCREEN_HEIGHT = 800
GRID_COLUMNS = 18
GRID_ROWS = 32
TILE_SIZE = 25
RIVER_HEIGHT = TILE_SIZE * 2
RIVER_TOP = (SCREEN_HEIGHT - RIVER_HEIGHT) // 2
FPS = 60
MATCH_DURATION_SECONDS = 3 * 60

ARENA_COLOR = (74, 145, 82)
ALTERNATE_TILE_COLOR = (78, 151, 86)
RIVER_COLOR = (56, 144, 201)
RIVER_HIGHLIGHT_COLOR = (91, 177, 224)
BRIDGE_PLANK_COLOR = (174, 112, 62)
BRIDGE_PLANK_LIGHT_COLOR = (213, 151, 87)
BRIDGE_EDGE_COLOR = (101, 64, 42)
GRID_COLOR = (42, 91, 51)
HOVER_COLOR = (255, 255, 255, 75)
SELECTED_COLOR = (255, 218, 71)
TEXT_COLOR = (245, 245, 245)
TEXT_BACKGROUND_COLOR = (20, 35, 24, 185)
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

LEFT_LANE_X = (SCREEN_WIDTH // 4) // TILE_SIZE * TILE_SIZE + TILE_SIZE // 2
RIGHT_LANE_X = SCREEN_WIDTH - LEFT_LANE_X
CENTER_LANE_X = SCREEN_WIDTH // 2
TOP_KING_Y = TILE_SIZE * 4
TOP_PRINCESS_Y = TILE_SIZE * 8
BRIDGE_WIDTH = TILE_SIZE * 2
BRIDGE_HEIGHT = RIVER_HEIGHT + TILE_SIZE
ELIXIR_MAX = 10.0
ELIXIR_START = 5.0
ELIXIR_SECONDS_PER_UNIT = 2.8
DOUBLE_ELIXIR_START = 120.0
TRIPLE_ELIXIR_START = 240.0
ELIXIR_COLOR = (220, 52, 213)
ELIXIR_HIGHLIGHT_COLOR = (255, 115, 241)
ELIXIR_DARK_COLOR = (85, 24, 105)
ELIXIR_EMPTY_COLOR = (48, 27, 64)
ELIXIR_FRAME_COLOR = (30, 18, 40)


@dataclass(frozen=True)
class Tower:
    """A tower placed in the arena."""

    kind: str
    team: str
    center: tuple[int, int]


@dataclass
class ElixirMeter:
    """Track the player's continuously generated battle elixir."""

    amount: float = ELIXIR_START
    full_notice_remaining: float = 0.0

    @staticmethod
    def multiplier_at(match_elapsed: float) -> int:
        """Return the standard battle multiplier at a match timestamp."""
        if match_elapsed >= TRIPLE_ELIXIR_START:
            return 3
        if match_elapsed >= DOUBLE_ELIXIR_START:
            return 2
        return 1

    def update(self, delta_seconds: float, match_elapsed: float) -> None:
        """Generate elixir, accounting for rate boundaries crossed by this tick."""
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
        if previous_amount < ELIXIR_MAX and self.amount >= ELIXIR_MAX:
            self.full_notice_remaining = 1.5

    def spend(self, cost: int) -> bool:
        """Spend whole elixir if the player can afford the requested cost."""
        if cost < 0 or self.amount + 1e-9 < cost:
            return False

        self.amount -= cost
        self.full_notice_remaining = 0.0
        return True


TOWERS = (
    Tower("king", "red", (CENTER_LANE_X, TOP_KING_Y)),
    Tower("princess", "red", (LEFT_LANE_X, TOP_PRINCESS_Y)),
    Tower("princess", "red", (RIGHT_LANE_X, TOP_PRINCESS_Y)),
    Tower(
        "princess",
        "blue",
        (LEFT_LANE_X, SCREEN_HEIGHT - TOP_PRINCESS_Y),
    ),
    Tower(
        "princess",
        "blue",
        (RIGHT_LANE_X, SCREEN_HEIGHT - TOP_PRINCESS_Y),
    ),
    Tower("king", "blue", (CENTER_LANE_X, SCREEN_HEIGHT - TOP_KING_Y)),
)


def remaining_match_seconds(start_ms: int, now_ms: int) -> int:
    """Return whole seconds left in a three-minute match."""
    elapsed_ms = max(0, now_ms - start_ms)
    remaining_ms = max(0, MATCH_DURATION_SECONDS * 1000 - elapsed_ms)
    return (remaining_ms + 999) // 1000


def format_match_time(seconds: int) -> str:
    """Format a countdown value as M:SS."""
    minutes, seconds_part = divmod(max(0, seconds), 60)
    return f"{minutes}:{seconds_part:02d}"


class ArenaViewer:
    """Display and interact with a grid-based arena."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Royale Simulator - Grid Arena")

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.timer_label_font = pygame.font.Font(None, 19)
        self.timer_label_font.set_bold(True)
        self.timer_font = pygame.font.Font(None, 42)
        self.timer_font.set_bold(True)
        self.match_over_font = pygame.font.Font(None, 52)
        self.match_over_font.set_bold(True)
        self.match_started_at = pygame.time.get_ticks()
        self.elixir_font = pygame.font.Font(None, 31)
        self.elixir_notice_font = pygame.font.Font(None, 25)
        self.running = True
        self.selected_tile: tuple[int, int] | None = None
        self.elixir = ElixirMeter()
        self.match_elapsed = 0.0

    @staticmethod
    def screen_to_tile(position: tuple[int, int]) -> tuple[int, int] | None:
        """Convert a mouse position into a grid coordinate."""
        mouse_x, mouse_y = position

        if not (0 <= mouse_x < SCREEN_WIDTH and 0 <= mouse_y < SCREEN_HEIGHT):
            return None

        column = mouse_x // TILE_SIZE
        row = mouse_y // TILE_SIZE
        return column, row

    @staticmethod
    def tile_rectangle(tile: tuple[int, int]) -> pygame.Rect:
        """Return the screen rectangle occupied by a tile."""
        column, row = tile
        return pygame.Rect(
            column * TILE_SIZE,
            row * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        )

    @staticmethod
    def river_rectangle() -> pygame.Rect:
        """Return a horizontal river centered exactly within the arena."""
        return pygame.Rect(0, RIVER_TOP, SCREEN_WIDTH, RIVER_HEIGHT)

    @staticmethod
    def bridge_rectangles() -> tuple[pygame.Rect, pygame.Rect]:
        """Return bridge geometry centered on both princess-tower lanes."""
        bridges = []

        for center_x in (LEFT_LANE_X, RIGHT_LANE_X):
            bridge = pygame.Rect(0, 0, BRIDGE_WIDTH, BRIDGE_HEIGHT)
            bridge.center = (center_x, SCREEN_HEIGHT // 2)
            bridges.append(bridge)

        return bridges[0], bridges[1]

    @classmethod
    def is_walkable_position(cls, position: tuple[int, int]) -> bool:
        """Return whether ground movement may occupy a screen position."""
        if cls.screen_to_tile(position) is None:
            return False

        if not cls.river_rectangle().collidepoint(position):
            return True

        return any(
            bridge.collidepoint(position)
            for bridge in cls.bridge_rectangles()
        )

    def handle_events(self) -> None:
        """Process window, keyboard, and mouse events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.selected_tile = None
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    self.elixir.spend(event.key - pygame.K_0)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.selected_tile = self.screen_to_tile(event.pos)

    def draw_arena(self) -> None:
        """Draw the arena background and grid."""
        self.screen.fill(ARENA_COLOR)

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
                (0, y),
                (SCREEN_WIDTH, y),
                2,
            )

        for column in range(GRID_COLUMNS + 1):
            x = column * TILE_SIZE
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (x, 0),
                (x, SCREEN_HEIGHT),
                1,
            )

        for row in range(GRID_ROWS + 1):
            y = row * TILE_SIZE
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (0, y),
                (SCREEN_WIDTH, y),
                1,
            )

        self.draw_bridges()

    def draw_bridges(self) -> None:
        """Draw wooden crossings over the river in both tower lanes."""
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
        """Draw a thick stone princess tower with an archer platform."""
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
        """Draw the larger central tower with battlements and a crown."""
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
        """Draw all six arena towers."""
        for tower in TOWERS:
            if tower.kind == "king":
                self.draw_king_tower(tower)
            else:
                self.draw_princess_tower(tower)

    def draw_tile_highlights(self) -> tuple[int, int] | None:
        """Draw the hovered and selected placement tiles."""
        hovered_tile = self.screen_to_tile(pygame.mouse.get_pos())

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

        return hovered_tile

    def draw_status(self, hovered_tile: tuple[int, int] | None) -> None:
        """Display simple interaction instructions and coordinates."""
        if hovered_tile is None:
            coordinate_text = "Tile: outside arena"
        else:
            coordinate_text = f"Tile: x={hovered_tile[0]}, y={hovered_tile[1]}"

        message = f"{coordinate_text} | 1-9: spend | Space: clear | Esc: quit"
        text_surface = self.font.render(message, True, TEXT_COLOR)

        background = pygame.Surface(
            (text_surface.get_width() + 16, text_surface.get_height() + 10),
            pygame.SRCALPHA,
        )
        background.fill(TEXT_BACKGROUND_COLOR)

        self.screen.blit(background, (6, 6))
        self.screen.blit(text_surface, (14, 11))

    def draw_match_timer(self) -> None:
        """Draw the match countdown in the upper-right corner."""
        seconds_left = remaining_match_seconds(
            self.match_started_at,
            pygame.time.get_ticks(),
        )

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

        label = self.timer_label_font.render("Time left", True, TEXT_COLOR)
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

        if seconds_left == 0:
            self.draw_match_over()

    def draw_match_over(self) -> None:
        """Display a centered banner after the countdown reaches zero."""
        banner_text = self.match_over_font.render("MATCH OVER", True, TEXT_COLOR)
        banner = pygame.Rect(
            0,
            0,
            banner_text.get_width() + 34,
            banner_text.get_height() + 20,
        )
        banner.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        overlay = pygame.Surface(banner.size, pygame.SRCALPHA)
        pygame.draw.rect(
            overlay,
            TIMER_SHADOW_COLOR,
            overlay.get_rect(),
            border_radius=9,
        )
        pygame.draw.rect(
            overlay,
            TIMER_BORDER_COLOR,
            overlay.get_rect(),
            3,
            border_radius=9,
        )
        overlay.blit(
            banner_text,
            (
                (banner.width - banner_text.get_width()) // 2,
                (banner.height - banner_text.get_height()) // 2,
            ),
        )
        self.screen.blit(overlay, banner.topleft)

    def draw_elixir_bar(self) -> None:
        """Draw the segmented purple elixir meter used on the player's HUD."""
        hud = pygame.Surface((SCREEN_WIDTH, 53), pygame.SRCALPHA)
        hud.fill((18, 13, 28, 215))
        self.screen.blit(hud, (0, SCREEN_HEIGHT - 53))

        bar_rect = pygame.Rect(54, SCREEN_HEIGHT - 31, 382, 22)
        pygame.draw.rect(
            self.screen,
            ELIXIR_FRAME_COLOR,
            bar_rect.inflate(6, 6),
            border_radius=8,
        )

        gap = 2
        cell_width = (bar_rect.width - gap * 9) / 10
        for index in range(10):
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
        badge_center = (28, SCREEN_HEIGHT - 21)
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
            pygame.Rect(15, SCREEN_HEIGHT - 38, 26, 25),
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

    def draw(self) -> None:
        """Render one frame."""
        self.draw_arena()
        hovered_tile = self.draw_tile_highlights()
        self.draw_towers()
        self.draw_status(hovered_tile)
        self.draw_match_timer()
        self.draw_elixir_bar()
        pygame.display.flip()

    def run(self) -> None:
        """Run the viewer until the user closes it."""
        while self.running:
            delta_seconds = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.elixir.update(delta_seconds, self.match_elapsed)
            self.match_elapsed += delta_seconds
            self.draw()

        pygame.quit()


def main() -> None:
    ArenaViewer().run()


if __name__ == "__main__":
    main()
