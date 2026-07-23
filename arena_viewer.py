"""Basic phone-sized grid arena for the simulator."""

from __future__ import annotations

from dataclasses import dataclass

import pygame


SCREEN_WIDTH = 450
SCREEN_HEIGHT = 800
GRID_COLUMNS = 18
GRID_ROWS = 32
TILE_SIZE = 25
RIVER_HEIGHT = TILE_SIZE * 2
RIVER_TOP = (SCREEN_HEIGHT - RIVER_HEIGHT) // 2
FPS = 60

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

LEFT_LANE_X = (SCREEN_WIDTH // 4) // TILE_SIZE * TILE_SIZE + TILE_SIZE // 2
RIGHT_LANE_X = SCREEN_WIDTH - LEFT_LANE_X
CENTER_LANE_X = SCREEN_WIDTH // 2
TOP_KING_Y = TILE_SIZE * 4
TOP_PRINCESS_Y = TILE_SIZE * 8
BRIDGE_WIDTH = TILE_SIZE * 2
BRIDGE_HEIGHT = RIVER_HEIGHT + TILE_SIZE


@dataclass(frozen=True)
class Tower:
    """A tower placed in the arena."""

    kind: str
    team: str
    center: tuple[int, int]


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


class ArenaViewer:
    """Display and interact with a grid-based arena."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Royale Simulator - Grid Arena")

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.running = True
        self.selected_tile: tuple[int, int] | None = None

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

        message = f"{coordinate_text}  |  Click: select  Space: clear  Esc: quit"
        text_surface = self.font.render(message, True, TEXT_COLOR)

        background = pygame.Surface(
            (text_surface.get_width() + 16, text_surface.get_height() + 10),
            pygame.SRCALPHA,
        )
        background.fill(TEXT_BACKGROUND_COLOR)

        self.screen.blit(background, (6, 6))
        self.screen.blit(text_surface, (14, 11))

    def draw(self) -> None:
        """Render one frame."""
        self.draw_arena()
        hovered_tile = self.draw_tile_highlights()
        self.draw_towers()
        self.draw_status(hovered_tile)
        pygame.display.flip()

    def run(self) -> None:
        """Run the viewer until the user closes it."""
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


def main() -> None:
    ArenaViewer().run()


if __name__ == "__main__":
    main()
