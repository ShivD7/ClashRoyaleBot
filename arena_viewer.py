"""Basic phone-sized grid arena for the simulator."""

from __future__ import annotations

import pygame


SCREEN_WIDTH = 450
SCREEN_HEIGHT = 500
GRID_COLUMNS = 18
GRID_ROWS = 32
TILE_SIZE = 25
FPS = 60

ARENA_COLOR = (74, 145, 82)
ALTERNATE_TILE_COLOR = (78, 151, 86)
GRID_COLOR = (42, 91, 51)
HOVER_COLOR = (255, 255, 255, 75)
SELECTED_COLOR = (255, 218, 71)
TEXT_COLOR = (245, 245, 245)
TEXT_BACKGROUND_COLOR = (20, 35, 24, 185)


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
