import pygame
import random

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
GRID_WIDTH = SCREEN_WIDTH // BLOCK_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // BLOCK_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)

# Tetrominoes
tetrominoes = [
    [[1, 1, 1, 1]],  # I
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]],  # J
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1], [1, 1]],  # O
    [[1, 1, 0], [0, 1, 1]]  # Z
]


class Tetris:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.game_over = False

    def new_piece(self):
        piece = random.choice(tetrominoes)
        piece_color = random.choice([CYAN, BLUE, ORANGE, YELLOW, GREEN, PURPLE])
        return {'shape': piece, 'color': piece_color, 'x': 3, 'y': 0}

    def draw_piece(self, piece):
        shape = piece['shape']
        color = piece['color']
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, color,
                                     pygame.Rect((piece['x'] + x) * BLOCK_SIZE, (piece['y'] + y) * BLOCK_SIZE,
                                                 BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.screen, BLACK,
                                     pygame.Rect((piece['x'] + x) * BLOCK_SIZE, (piece['y'] + y) * BLOCK_SIZE,
                                                 BLOCK_SIZE, BLOCK_SIZE), 1)

    def draw_grid(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                pygame.draw.rect(self.screen, GRAY,
                                 pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)
                if self.grid[y][x]:
                    pygame.draw.rect(self.screen, self.grid[y][x],
                                     pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def move_piece(self, dx, dy):
        if not self.check_collision(self.current_piece, dx, dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy

    def rotate_piece(self):
        rotated_piece = [[row[i] for row in self.current_piece['shape'][::-1]] for i in range(len(self.current_piece['shape'][0]))]
        if not self.check_collision({'shape': rotated_piece, 'color': self.current_piece['color'], 'x': self.current_piece['x'], 'y': self.current_piece['y']}, 0, 0):
            self.current_piece['shape'] = rotated_piece

    def check_collision(self, piece, dx=0, dy=0):
        shape = piece['shape']
        x = piece['x'] + dx
        y = piece['y'] + dy
        for y_index, row in enumerate(shape):
            for x_index, cell in enumerate(row):
                if cell:
                    if x + x_index < 0 or x + x_index >= GRID_WIDTH or \
                            y + y_index >= GRID_HEIGHT or \
                            self.grid[y + y_index][x + x_index]:
                        return True
        return False

    def merge_piece(self):
        shape = self.current_piece['shape']
        color = self.current_piece['color']
        x = self.current_piece['x']
        y = self.current_piece['y']
        for y_index, row in enumerate(shape):
            for x_index, cell in enumerate(row):
                if cell:
                    self.grid[y + y_index][x + x_index] = color

    def check_lines(self):
        lines_cleared = 0
        y = GRID_HEIGHT - 1
        while y >= 0:
            if all(self.grid[y]):
                for row in range(y, 0, -1):
                    self.grid[row] = self.grid[row - 1]
                self.grid[0] = [0] * GRID_WIDTH
                lines_cleared += 1
            else:
                y -= 1
        return lines_cleared

    def game_loop(self):
        while not self.game_over:
            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_piece(self.current_piece)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move_piece(-1, 0)
                    if event.key == pygame.K_RIGHT:
                        self.move_piece(1, 0)
                    if event.key == pygame.K_DOWN:
                        self.move_piece(0, 1)
                    if event.key == pygame.K_UP:
                        self.rotate_piece()

            if self.check_collision(self.current_piece, 0, 1):
                self.merge_piece()
                lines_cleared = self.check_lines()
                if lines_cleared:
                    print(f"Lines cleared: {lines_cleared}")
                self.current_piece = self.new_piece()
                if self.check_collision(self.current_piece):
                    self.game_over = True

            self.move_piece(0, 1)

            self.clock.tick(5)  # Adjust speed here

        pygame.quit()


if __name__ == '__main__':
    pygame.init()
    Tetris().game_loop()
