import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

# Enumeration representing the directions that the snake can move
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Font of the text in the game
game_font = pygame.font.SysFont("DejaVu Sans Mono", 25)

Point = namedtuple('Point', 'x, y')

# colors used in the game
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE = (50, 153, 213)
BLACK = (0,0,0)
GREEN = (0, 255, 0)

SNAKE_BLOCK = 20
SNAKE_SPEED = 40

class SnakeGameAI:

    # Initializing the game
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        # Creating the display window
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Group P11 Snake')
        # Clock object controls the frame rate of the game
        self.clock = pygame.time.Clock()
        # Resets the game states
        self.reset()

    def reset(self):
        # Initializing the direction of the snake
        self.direction = Direction.RIGHT

        # Initializing the snake by positioning the snake block in the center 
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head,
                      Point(self.head.x-SNAKE_BLOCK, self.head.y),
                      Point(self.head.x-(2*SNAKE_BLOCK), self.head.y)]

        self.score = 0
        self.food = None
        self._food()
        self.iteration = 0


    def _food(self):
        # Places the food randomly
        x = random.randint(0, (self.width-SNAKE_BLOCK )//SNAKE_BLOCK )*SNAKE_BLOCK
        y = random.randint(0, (self.height-SNAKE_BLOCK )//SNAKE_BLOCK )*SNAKE_BLOCK
        self.food = Point(x, y)

        # If there is overlapping of the food and snake
        if self.food in self.snake:
            self._food()


    def play_step(self, action):
        # Increasing the number of frame iterations by 1
        self.iteration += 1

        # Checking if the user has attempted to close the game window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        

        # Moving the snake object and inserting the head
        self._move(action) 
        self.snake.insert(0, self.head)
        
        # initializing the reward and game state if it is over or not
        reward = 0
        done = False

        # If the snake has collided either with itself or with the wall, or if the snake is running for too long without eating any food
        if self.is_colliding() or self.iteration > 100*len(self.snake):
            done = True
            reward = -10
            return reward, done, self.score

        # if the snake's head's position and the food's position matches
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._food()
        # else removes the last segment making it look like snake is moving forward
        else:
            self.snake.pop()
        
        # Updating the UI i.e. includes rending a new game state
        self._ui()
        self.clock.tick(SNAKE_SPEED)
        return reward, done, self.score

    # Function for detecting collision
    def is_colliding(self, pnt=None):
        if pnt is None:
            pnt = self.head

        # Checking if the snake is collidig with the right boundary or the left boundary
        if pnt.x > self.width - SNAKE_BLOCK or pnt.x < 0 or pnt.y > self.height - SNAKE_BLOCK or pnt.y < 0:
            return True
        if pnt in self.snake[1:]:
            return True

        return False

    # Updating the game's user interface
    def _ui(self):
        self.display.fill(BLUE)

        # Drawing each block in a snake through a loop
        for pnt in self.snake:
            pygame.draw.rect(self.display, BLACK, pygame.Rect(pnt.x, pnt.y, SNAKE_BLOCK, SNAKE_BLOCK))

        # Drawing the food item 
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food.x, self.food.y, SNAKE_BLOCK, SNAKE_BLOCK))

        text = game_font.render("Your Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):

        # Defining a list with four directions that move clockwise 
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        i = clock_wise.index(self.direction)

        # Based on what the action is, that will be the direction the snake moves at i.e. [1, 0, 0] = straight, [0, 1, 0] = right, [0, 0, 1] = left
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[i] 
        elif np.array_equal(action, [0, 1, 0]):
            # Calculate the index of the new direction in a clockwise manner
            next_i = (i + 1) % 4
            new_direction = clock_wise[next_i] 
        else: 
            # Calculate the index of the previous direction in a clockwise manner
            next_i = (i - 1) % 4
            new_direction = clock_wise[next_i]

        # updating the direction
        self.direction = new_direction

        # Based on the current direction, the position of the head needs to be updated
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += SNAKE_BLOCK
        elif self.direction == Direction.LEFT:
            x -= SNAKE_BLOCK
        elif self.direction == Direction.DOWN:
            y += SNAKE_BLOCK
        elif self.direction == Direction.UP:
            y -= SNAKE_BLOCK

        # Finally, we needed to update the position of the head with the points of the new x and the y coordinate
        self.head = Point(x, y)