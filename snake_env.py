"""
Includes enviromnent for snake AI gym
"""

import pygame
import random
from enum import Enum
from collections import namedtuple
pygame.init()


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', ['x', 'y'])


# colors for the UI
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
GREEN1 = (0,255,0)
GREEN2 = (0,255,100)



BLOCK_SIZE = 20
SPEED = 10

font = pygame.font.Font("arial.ttf", 25)

class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # int display
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

        # int game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.body = [self.head, 
                     Point(self.head.x-BLOCK_SIZE, self.head.y), 
                     Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.food = None
        self._place_food()
        self.score = 0

    def _place_food(self):
        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE

        self.food = Point(x,y)
        if self.food in self.body:
            self._place_food()

    def _update_ui(self):
        self.display.fill(BLACK)

        # render head
        # pygame.draw.rect(self.display, GREEN1, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
        # pygame.draw.rect(self.display, GREEN2, pygame.Rect(self.head.x+4, self.head.y+4, 12, 12))

        # render body
        for i, pt in enumerate(self.body):
            if i == 0:
                pygame.draw.rect(self.display, GREEN2, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(self.head.x+4, self.head.y+4, 12, 12))
                continue

            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)) 
        text = font.render("Score: "+str(self.score), True, WHITE)
        self.display.blit(text, [0,0])
        pygame.display.flip()

    def _move(self, dir):
        x = self.head.x
        y = self.head.y

        if dir == Direction.RIGHT:
            x+=BLOCK_SIZE
        elif dir == Direction.LEFT:
            x-=BLOCK_SIZE
        elif dir == Direction.UP:
            y-=BLOCK_SIZE
        elif dir == Direction.DOWN:
            y+=BLOCK_SIZE
        
        self.head = Point(x,y)
    
    def _is_collision(self):
        # check boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        
        # check self
        if self.head in self.body[1:]:
            return True
        
        return False

    def play_step(self):
        # get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    if self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    if self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    if self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    if self.direction != Direction.UP:
                        self.direction = Direction.DOWN

        # move
        self._move(self.direction)
        self.body.insert(0, self.head)

        # check collision
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # place food or move
        if self.food == self.head:
            self.score+=1
            self._place_food()
        else:
            self.body.pop()

        # update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # return game state and score
        
        return game_over, self.score


if __name__ == "__main__":

    game = SnakeGame()

    while True:
        over, score = game.play_step()
        if over:
            break

    
    pygame.quit()