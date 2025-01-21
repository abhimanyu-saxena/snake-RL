"""
Includes enviromnent for snake AI gym
"""

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

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

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # int display
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.reset()
        

    def reset(self):
        # int game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.body = [self.head, 
                     Point(self.head.x-BLOCK_SIZE, self.head.y), 
                     Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.food = None
        self._place_food()
        self.score = 0
        self.frame_iter = 0


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

    def _move(self, action):

        order = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = order.index(self.direction)

        if np.array_equal(action, [1,0,0]):
            new_dir = order[idx] # no change
        elif np.array_equal(action, [0,1,0]): 
            new_idx = (idx + 1) % 4
            new_dir = order[new_idx] # turn right
        else:
            new_idx = (idx - 1) % 4
            new_dir = order[new_idx] # turn left

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x+=BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x-=BLOCK_SIZE
        elif self.direction == Direction.UP:
            y-=BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y+=BLOCK_SIZE
        
        self.head = Point(x,y)
    
    def is_collision(self, pt=None):
        if pt == None:
            pt = self.head
        # check boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # check self
        if pt in self.body[1:]:
            return True
        
        return False

    def play_step(self, action):
        # update frame count and play action
        self.frame_iter+=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move
        self._move(action)
        self.body.insert(0, self.head)

        # check collision
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iter > 100*len(self.body):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # place food or move
        if self.food == self.head:
            self.score+=1
            reward = 10
            self._place_food()
        else:
            self.body.pop()

        # update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # return game state and score
        
        return reward, game_over, self.score