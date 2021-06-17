import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001

w = 640
h = 640

LOAD = 0
SAVE = 1

class Agent:

    def __init__(self, load = False):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.8 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(28, 256, 3)
        if load:
            print("====================LOAD MODEL====================")
            self.model.load_state_dict(torch.load('./model/model.pth'))
            self.model.eval()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        tail = game.snake[-1]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        x_diff_food = game.food.x - head.x
        y_diff_food = game.food.y - head.y
        food_incross = (abs(x_diff_food) == abs(y_diff_food))
        
        x_diff_tail = tail.x - head.x
        y_diff_tail = tail.y - head.y
        tail_incross = (abs(x_diff_tail) == abs(y_diff_tail))
        
        snake_part = [0] * 8
        for cur in game.snake[1:]:
            x_diff = cur.x - head.x
            y_diff = cur.y - head.y
            incross = (abs(x_diff) == abs(y_diff))
            if snake_part[0] == 0:
                if cur.x == head.x and cur.y > head.y:
                    snake_part[0] = cur.y - head.y
            else:
                snake_part[0] = min(cur.y - head.y, snake_part[0])
            if snake_part[1] == 0:
                if cur.x == head.x and cur.y < head.y:
                    snake_part[1] = head.y - cur.y
            else:
                snake_part[1] = min(head.y - cur.y, snake_part[1])

            if snake_part[2] == 0:
                if cur.y == head.y and cur.x > head.x:
                    snake_part[2] = cur.x - head.x
            else:
                snake_part[2] = min(cur.x - head.x, snake_part[2])

            if snake_part[3] == 0:
                if cur.y == head.y and cur.x < head.x:
                    snake_part[3] = head.x - cur.x
            else:
                snake_part[3] = min(head.x - cur.x, snake_part[3])

            if snake_part[4] == 0:
                if incross and x_diff > 0:
                    snake_part[4] = x_diff * 1.414
            else:
                snake_part[4] = min(x_diff * 1.414, snake_part[4])

            if snake_part[5] == 0:
                if incross and x_diff < 0:
                    snake_part[5] = -x_diff * 1.414
            else:
                snake_part[5] = min(-x_diff * 1.414, snake_part[5])
                    
            if snake_part[6] == 0:
                if incross and y_diff > 0:
                    snake_part[6] = y_diff * 1.414
            else:
                snake_part[6] = min(y_diff * 1.414, snake_part[6])

            if snake_part[7] == 0:
                if incross and y_diff < 0:
                    snake_part[7] = -y_diff * 1.414
            else:
                snake_part[7] = min(-y_diff * 1.414, snake_part[7])
            
        food_part = [0] * 8
        #tail vision
        if(game.food.x == head.x) and (game.food.y > head.y):
            food_part[0] = game.food.y - head.y
        if(game.food.x == head.x) and (game.food.y < head.y): 
            food_part[1] = -game.food.y + head.y
        if(game.food.y == head.y) and (game.food.x > head.x): 
            food_part[2] = game.food.x - head.x
        if(game.food.y == head.y) and (game.food.x < head.x): 
            food_part[3] = -game.food.x + head.x
        
        if food_incross and (x_diff_food > 0):
            food_part[4] = x_diff_food * 1.414
        if food_incross and (x_diff_food < 0): 
            food_part[5] = -x_diff_food*1.414
        if food_incross and (y_diff_food > 0): 
            food_part[6] = y_diff_food * 1.414
        if food_incross and (y_diff_food < 0):
            food_part[7] = -y_diff_food * 1.414

        state = [
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            food_part[0],
            food_part[1],
            food_part[2],
            food_part[3],
            food_part[4],
            food_part[5],
            food_part[6],
            food_part[7],
            
            snake_part[0],
            snake_part[1],
            snake_part[2],
            snake_part[3],
            snake_part[4],
            snake_part[5],
            snake_part[6],
            snake_part[7],

            head.x,
            head.y,
            w - head.x,
            h - head.y,
            min(head.x, head.y) * 1.414,
            min(w-head.x, head.y) * 1.414,
            min(head.x, h-head.y) * 1.414,
            min(w-head.x, h-head.y) * 1.414
            
            ]

        return np.array(state, dtype = np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon and not LOAD:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(LOAD)
    game = SnakeGameAI(w, h)
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score >= record and SAVE:
                record = score
                agent.model.save()
                print("====================SAVE MODEL====================")

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()