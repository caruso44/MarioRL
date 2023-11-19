import pandas as pd
from SMB_RAMWRAPPER import SMBRamWrapper
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import numpy as np
import gymnasium.utils.save_video as sv
import time


def find_mario(state):
        mask = (state == 2)
        if len(np.argwhere(mask)) == 0:
            return mask
        return np.argwhere(mask)[0]

def find_enemy(state):
    mask = (state == -1)
    return np.argwhere(mask)

def check_ground(state):
    mario_pos = find_mario(state)
    return (state[mario_pos[0] + 1, mario_pos[1], 0] == 1)

def obstacle_ahead(state):
    mario_pos = find_mario(state)
    return state[mario_pos[0] - 2: mario_pos[0] + 1, mario_pos[1] + 1, 0]

def obstacle_near(state):
    mario_pos = find_mario(state)
    return state[mario_pos[0] - 2: mario_pos[0] + 1, mario_pos[1] + 2, 0]

def can_jump(state):
    mario_pos = find_mario(state)
    return (state[mario_pos[0] - 1, mario_pos[1], 0] != 1) and (state[mario_pos[0] - 2,mario_pos[1],0] != 1)

def enemy_near(state):
    mario_pos = find_mario(state)
    enemies = find_enemy(state)
    max_dist = 0
    if len(enemies) == 0:
        return False
    for enemy in enemies:
        if enemy[1] - mario_pos[1] < 0: continue
        dist = max(abs(enemy[0] - mario_pos[0]), enemy[1] - mario_pos[1])
        max_dist = max(max_dist, dist)
    return max_dist <= 2

def enemy_bellow(state):
    mario_pos = find_mario(state)
    if state[mario_pos[0] + 1, mario_pos[1], 0] == -1:
        return True
    return False

def enemy_mid(state):
    mario_pos = find_mario(state)
    enemies = find_enemy(state)
    max_dist = 0
    if len(enemies) == 0:
        return False
    for enemy in enemies:
        if enemy[1] - mario_pos[1] < 0: continue
        dist = max(abs(enemy[0] - mario_pos[0]), enemy[1] - mario_pos[1])
        max_dist = max(max_dist, dist)
    return max_dist <= 3

def enemy_far(state):
    mario_pos = find_mario(state)
    enemies = find_enemy(state)
    max_dist = 0
    if len(enemies) == 0:
        return False
    for enemy in enemies:
        if enemy[1] - mario_pos[1] < 0: continue
        dist = max(abs(enemy[0] - mario_pos[0]), enemy[1] - mario_pos[1])
        max_dist = max(max_dist, dist)
    return max_dist <= 4

def colision(state):
    mario_pos = find_mario(state)
    enemies = find_enemy(state)
    colision = False
    for enemy in enemies:
        if (mario_pos[0] - enemy[0] == 1) or (enemy[1] - mario_pos[1] == 1):
            colision = True

    return colision


def get_state_action_tuple(state, action):
        ground = check_ground(state)
        jump = can_jump(state)
        collision = colision(state)
        near = enemy_near(state)
        mid = enemy_mid(state)
        far = enemy_far(state)
        obstacle = obstacle_ahead(state)
        enemy_bellow2 = enemy_bellow(state)
        obstacle_near2 = obstacle_near(state)
        return tuple([ground, jump, collision, near, mid, far, obstacle[0], obstacle[1], obstacle[2], enemy_bellow2,
                       obstacle_near2[0], obstacle_near2[1], obstacle_near2[2], action])

def select_action(state, Q_table):

    dic0 = get_state_action_tuple(state, 0)
    dic1 = get_state_action_tuple(state, 1)
    aux0 = 0
    aux1 = 0
    if dic0 in Q_table:
        aux0 = Q_table[dic0]
    if dic1 in Q_table:
        aux1 = Q_table[dic1]
    if aux1 > aux0:
        return 1
    return 0


q_table = pd.read_pickle("q_table/Q_table_prov.pkl")
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
env = JoypadSpace(env, [["right"], ["right", "A"]])
# Setup cropping size
x0 = 0
x1 = 16
y0 = 0
y1 = 13
n_stack = 1
n_skip = 4

env_wrap = SMBRamWrapper(env, [x0, x1, y0, y1], n_stack=n_stack, n_skip=n_skip)
state = env_wrap.reset()
done = False
for i in range(2):
    state = env_wrap.reset()
    while(not done):
        if len(np.argwhere(find_mario(state))) == 0 or find_mario(state).shape[0] > 3:
            break
        action = select_action(state, q_table)
        new_state, reward, done, info = env_wrap.step(action)
        env.render()
        state = new_state
        time.sleep(0.05)
