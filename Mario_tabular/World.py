import numpy as np
import random
import pickle


class World:
    def __init__(self, env):
        self.env = env
        self.mario_pos = 0
        self.enemy_pos = []
        self.episodes = 10000
        self.episolon = 1
        self.episilon_rate = 1/self.episodes
        self.Q_table = {}
        self.store_state = []
        self.rng = np.random.default_rng()
        self.total_reward = []

    def update_eps(self, episode):
        self.episolon = max(0.1, self.episolon - episode * self.episilon_rate)
    def find_mario(self, state):
        mask = (state == 2)
        if len(np.argwhere(mask)) == 0:
            return mask
        return np.argwhere(mask)[0]

    def find_enemy(self, state):
        mask = (state == -1)
        return np.argwhere(mask)


    def check_ground(self, state):
        mario_pos = self.find_mario(state)
        return (state[mario_pos[0] + 1, mario_pos[1], 0] == 1)

    def obstacle_ahead(self, state):
        mario_pos = self.find_mario(state)
        return state[mario_pos[0] - 2: mario_pos[0] + 1, mario_pos[1] + 1, 0]
    
    def obstacle_near(self, state):
        mario_pos = self.find_mario(state)
        return state[mario_pos[0] - 2: mario_pos[0] + 1, mario_pos[1] + 2, 0]

    def can_jump(self, state):
        mario_pos = self.find_mario(state)
        return (state[mario_pos[0] - 1, mario_pos[1], 0] != 1) and (state[mario_pos[0] - 2,mario_pos[1],0] != 1)

    def enemy_near(self, state):
        mario_pos = self.find_mario(state)
        enemies = self.find_enemy(state)
        max_dist = 0
        if len(enemies) == 0:
            return False
        for enemy in enemies:
            if enemy[1] - mario_pos[1] < 0: continue
            dist = max(abs(enemy[0] - mario_pos[0]), enemy[1] - mario_pos[1])
            max_dist = max(max_dist, dist)
        return max_dist <= 2
    
    def enemy_bellow(self, state):
        mario_pos = self.find_mario(state)
        if state[mario_pos[0] + 1, mario_pos[1], 0] == -1:
            return True
        return False

    def enemy_mid(self, state):
        mario_pos = self.find_mario(state)
        enemies = self.find_enemy(state)
        max_dist = 0
        if len(enemies) == 0:
            return False
        for enemy in enemies:
            if enemy[1] - mario_pos[1] < 0: continue
            dist = max(abs(enemy[0] - mario_pos[0]), enemy[1] - mario_pos[1])
            max_dist = max(max_dist, dist)
        return max_dist <= 3

    def enemy_far(self, state):
        mario_pos = self.find_mario(state)
        enemies = self.find_enemy(state)
        max_dist = 0
        if len(enemies) == 0:
            return False
        for enemy in enemies:
            if enemy[1] - mario_pos[1] < 0: continue
            dist = max(abs(enemy[0] - mario_pos[0]), enemy[1] - mario_pos[1])
            max_dist = max(max_dist, dist)
        return max_dist <= 4

    def colision(self, state):
        mario_pos = self.find_mario(state)
        enemies = self.find_enemy(state)
        colision = False
        for enemy in enemies:
            if (mario_pos[0] - enemy[0] == 1) or (enemy[1] - mario_pos[1] == 1):
                colision = True

        return colision

    def get_state_action_tuple(self, state, action):
        ground = self.check_ground(state)
        jump = self.can_jump(state)
        collision = self.colision(state)
        near = self.enemy_near(state)
        mid = self.enemy_mid(state)
        far = self.enemy_far(state)
        obstacle = self.obstacle_ahead(state)
        enemy_bellow = self.enemy_bellow(state)
        obstacle_near = self.obstacle_near(state)
        return tuple([ground, jump, collision, near, mid, far, obstacle[0], obstacle[1], obstacle[2], enemy_bellow,
                       obstacle_near[0], obstacle_near[1], obstacle_near[2], action])

    def check_dict(self, dic):
        for state in self.store_state:
            if state == dic:
                return True
        return False

    def adjust_reward(self, state, reward, action):
        enemies = self.find_enemy(state)
        mario_pos = self.find_mario(state)
        if action == 0 or mario_pos.shape[0] != 3:
            return reward
        if state[mario_pos[0] + 1, mario_pos[1] + 1, 0] == 0: #não penalizar por pular poço
            return reward
        if state[mario_pos[0], mario_pos[1] + 1, 0] == 1 or state[mario_pos[0], mario_pos[1] + 2, 0] == 1: #não penalisar por pular obstáculo
            return reward
        if len(enemies) == 0:
            return - 1
        for enemy in enemies:
            if (enemy[1] - mario_pos[1]) > 2: #penalisar o mario por pular de forma desnecessária
                return -1
            
        return reward

    def select_action(self, state):
        number = self.rng.random()
        if number < self.episolon:
            action = random.choice([0,1])
            return action
        else:
            dic0 = self.get_state_action_tuple(state, 0)
            dic1 = self.get_state_action_tuple(state, 1)
            aux0 = 0
            aux1 = 0
            if self.check_dict(dic0):
                aux0 = self.Q_table[dic0]
            if self.check_dict(dic1):
                aux1 = self.Q_table[dic1]
            if aux1 > aux0:
                return 1
            return 0

    def select_best_action(self, state):
        dic0 = self.get_state_action_tuple(state, 0)
        dic1 = self.get_state_action_tuple(state, 1)
        aux0 = 0
        aux1 = 0
        if self.check_dict(dic0):
            aux0 = self.Q_table[dic0]
        if self.check_dict(dic1):
            aux1 = self.Q_table[dic1]
        return max(aux0, aux1)
    
    def save(self):
        with open('Q_table.pkl', 'wb') as file:
            pickle.dump(self.Q_table, file)
        
        with open('reward.pkl', 'wb') as file:
            pickle.dump(self.Q_table, file)
