import numpy as np


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

    def update_epsilon(self, episode):
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
        if mario_pos[0] + 1 == 13:
            return False
        return (state[mario_pos[0] + 1, mario_pos[1]] == 1)

    def obstacle_ahead(self, state):
        mario_pos = self.find_mario(state)
        return state[mario_pos[0] - 1: mario_pos[0] + 2, mario_pos[1] + 1]

    def obstacle_near(self, state):
        mario_pos = self.find_mario(state)
        return state[mario_pos[0] - 1: mario_pos[0] + 2, mario_pos[1] + 2]

    def can_jump(self, state):
        mario_pos = self.find_mario(state)
        return (state[mario_pos[0] - 1, mario_pos[1]] != 1) and (state[mario_pos[0] - 2,mario_pos[1]] != 1)

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
        if state[mario_pos[0] + 1, mario_pos[1]] == -1:
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


    def get_state_action_tuple(self, state):
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
                       obstacle_near[0], obstacle_near[1], obstacle_near[2]])

    def get_states_action_tuple(self, states):
        tp = tuple([])
        for i in range(states.shape[2]):
            tp += self.get_state_action_tuple(states[:,:,i])
        return np.asarray(tp)


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
