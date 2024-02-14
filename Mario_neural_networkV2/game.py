import numpy as np
from pathlib import Path
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from world import World
from SMB_WRAPPER import SMBRamWrapper
from mario import Mario
from tqdm import tqdm


class Game:

    def __init__(self):
        self.x0 = 0
        self.x1 = 16
        self.y0 = 0
        self.y1 = 13
        self.n_stack = 4
        self.n_skip = 4
        self.alfa = 0.5
        self.gamma = 0.7
        self.Episodes = 2000
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
        env = JoypadSpace(env, [["right"], ["right", "A"]])
        self.env_wrap = SMBRamWrapper(env, [self.x0, self.x1, self.y0, self.y1], n_stack=self.n_stack, n_skip=self.n_skip)
        self.world = World(env)
        save_dir = Path("checkpoints")
        save_dir.mkdir(parents=True, exist_ok=True)
        self.mario = Mario(52, 2, save_dir)


    def update_q_table(self, target, states, action):
        if self.world.check_dict(self.world.get_states_action_tuple(states, action)):
            old_value = self.world.Q_table[self.world.get_states_action_tuple(states, action)]
            self.world.Q_table[self.world.get_states_action_tuple(states, action)] = (1 - self.alfa) * old_value + target
        else:
            self.world.store_state.append(self.world.get_states_action_tuple(states, action))
            self.world.Q_table.update({self.world.get_states_action_tuple(states, action): target})

    def adjust_reward(self, state, reward, action, repeat_pos, info, done):
        reward = self.world.adjust_reward(state, reward, action)
        if repeat_pos[info['x_pos']] > 300:
            reward -= 15
            done = True
        return done

    def check_mario(self, states):
        for i in range(states.shape[2]):
            if len(np.argwhere(self.world.find_mario(states[:,:,i]))) == 0 or self.world.find_mario(states[:,:,i]).shape[0] > 3:
                return False
        return True

    def play(self):
        done = False
        states = self.env_wrap.reset()
        total_reward = 0
        repeat_pos = np.zeros(4000)
        max_pos = 0
        while(not done):
            if self.check_mario(states) is False:
                break
            state = self.world.get_states_action_tuple(states)
            action = self.mario.act(state)
            new_states, reward, done, info = self.env_wrap.step(action)
            if self.check_mario(new_states) is False:
                break
            done = self.adjust_reward(states[:,:,0],reward,action,repeat_pos,info, done)
            new_state = self.world.get_states_action_tuple(new_states)
            self.mario.cache(state, new_state, action, reward, done)
            q, loss = self.mario.learn()
            states = new_states
            repeat_pos[info['x_pos']] += 1
            total_reward += reward
            max_pos = info['x_pos']
            if done or info["flag_get"]:
                break
        return total_reward, max_pos

    def run(self):
        with tqdm(total= self.Episodes) as pbar:
            for episode in range(self.Episodes):
                total_reward, max_pos = self.play()
                self.world.update_epsilon(episode)
                self.world.total_reward.append(total_reward)
                if episode % 100 == 0:
                    print(f'Epoca {episode + 1}: {total_reward}')
                    print(max_pos)
                pbar.update(1)
