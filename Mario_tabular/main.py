from SMB_RAMWRAPPER import SMBRamWrapper
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from World import World
from tqdm import tqdm

def load_smb_env(name='SuperMarioBros-1-1-v0', crop_dim=[0,16,0,13], n_stack=2, n_skip=4):
    env = gym_super_mario_bros.make(name)
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env_wrap = SMBRamWrapper(env, crop_dim, n_stack=n_stack, n_skip=n_skip)
    env_wrap = DummyVecEnv([lambda: env_wrap])

    return env_wrap


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
env = JoypadSpace(env, [["right"], ["right", "A"]])

x0 = 0
x1 = 16
y0 = 0
y1 = 13
n_stack = 1
n_skip = 4

env_wrap = SMBRamWrapper(env, [x0, x1, y0, y1], n_stack=n_stack, n_skip=n_skip)

world = World(env)

state = env_wrap.reset()
alfa = 0.5
gamma = 0.7
Episodes = 10000
with tqdm(total= Episodes) as pbar:
    for episode in range(Episodes):
        done = False
        state = env_wrap.reset()
        repeat_pos = np.zeros(4000)
        total_reward = 0
        max_pos = 0
        while(not done):
            # pular o estado caso o mario não esteja na imagem. Situações que ocorrem quando ele morre
            if len(np.argwhere(world.find_mario(state))) == 0 or world.find_mario(state).shape[0] > 3:
                break
            action = world.select_action(state)
            new_state, reward, done, info = env_wrap.step(action)
            env.render()
            # pular o estado caso o mario não esteja na imagem. Situações que ocorrem quando ele morre
            if len(np.argwhere(world.find_mario(new_state))) == 0 or world.find_mario(new_state).shape[0] > 3:
                break
            reward = world.adjust_reward(state, reward, action)
            # Matar o mario quando ele ficar preso em algum lugar
            if repeat_pos[info['x_pos']] > 300:
                reward -= 15
                done = True

            target = alfa * (reward +  gamma * world.select_best_action(new_state))
            if world.check_dict(world.get_state_action_tuple(state, action)):
                old_value = world.Q_table[world.get_state_action_tuple(state, action)]
                world.Q_table[world.get_state_action_tuple(state, action)] = (1 - alfa) * old_value + target
            else:
                world.store_state.append(world.get_state_action_tuple(state, action))
                world.Q_table.update({world.get_state_action_tuple(state, action): target})

            state = new_state
            repeat_pos[info['x_pos']] += 1
            total_reward += reward
            max_pos = info['x_pos']
        world.update_eps(episode)
        world.total_reward.append(total_reward)
        if episode % 1000 == 0:
            print(f'Epoca {episode + 1}: {total_reward}')
            print(max_pos)
        pbar.update(1)

world.save()



