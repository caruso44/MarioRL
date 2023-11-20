import torch
from pathlib import Path
import gym
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from skip_frame import SkipFrame
from gray_sclae_observation import GrayScaleObservation
from Resize_observation import ResizeObservation    
from mario import Mario
from tqdm import tqdm
import gc
from MetricLogger import MetricLogger




if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb_array', apply_api_compatibility=True)

# Limitar o espaço de ação
#   0. andar para direita
#   1. pular para direita
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

# Aplicar Wrappers para o environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)


use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

save_dir = Path("checkpoints")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)
episodes = 1000
with tqdm(total= episodes) as pbar:
    for e in range(episodes):

        state = env.reset()

        while True:

            #agente escolhe a ação
            action = mario.act(state)

            # Agente performa a ação
            next_state, reward, done, trunc, info = env.step(action)

            # armazena a tupla com (estado, proximo estado, ação, recompensa)
            mario.cache(state, next_state, action, reward, done)

            # Aprende
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update no estado
            state = next_state

            # checa se o jogo acabou
            if done or info["flag_get"]:
                break
        logger.log_episode()

        if e % 20 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

        pbar.update(1)
mario.save()
del(mario)
gc.collect()