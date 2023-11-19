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
from Mario_net import MarioNet
import time


device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)


use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")


checkpoint = torch.load('checkpoints\mario_net_9.chkpt')
net = MarioNet((4, 84, 84), env.action_space.n).to(device)
net.load_state_dict(checkpoint['model'])
net.eval()
state = env.reset()
done = False

while not done:

    # Run agent on the state
    state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
    state = torch.tensor(state, device=device).unsqueeze(0)
    action_values = net(state, model="online")
    action = torch.argmax(action_values, axis=1).item()

    # Agent performs action
    next_state, reward, done, trunc, info = env.step(action)
    env.render()

    # Update state
    state = next_state
    time.sleep(0.01)

gc.collect()