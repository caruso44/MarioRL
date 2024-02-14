import torch
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gc
from Mario_net import MarioNet
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()



use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")


checkpoint = torch.load('checkpoints\mario_net_10.chkpt')
net = MarioNet(52, 2).float()
net = net.to(device=device)
net.load_state_dict(checkpoint['model'])
net.eval()
state = env.reset()
done = False

while not done:

    # Run agent on the state
    state = np.array(state[0].__array__() if isinstance(state, tuple) else state.__array__(), copy=True)
    state = torch.tensor(state, device=device).unsqueeze(0)
    action_values = net(state, model="online")
    action = torch.argmax(action_values, axis=1).item()

    # Agent performs action
    next_state, reward, done, trunc, info = env.step(action)
    env.render()

    # Update state
    state = next_state
    time.sleep(0.05)

gc.collect()
