import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from world import World
from SMB_WRAPPER import SMBRamWrapper
import torch
from Mario_net import MarioNet
import time
import gc


class Test:

    def __init__(self):
        self.x0 = 0
        self.x1 = 16
        self.y0 = 0
        self.y1 = 13
        self.n_stack = 4
        self.n_skip = 4
        env = gym_super_mario_bros.make('SuperMarioBros-1-2-v1')
        env = JoypadSpace(env, [["right"], ["right", "A"]])
        self.env_wrap = SMBRamWrapper(env, [self.x0, self.x1, self.y0, self.y1], n_stack=self.n_stack, n_skip=self.n_skip)
        self.world = World(env)
        checkpoint = torch.load('checkpoints\mario_net_2.chkpt')
        net = MarioNet(52, 2).float()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        net = net.to(device=self.device)
        net.load_state_dict(checkpoint['model'])
        net.eval()
        self.net = net


    def run(self):
        states = self.env_wrap.reset()
        done = False

        while not done:
            state = self.world.get_states_action_tuple(states)
            # Run agent on the state
            state = np.array(state[0].__array__() if isinstance(state, tuple) else state.__array__(), copy=True)
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            if state.dtype != next(self.net.parameters()).dtype:
                state = state.to(next(self.net.parameters()).dtype)
            action_values = self.net(state, model="online")
            action = torch.argmax(action_values, axis=1).item()

            # Agent performs action
            next_states, reward, done, info = self.env_wrap.step(action)
            self.env_wrap.render()

            # Update state
            states = next_states
            time.sleep(0.005)

        gc.collect()
