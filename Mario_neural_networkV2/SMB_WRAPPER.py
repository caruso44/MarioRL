import numpy as np
import gym
from gym import spaces
from SMB_GRIG import smb_grid

class SMBRamWrapper(gym.ObservationWrapper):
    def __init__(self, env, crop_dim=[0, 16, 0, 13], n_stack=4, n_skip=2):
        '''
        crop_dim: [x0, x1, y0, y1]
        obs shape = (height, width, n_stack), n_stack=0 is the most recent frame
        n_skip: e.g. n_stack=4, n_skip=2, use frames [0, 2, 4, 6]
        '''
        gym.Wrapper.__init__(self, env)
        self.crop_dim = crop_dim
        self.n_stack = n_stack
        self.n_skip = n_skip
        # Modified from stable_baselines3.common.atari_wrappers.WarpFrame()
        # https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#AtariWrapper
        self.width = crop_dim[1] - crop_dim[0]
        self.height = crop_dim[3] - crop_dim[2]
        self.observation_space = spaces.Box(
            low=-1, high=2, shape=(self.height, self.width, self.n_stack), dtype=int
        )

        self.frame_stack = np.zeros((self.height, self.width, (self.n_stack-1)*self.n_skip+1))
        #self.INDEX_SKIP = 1

    def observation(self, obs):
        grid = smb_grid(self.env)
        frame = grid.rendered_screen # 2d array
        frame = self.crop_obs(frame)

        self.frame_stack[:,:,1:] = self.frame_stack[:,:,:-1] # shift frame_stack by 1
        self.frame_stack[:,:,0] = frame # add current frame to stack
        obs = self.frame_stack[:,:,::self.n_skip]
        return obs

    def reset(self):
        obs = self.env.reset()
        self.frame_stack = np.zeros((self.height, self.width, (self.n_stack-1)*self.n_skip+1))
        grid = smb_grid(self.env)
        frame = grid.rendered_screen # 2d array
        frame = self.crop_obs(frame)
        for i in range(self.frame_stack.shape[-1]):
            self.frame_stack[:,:,i] = frame
        obs = self.frame_stack[:,:,::self.n_skip]
        return obs

    def crop_obs(self, im):
        '''
        Crop observed frame image to reduce input size
        Returns cropped_frame = original_frame[y0:y1, x0:x1]
        '''
        [x0, x1, y0, y1] = self.crop_dim
        im_crop = im[y0:y1, x0:x1]
        return im_crop