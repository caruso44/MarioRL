import time
import matplotlib.pyplot as plt
from matplotlib import colors
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import obs_as_tensor
from stable_baselines3.common.evaluation import evaluate_policy
import imageio


class SMB():
    '''
    Wrapper function containing the processed environment and the loaded model
    '''
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def play(self, episodes=5, deterministic=False, render=True, return_eval=False):
        for episode in range(1, episodes+1):
            states = self.env.reset()
            done = False
            score = 0

            if render == True:
                while not done:
                    self.env.render()
                    action, _ = self.model.predict(states, deterministic=deterministic)
                    states, reward, done, info = self.env.step(action)
                    score += reward
                    time.sleep(0.01)
                print('Episode:{} Score:{}'.format(episode, score))
            else:
                while not done:
                    action, _ = self.model.predict(states, deterministic=deterministic)
                    states, reward, done, info = self.env.step(action)
                    score += reward
        if return_eval == True:
            return score, info
        else:
            return

    def evaluate(self, episodes=20, deterministic=False):
        '''
        returns rewards, steps (both have length [episodes])
        '''
        rewards, steps = evaluate_policy(self.model, self.env, n_eval_episodes=episodes,
                                 deterministic=deterministic, render=False,
                                 return_episode_rewards=True)
        return rewards, steps




    def predict_proba(self, state):
        '''
        Predict the probability of each action given a state
        https://stackoverflow.com/questions/66428307/how-to-get-action-propability-in-stable-baselines-3/70012691#70012691?newreg=bd5479b970664069b359903e0151b4a1
        '''
        model = self.model
        obs = obs_as_tensor(state, model.policy.device)
        dis = model.policy.get_distribution(obs)
        probs = dis.distribution.probs
        probs_np = probs.detach().numpy()
        return probs_np

    #############
    # functions for making plots & videos

    def make_video_frames(self, deterministic=False):
        '''
        For each step, plot obs & rendered screen in one figure for making videoes
        '''
        state = self.env.reset()
        done = False
        score = [0]
        #self._make_combined_plot2(state, score, prob_actions)
        #self._make_combined_plot(state, score)


        while not done:
        #for i in range(1):
            prob_actions = self.predict_proba(state)
            action, _ = self.model.predict(state, deterministic=deterministic)
            state, reward, done, info = self.env.step(action)
            score += reward
            self._make_combined_plot2(state, score, prob_actions)
            #self._make_combined_plot(state, score)


    def _make_combined_plot2(self, state, score, prob_actions):
        '''
        Originally made for n_stack = 4 & n_skip = 4, SIMPLE_MOVEMENT
        '''
        # get rendered screen
        im_render = self.env.render(mode="rgb_array")

        n_stack = state.shape[-1]
        cmap = colors.ListedColormap(['red', 'skyblue', 'brown', 'blue'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        #obs_loc = [[0, 1], [0, 2], [1, 1], [1, 2]]
        obs_loc = [[0, 1], [1, 1], [2, 1], [3, 1]]
        obs_text = ['t (current frame)', 't-4', 't-8', 't-12']
        action_list = ['NOOP', 'right', 'right+A', 'right+B', 'right+A+B', 'A', 'left']


        ##########
        fig = plt.figure(dpi=100, figsize=(6, 6), constrained_layout=False, tight_layout=True)
        gs = fig.add_gridspec(4, 2, width_ratios=[3, 1])

        # individual obs frames
        for n in range(n_stack):
            ax = fig.add_subplot(gs[obs_loc[n][0], obs_loc[n][1]])
            im = ax.imshow(state[0,:,:,n], cmap=cmap, norm=norm)
            ax.set_axis_off()
            ax.text(-0.5, 14.5, obs_text[n])

        # prob_actions
        ax = fig.add_subplot(gs[3, 0])
        ax.bar(action_list, prob_actions[0])
        plt.xticks(rotation=45)
        ax.set_ylim(0, 1.05)

        # rendered screen
        ax = fig.add_subplot(gs[0:3, 0])
        im = ax.imshow(im_render)
        ax.set_axis_off()
        ax.text(0, -5, 'score: '+str(int(score[0])))

        plt.show()


    def _make_combined_plot(self, state, score):
        # get rendered screen
        im_render = self.env.render(mode="rgb_array")
        n_stack = state.shape[-1]

        cmap = colors.ListedColormap(['red', 'skyblue', 'brown', 'blue'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        #obs_text = ['t (current frame)', 't-4', 't-8', 't-12']

        fig = plt.figure(dpi=100, figsize=(5.5, 4), constrained_layout=False, tight_layout=True)
        gs = fig.add_gridspec(4, 2, width_ratios=[4, 1])

        # individual obs frames
        for n in range(n_stack):
            ax = fig.add_subplot(gs[n, 1])
            im = ax.imshow(state[0,:,:,n], cmap=cmap, norm=norm)
            ax.set_axis_off()

        # rendered screen
        ax = fig.add_subplot(gs[:, 0])
        im = ax.imshow(im_render)
        ax.set_axis_off()
        ax.text(0, -5, 'score: '+str(int(score[0])))

        plt.show()

    def make_animation(self, deterministic=True, filename='gym_animation.gif', RETURN_FRAMES=False):
        '''
        Make an animation of the rendered screen
        '''
        # run policy
        frames = []
        states = self.env.reset()
        done = False

        while not done:
            #frames.append(self.env.render(mode="rgb_array"))
            im = self.env.render(mode="rgb_array")
            frames.append(im.copy())
            action, _ = self.model.predict(states, deterministic=deterministic)
            states, reward, done, info = self.env.step(action)

        if RETURN_FRAMES == False:
            # make animation
            imageio.mimsave(filename, frames, fps=50)
        else: # make animation manually in case Mario gets stuck in the level and drags the animation for too long
            return frames