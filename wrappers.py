import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper

class FrankaObservationWrapper(ObservationWrapper):
    def __init__(self, env, goal='microwave'):
        super(FrankaObservationWrapper, self).__init__(env)
        self.goal = goal
        # modify graviry for human input purpose
        env_model = env.env.env.env.model
        env_model.opt.gravity[:] = [0, 0, -1]

    def set_goal(self, goal):
        # to be able to change the goal of env on fly
        self.goal = goal

    def reset(self):
        # account for precess obs
        observation, info = self.env.reset()
        observation = self.process_observation(observation)
        return observation, info
    
    def step(self, action):
        # account for precess obs
        observation, reward, done, trunc, info = self.env.step(action)
        observation = self.process_observation(observation)
        return observation, reward, done, trunc, info

    def process_observation(self, observation):
        # concate all info for each goal task
        obs_pos = observation['observation']
        obs_achieved_goal = observation['achieved_goal']
        obs_desired_goal = observation['desired_goal']
        obs_concat = np.concatenate((obs_pos, obs_achieved_goal[self.goal], obs_desired_goal[self.goal]))
        return obs_concat
