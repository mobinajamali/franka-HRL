import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper

class FrankaObservationWrapper(ObservationWrapper):
    def __init__(self, env, goal='microwave'):
        super(FrankaObservationWrapper, self).__init__(env)
        self.goal = goal
        env_model = env.env.env.env.model
        env_model.opt.gravity[:] = [0, 0, -1]

    #def set_goal(self, goal):
    #    self.goal = goal

    def reset(self):
        observation, info = self.env.reset()
        observation = self.process_observation(observation)
        return observation, info
    
    def step(self, action):
        observation, reward, done, trunc, info = self.env.reset()
        observation = self.process_observation(observation)
        return observation, reward, done, trunc, info

    def process_observation(self, observation):
        obs_pos = observation['observation']
        obs_achieved_goal = observation['achieved_goal']
        obs_desired_goal = observation['desired_goal']
        obs_concat = np.concatenate((obs_pos, obs_achieved_goal[self.goal], obs_desired_goal[self.goal]))
        return obs_concat
