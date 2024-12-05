import gymnasium as gym
import numpy as np
import time
from wrappers import FrankaObservationWrapper
from agent import Agent
from replay_buffer import Buffer
from networks import Actor, Critic

if __name__ == '__main__':
    max_episode_steps = 500
    replay_buffer_size = 1_000_000
    task = 'microwave'
    task = task.replace(" ", "_")
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    updates_per_steps = 4
    target_update_interval = 1
    fc1_dim = 512
    fc2_dim = 512
    lr = 0.0001
    batch_size = 64

    env = gym.make('FrankaKitchen-v1', max_episode_steps=max_episode_steps, tasks_to_complete=[task], render_mode='human')
    env = FrankaObservationWrapper(env, goal=task)
    obs, info = env.reset()
    obs_size = obs.shape[0]
    agent = Agent(obs_size, env.action_space, gamma=0.99, fc1_dim=512, fc2_dim=512,
                  lr=0.0001, tau=0.005, alpha=0.1, target_update_interval=1, goal=task)
    memory = Buffer(replay_buffer_size,input_dim=obs_size, n_action=env.action_space.shape[0],
                     augment_rewards=True, augment_data=True)
    agent.load_models(evaluate=True)
    agent.test(env=env, episodes=3, max_episode_steps=max_episode_steps)
    env.close()
    