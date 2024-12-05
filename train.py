import gymnasium as gym
import numpy as np
import time
from wrappers import FrankaObservationWrapper
from agent import Agent
from replay_buffer import Buffer

if __name__ == '__main__':
    max_episode_steps = 500
    replay_buffer_size = 1_000_000
    task = 'microwave'
    task_no_spaces = task.replace(" ", "_")
    #print(f'checkpoints/controller_{task_no_spaces}_memory.npz')
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    updates_per_steps = 4
    target_update_interval = 1
    fc1_dim = 512
    fc2_dim = 512
    lr = 0.0001
    batch_size = 64

    env = gym.make('FrankaKitchen-v1', max_episode_steps=max_episode_steps, tasks_to_complete=[task])
    env = FrankaObservationWrapper(env, goal=task)
    obs, info = env.reset()
    obs_size = obs.shape[0]
    print(f'obs_size: {obs_size}')
    agent = Agent(obs_size, env.action_space, gamma=0.99, fc1_dim=512, fc2_dim=512,
                  lr=0.0001, tau=0.005, alpha=0.1, target_update_interval=1, goal=task_no_spaces)
    memory = Buffer(replay_buffer_size,input_dim=obs_size, n_action=env.action_space.shape[0],
                     augment_rewards=True, augment_data=True)
    memory.load_from_csv(filename=f'checkpoints/controller_{task_no_spaces}_memory.npz')
    
    time.sleep(2)

    # training phases
    memory.expert_data_ratio = 0.5
    agent.train(env=env, memory=memory, episodes=150, batch_size=batch_size,
                updates_per_step=updates_per_steps, summary_writer_name=f'live_train_phase_1_{task_no_spaces}',
                max_episode_steps=max_episode_steps)
    
    memory.expert_data_ratio = 0.25
    agent.train(env=env, memory=memory, episodes=250, batch_size=batch_size,
                updates_per_step=updates_per_steps, summary_writer_name=f'live_train_phase_2_{task_no_spaces}',
                max_episode_steps=max_episode_steps)
    
    memory.expert_data_ratio = 0
    agent.train(env=env, memory=memory, episodes=500, batch_size=batch_size,
                updates_per_step=updates_per_steps, summary_writer_name=f'live_train_phase_3_{task_no_spaces}',
                max_episode_steps=max_episode_steps)
    


