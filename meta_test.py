import gymnasium as gym
import numpy as np
from wrappers import FrankaObservationWrapper
from meta_agent import MetaAgent
from replay_buffer import Buffer


if __name__ == '__main__':
    max_episode_steps = 1100
    replay_buffer_size = 1_000_000
    tasks = ['microwave', 'top burner', 'hinge cabinet']
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    updates_per_steps = 4
    target_update_interval = 1
    fc1_dim = 512
    fc2_dim = 512
    lr = 0.0001
    batch_size = 64
    # for testing change live_test to true
    live_test = False
    generate_score = True

    if live_test:
        env = gym.make('FrankaKitchen-v1', max_episode_steps=max_episode_steps, tasks_to_complete=tasks, render_mode='human')
        env = FrankaObservationWrapper(env)
        meta = MetaAgent(env=env, tasks=tasks, max_episode_steps=max_episode_steps)
        meta.initialize_agents()
        meta.test()
        env.close()
    
    if generate_score:
        print(f'generating performance score')
        env = gym.make('FrankaKitchen-v1', max_episode_steps=max_episode_steps, tasks_to_complete=tasks)
        env = FrankaObservationWrapper(env)
        obs, _ = env.reset()
        obs_size = obs.shape[0]
        meta_agent = MetaAgent(env=env, tasks=tasks, max_episode_steps=max_episode_steps)
        meta_agent.initialize_agents()
        performance_score_epoches = 10
        total_score = 0

        for i in range(performance_score_epoches):
            score = meta_agent.test()
            total_score += score

        success_ratio = ((total_score / len(tasks)) / performance_score_epoches) * 100
        print(f'success ratio: {success_ratio:.2f}%')



    