from wrappers import FrankaObservationWrapper
from replay_buffer import Buffer
from controller import FrankaController
import gymnasium as gym
import pygame
import numpy as np
import time

if __name__ == '__main__':
    max_episode_steps = 500
    task = 'microwave'
    task_no_spaces = task.replace(" ", "_")
    env = gym.make('FrankaKitchen-v1', max_episode_steps=max_episode_steps, tasks_to_complete=[task],
                   render_mode='human', autoreset=False)
    env = FrankaObservationWrapper(env, goal=task)
    #print(env.env.env.env.env.model.opt.gravity)

    obs, _ = env.reset()
    # print(f'obs.shape = {obs.shape}, obs.shape[0] = {obs.shape[0]}')

    memory = Buffer(mem_size=1_000_000,  
                    input_dim=obs.shape[0], 
                    n_action=env.action_space.shape[0])
    
    #memory.load_from_csv(filename=f'checkpoints/controller_{task_no_spaces}_memory.npz')

    starting_memory_size = memory.mem_cnt
    print(f'starting memory size is {starting_memory_size}')
    controller = FrankaController()
    while True:
        episode_step = 0
        done = False
        obs, info = env.reset()
        while not done and episode_step < max_episode_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            action = controller.get_action()
    
            if (action is not None):
                state_, reward, done, trunc, info = env.step(action)
                mask = 1 if episode_step == max_episode_steps else float(not done)
                memory.store_transitions(obs, action, reward, state_, mask)
                print(f'episode step: {episode_step}, reward: {reward}, successfully added {memory.mem_cnt - starting_memory_size} steps to memory. total: {memory.mem_cnt}')
                state = state_
                episode_step += 1
            time.sleep(0.05)
        
        memory.save_to_csv(filename=f'checkpoints/controller_{task_no_spaces}_memory.npz')
    
    env.close()


