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
    task = task.replace(' ', '_')
    env = gym.make('FrankaKitchen-v1', max_episode_steps=max_episode_steps, tasks_to_complete=[task],
                   render_mode='human', autoreset=False)
    env = FrankaObservationWrapper(env, goal=task)
    #print(env.env.env.env.env.model.opt.gravity)

    obs, _ = env.reset()
    # print(f'obs.shape = {obs.shape}, obs.shape[0] = {obs.shape[0]}')

    memory = Buffer(mem_size=1_000_000,  
                    input_dim=obs.shape[0], 
                    n_action=env.action_space.shape[0])
    
    #memory.load_from_csv(filename=f'checkpoints/human_memory_{task}.npz')

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
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        env.render()
                    action = controller.get_action()
            action = controller.get_action()

            if (action is not None):
                state_, reward, done, _, _ = env.step(action)
                mask = 1 if episode_step == max_episode_steps else float(not done)
                memory.store_transitions(state, action, reward, state_, mask)
                print(f'episode step: {episode_step}, reward: {reward}, successfully added \
                      {memory.mem_cnt - starting_memory_size} steps to memory. total: {memory.mem_cnt}')
                state = state_
                episode_step += 1
            time.sleep(0.05)
        
        memory.save_to_csv(filename=f'checkpoints/human_memory_{task}.npz')
    
    env.close()


