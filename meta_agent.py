import torch as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import numpy as np
import os
import random
from agent import Agent
from replay_buffer import Buffer
from wrappers import FrankaObservationWrapper

class MetaAgent(object):
    def __init__(self, env: FrankaObservationWrapper, goal_list=['microwave'],
                 replay_buffer_size=1_000_000, max_episode_steps=500):
        self.agent_dict = {}
        self.mem_dict = {}
        goal_list_no_spaces = [a.replace(" ", "_") for a in goal_list]
        self.goal_dict = dict(zip(goal_list_no_spaces, goal_list))
        self.env = env
        self.agent: Agent = None
        self.replay_buffer_size = replay_buffer_size
        self.max_episode_steps = max_episode_steps

    def initialize_memory(self, augment_rewards=True, augment_data=True, augment_noise_ratio=0.1):
        for goal in self.goal_dict:
            self.env.set_goal(self.goal_dict[goal])
            obs, _ = self.env.reset()
            obs_size = obs.shape[0]

            memory = Buffer(self.replay_buffer_size, input_size=obs_size, augment_noise_ratio=augment_noise_ratio,
                                  n_actions=self.env.action_space.shape[0], augment_rewards=augment_rewards,
                                  augment_data=augment_data, expert_data_ratio=0)
            
            self.mem_dict[goal] = memory
        
    def load_memory(self):
        for buffer in self.mem_dict:
            self.mem_dict[buffer].load_from_csv(filename=f"checkpoints/controller_{buffer}_memory.npz")


    def initialize_agents(self, gamma=0.99, tau=0.005, alpha=0.1,
                          target_update_interval=2, fc1_dim=512, fc2_dim=512,
                          lr=0.0001):
        
        for goal in self.goal_dict:
            self.env.set_goal(self.goal_dict[goal])
            obs, _ = self.env.reset()
            obs_size = obs.shape[0]

            agent = Agent(obs_size, self.env.action_space, gamma=gamma, tau=tau, alpha=alpha,
                          target_update_interval=target_update_interval, fc1_dim=fc1_dim, fc2_dim=fc2_dim,
                          lr=lr, goal=goal)
            
            print(f"Loading checkpoint for {goal}")

            agent.load_models(evaluate=True)

            self.agent_dict[goal] = agent

    def save_models(self):
        for agent in self.agent_dict:
            self.agent_dict[agent].save_models()


    def train(self, episodes, batch_size=64, summary_writer_name='meta_agent'):
        
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_' + summary_writer_name
        writer = SummaryWriter(summary_writer_name)

        updates = 0
        for episode in range(episodes):

            last_action = None
            action = None
            episode_reward = 0
            episode_steps = 0
            print(f"Starting episode: {episode}")

            num_samples = random.choice([1, 2])
            for goal in random.sample(list(self.goal_dict.keys()), num_samples):

                done = False
                self.env.set_goal(self.goal_dict[goal])
                state, _ = self.env.reset()

                while not done and episode_steps < self.max_episode_steps:

                    if last_action is not None:
                        action = last_action
                        last_action = None
                    else:
                        action = self.agent_dict[goal].choose_action(state)
                    
                    if self.mem_dict[goal].ready(batch_size=64):
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent_dict[goal].update_parameters(self.mem_dict[goal],
                                                                                                                             batch_size,
                                                                                                                             updates)
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        updates += 1
                    
                    next_state, reward, done, _, _ = self.env.step(action)

                    if reward == 1:
                        done = True
                        last_action = action
                    
                    episode_steps += 1
                    episode_reward += reward

                    mask = 1 if episode_steps == self.max_episode_steps else float(not done)

                    self.mem_dict[goal].store_transitions(state, action, reward, next_state, mask)

                    state = next_state

            episode_reward = episode_reward / num_samples

            writer.add_scalar('reward/train', episode_reward, episode)
            writer.add_scalar('reward/episode_steps', episode_steps, episode)
            print(f"Episode: {episode}, Episode steps: {episode_steps}, reward: {episode_reward}")

            if episode % 10 == 0:
                self.save_models()

    def test(self):
        action = None
        episode_reward = 0

        for goal in self.goal_dict:
            print(f"Attempting goal {goal}...")
            self.env.set_goal(self.goal_dict[goal])
            self.agent = self.agent_dict[goal]

            action, reward = self.agent.test(env=self.env, episodes=1, max_episode_steps=self.max_episode_steps, prev_action=action)

            episode_reward += reward
        
        return episode_reward


