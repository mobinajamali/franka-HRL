import torch as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import numpy as np
import os
from networks import Actor, Critic
from replay_buffer import Buffer
from wrappers import FrankaObservationWrapper

class Agent(object):
    def __init__(self, input_dim, action_space, fc1_dim, fc2_dim, goal, lr, 
                 target_update_interval, alpha, tau, gamma):
        
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval

        self.critic = Critic(input_dim=input_dim, n_action=action_space.shape[0], fc1_dim=fc1_dim, fc2_dim=fc2_dim, lr=lr, name=f'critic_{goal}')
        self.critic_target = Critic(input_dim=input_dim, n_action=action_space.shape[0], fc1_dim=fc1_dim, fc2_dim=fc2_dim, lr=lr, name=f'critic_target_{goal}')
        self.actor = Actor(input_dim=input_dim, n_action=action_space.shape[0], fc1_dim=fc1_dim, fc2_dim=fc2_dim, lr=lr, action_space=action_space, name=f'actor_{goal}')
        self.update_network_parameters(self.critic_target, self.critic, tau=1)


    def choose_action(self, observation, evaluate=False):
        state = T.FloatTensor(observation).to(self.actor.device).unsqueeze(0)
        if evaluate is False:
            actions, _, _ = self.actor.sample_normal(state, reparametrization=False)
        else:
            _, _, actions =  self.actor.sample_normal(state, reparametrization=False)
        return actions.detach().cpu().numpy()[0]
    

    def learn(self, memory: Buffer, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_memory(batch_size)

        state_batch = T.FloatTensor(state_batch).to(self.actor.device)
        action_batch = T.FloatTensor(action_batch).to(self.actor.device)
        next_state_batch = T.FloatTensor(next_state_batch).to(self.actor.device)
        reward_batch = T.FloatTensor(reward_batch).to(self.actor.device).unsqueeze(1)
        mask_batch = T.FloatTensor(mask_batch).to(self.actor.device).unsqueeze(1)

        # compute critic loss
        with T.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample_normal(next_state_batch, reparametrization=False)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = T.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic.optimizer.zero_grad()
        qf_loss.backward()
        self.critic.optimizer.step()

        # compute policy loss
        pi, log_pi, _ = self.actor.sample_normal(state_batch, reparametrization=True)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = T.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        alpha_loss = T.tensor(0.).to(self.actor.device)
        alpha_tlogs = T.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            # define soft update function
            self.update_network_parameters(self.critic_target, self.critic)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def update_network_parameters(self, target, source, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau) 


    def train(self, env: FrankaObservationWrapper, memory: Buffer, episodes=1000, batch_size=64, updates_per_step=1, summary_writer_name='', max_episode_steps=100):
        summary_writer_name = f"runs/{datetime.datetime.now()}_" + summary_writer_name
        writer = SummaryWriter(summary_writer_name)

        total_numsteps = 0
        updates = 0

        for episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            state, _ = env.reset()

            while not done and episode_steps < max_episode_steps:
                action = self.choose_action(state)
                if memory.ready(batch_size=batch_size):
                    for i in range(updates_per_step):
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.learn(memory,
                                                                                                batch_size,
                                                                                                updates)
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        updates += 1
                next_state, reward, done, _, _ = env.step(action)
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                mask = 1 if episode_steps == max_episode_steps else float(not done)
                memory.store_transitions(state, action, reward, next_state, mask)
                state = next_state
            writer.add_scalar('reward/train', episode_reward, episode)
            print(f'episode:{episode}, total numsteps:{total_numsteps}, episode steps:{episode_steps}, reward:{round(episode_reward, 2)}')

            if episode % 10 == 0:
                self.save_models()

    def test(self, env: FrankaObservationWrapper, episodes=1,
             max_episode_steps=500, prev_action=None):
        for episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            if prev_action is not None:
                state, reward, done, _, _ = env.step(prev_action)
            else:
                state, _ = env.reset()

            while not done and episode_steps < max_episode_steps:
                action = self.choose_action(state, evaluate=True)
                next_state, reward, done, _, _ = env.step(action)
                episode_steps += 1
                if reward == 1:
                    done = True
                    prev_action = action
                episode_reward += reward

                mask = 1 if episode_steps == max_episode_steps else float(not done)
                state = next_state

                if env.env.render_mode == 'human':
                    time.sleep(0.05)
            
            print(f'episode:{episode}, episode steps:{episode_steps}, reward:{round(episode_reward, 2)}')
            return prev_action, episode_reward

    def save_models(self):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        print('..... saving models .....')
        self.actor.save_checkpoints()
        self.critic.save_checkpoints()
        self.critic_target.save_checkpoints()

    def load_models(self, evaluate=False):
        try:
            print('..... loading models .....')
            self.actor.load_checkpoints()
            self.critic.load_checkpoints()
            self.critic_target.load_checkpoints()
            print('..... successfully loaded models .....')
        except:
            if evaluate:
                raise Exception('Could not load models and evaluate models')
            else:
                print('Coulf not load models. starting from the scratch')
        if evaluate:
            self.actor.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.actor.train()
            self.critic.train()
            self.critic_target.train()


