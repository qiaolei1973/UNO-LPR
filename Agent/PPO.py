import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .agent_utils import RL_Data, Memory, PolicyNet, ValueNet

class PPO(nn.Module):
    def __init__(self, device, state_dim, action_dim, hidden_dim=300, 
                 gamma=0.98, lr=1e-4, beta=0.1, clip_epsilon=0.2, 
                 ppo_epochs=4, mini_batch_size=64,
                 add_goal=True, temperature=1):
        
        super(PPO, self).__init__()
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.action_dim = action_dim
        self.memory = Memory(device)

        self.add_goal = add_goal
        input_dim = state_dim * 2 if add_goal else state_dim
        
        self.policy_net = PolicyNet(input_dim, hidden_dim, action_dim,temperature)
        self.value_net = ValueNet(input_dim, hidden_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': lr*beta},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])
        self.device = device
        self.to(device)
        
    def step(self, state, candidates: list = []):
        if self.add_goal:
            state = torch.cat((state.unsqueeze(0), self.learner_goal), dim=-1)
        else:
            state = state.unsqueeze(0)

        probs = self.policy_net(state).squeeze()
        probs[0] = 0
        action_dist = torch.distributions.Categorical(probs)
        
        action = action_dist.sample().item()

        while action == 0:
            action = action_dist.sample().item()

        return action
    

    def begin_episode(self, learner_profile):
        self.learner_goal = torch.sum(learner_profile['targets_embedding'],dim=-2)
        
    def end_episode(self):
        actor_loss, critic_loss = self.learn()
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss
        }
    
    def observe(self, reward, done, step_info):
   
        state = step_info['last_state']
        next_state = step_info['current_state']
        action = step_info['item_id']
        data = RL_Data(state, action, reward, next_state, done)
        self.memory.set(data)

    def compute_gae(self, rewards, values, next_values, dones):
        """计算GAE(Generalized Advantage Estimation)"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t]*dones[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae  # 0.95是GAE lambda参数
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def learn(self, states=None, actions=None, rewards=None, next_states=None, dones=None):
        """PPO的核心学习逻辑"""
        states, actions, rewards, next_states, dones = self.memory.get()
        # print('high: ',states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        if self.add_goal:
            states = torch.cat((states, self.learner_goal.repeat((states.shape[0],1))), dim=-1)
            next_states = torch.cat((next_states, self.learner_goal.repeat((states.shape[0],1))), dim=-1)
        
        with torch.no_grad():
            values = self.value_net(states)
            next_values = self.value_net(next_states)
        
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
   
        # 存储旧策略的概率
        probs = self.policy_net(states)
        old_probs = probs.gather(1, actions).detach()

        pl, vl=[], []
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]
                
                # 获取mini-batch数据
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_probs = old_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                # 计算新策略概率
                new_probs = self.policy_net(mb_states).gather(1, mb_actions)
                ratio = new_probs / (mb_old_probs + 1e-8)
                
                # 计算策略损失
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = F.mse_loss(self.value_net(mb_states), mb_returns)
                
                # 计算总损失
                loss = policy_loss + value_loss  # 0.5是价值损失系数
                
                # 优化步骤
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
                pl.append(policy_loss.item())
                vl.append(value_loss.item())
        
        self.memory.reset()
        return sum(pl)/len(pl), sum(vl)/len(vl)