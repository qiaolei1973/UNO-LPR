import torch
import torch.nn as nn
import torch.nn.functional as F
from .agent_utils import RL_Data, Memory, PolicyNet, ValueNet
import random
import numpy as np
        
class AC(nn.Module):
    def __init__(self, device, state_dim, action_dim, hidden_dim=300,
                  gamma=0.98, lr=1e-5, beta=0.1, add_goal=True,
                  temperature=1):
        super(AC, self).__init__()
        self.gamma = gamma
        self.action_dim = action_dim
        self.memory = Memory(device)
        self.device = device
        

        self.add_goal = add_goal
        input_dim = state_dim * 2 if add_goal else state_dim

        self.policy_net = PolicyNet(input_dim, hidden_dim, action_dim, temperature).to(device)
        
        self.value_net = ValueNet(input_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                                lr=lr*beta)
        self.critic_optimizer = torch.optim.Adam(self.value_net.parameters(),
                                                 lr=lr)

    def step(self, state, candidates: list = []):
        if self.add_goal:
            state = torch.cat((state.unsqueeze(0), self.learner_goal), dim=-1)
        else:
            state = state.unsqueeze(0)

        probs = self.policy_net(state).squeeze()

        if len(candidates) == 0 or candidates is None:
            candidates = range(1, self.action_dim-1)

        candidate_probs = torch.gather(
            probs,
            0,
            torch.tensor(candidates).to(self.device)
        )  
        weights = candidate_probs.view(-1).detach().cpu().numpy()
        
        action = random.choices(
            candidates,
            weights=weights,
            k=1
        )[0]

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


    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.get()
        
        if self.add_goal:
            states = torch.cat((states, self.learner_goal.repeat((states.shape[0],1))), dim=-1)
            next_states = torch.cat((next_states, self.learner_goal.repeat((states.shape[0],1))), dim=-1)
        
        returns = torch.zeros_like(rewards)
        G = torch.tensor([0.0], device=states.device)
        for i in reversed(range(len(rewards))):
            G = dones[i] * rewards[i]  + self.gamma * G * (1 - dones[i])
            returns[i] = G
        rewards = returns

        td_target = rewards + self.gamma * self.value_net(next_states) * (1 - dones)
        states_value = self.value_net(states)

        critic_loss = torch.mean(
            F.mse_loss(states_value, td_target.detach()))
        
        td_delta = td_target - states_value
        probs = self.policy_net(states) 
        log_probs = torch.log(probs.squeeze(1).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        critic_loss.backward() 
        actor_loss.backward()  

        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        
        self.critic_optimizer.step()  
        self.actor_optimizer.step()  

        self.memory.reset()
       
        return actor_loss.item(), critic_loss.item()
