import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .agent_utils import RL_Data, Memory, PolicyNet, ValueNet
from LifeLongKT import POMDPDKT, related_loss
import random

class UniLPR_PPO(nn.Module):
    def __init__(
            self, device, state_dim, action_dim, max_len=300,hidden_dim=300, 
                 gamma=0.98, lr=5e-4, clip_epsilon=0.3, po_epochs=4, 
                 mini_batch_size=64, group_size=8, beta=0.1, add_goal=False,
                 kt_predict=True, seq_predict=False, kt_beta=0.1, seq_beta=0.2,
                 strategy='PPO'):
        super(UniLPR_PPO, self).__init__()
        self.gamma = gamma
        self.strategy = strategy
        self.clip_epsilon = clip_epsilon
        self.po_epochs = po_epochs
        self.mini_batch_size = mini_batch_size
        self.group_size = group_size
        self.beta = beta
        self.action_dim = action_dim
        self.memory = Memory(device)
        self.max_len = max_len

        self.add_goal = add_goal
        self.kt_predict = kt_predict
        self.seq_predict = seq_predict
        self.seq_beta = seq_beta
        self.kt_beta = kt_beta

        self.UniRec_model = POMDPDKT(action_dim, device, hidden_dim, hidden_dim,max_len,sigmoid_func=True)
        self.output_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(hidden_dim*2, hidden_dim))
        
        input_dim = hidden_dim
        self.value_net = ValueNet(input_dim, hidden_dim)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam([
            {'params': self.UniRec_model.parameters(), 'lr': lr*kt_beta}])
        
        self.ppo_optimizer = optim.Adam([
            {'params': self.value_net.parameters(), 'lr': lr},
            {'params': self.UniRec_model.parameters(), 'lr': lr*self.beta}
        ])
        
        self.device = device
        self.to(device)


    def begin_episode(self, learner_profile):
        self.item_embedding = self.UniRec_model.item_embedding.weight
        self.learner_goal = torch.as_tensor(learner_profile['targets'],dtype=torch.long, device=self.device)
        
        item_ids = torch.as_tensor(learner_profile["item_ids"], device=self.device)  
        corrects = torch.as_tensor(learner_profile["corrects"], device=self.device)
        batch = {
            "item_ids": item_ids.unsqueeze(0),
            "corrects": corrects.unsqueeze(0)
        }

        targets = item_ids[1:] 
        labels = corrects[:-1].float()
        self.train()

        outputs, question_output = self.UniRec_model(batch)
        self.current_state = batch

        self.hist_items = item_ids.clone() 
        self.hist_corrects = corrects.clone()

        self.kt_loss = torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)

        if self.kt_predict:
            loss += self.criterion(outputs.squeeze(), labels) 
        if self.seq_predict:
            loss += self.seq_beta * related_loss(question_output[0,:-1,:], targets, 1, self.action_dim, self.UniRec_model.item_embedding, self.device)

        if self.kt_predict or self.seq_predict:
            loss = loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def end_episode(self):
        if self.strategy == 'AC':
            actor_loss, critic_loss = self.learn_AC()
        else:
            actor_loss, critic_loss = self.learn_PPO()
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss
        }

    def step(self, observation=None):
        batch = {
            "item_ids":self.hist_items.unsqueeze(0), 
            "corrects":self.hist_corrects.unsqueeze(0)
        }
        if self.add_goal:
            batch["goal"]=self.learner_goal

        outputs, question_output = self.UniRec_model(batch)

        state_encoding = self.output_fc(question_output[0, -1, :])
        logits = torch.matmul(state_encoding, self.item_embedding.T)
        probs = F.softmax(logits,dim=-1)
        probs[0] = 0
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()

        return action
    
    def get_padded_batch(self):
        def pad_sequence(seq, max_len, pad_val=0):
            if len(seq) >= max_len:
                return seq[-max_len:]
            else:
                pad_len = max_len - len(seq)
                padding = torch.full((pad_len,), pad_val, dtype=seq.dtype, device=self.device)
                return torch.cat([padding, seq])
        a = 1 if self.add_goal else 0
        padded_items = pad_sequence(self.hist_items, self.max_len-a)
        padded_corrects = pad_sequence(self.hist_corrects, self.max_len-a)

        batch = {
            "item_ids": padded_items.unsqueeze(0),    
            "corrects": padded_corrects.unsqueeze(0)  
        }
        
        return batch

    def observe(self, reward, done, step_info):

        item_id = step_info.get('item_id')
        correct = step_info.get('correct')

        last_state = self.get_padded_batch()
        
        self.hist_items = torch.cat([
            self.hist_items,
            torch.as_tensor([item_id], dtype=torch.long, device=self.device)
        ])
        self.hist_corrects = torch.cat([
            self.hist_corrects,
            torch.as_tensor([correct], dtype=torch.long, device=self.device)
        ])
        
        if len(self.hist_items) > self.max_len:
            self.hist_items = self.hist_items[-self.max_len:]
            self.hist_corrects = self.hist_corrects[-self.max_len:]
        
        self.current_state = self.get_padded_batch()

        action = step_info['item_id']

        data = RL_Data(last_state, action, reward, self.current_state, done)
        self.memory.set(data)

    def learn_AC(self):
        states, actions, rewards, next_states, dones = self.memory.get()
        returns = torch.zeros_like(rewards)
        G = torch.tensor([0.0], device=self.device)
        for i in reversed(range(len(rewards))):
            reward = 0 if dones[i] else rewards[i] 
            G = reward + self.gamma * G * (1 - dones[i])
            returns[i] = G
        rewards = returns

        if self.add_goal:
            states["goal"] = self.learner_goal
            next_states["goal"] = self.learner_goal
            
        with torch.no_grad():
            if self.add_goal:
                states_encoding = self.UniRec_model(states)[1][:, -1, :]
                next_states_encoding = self.UniRec_model(next_states)[1][:, -1, :]
            else:
                _, states_encoding = self.UniRec_model(states)
                states_encoding = states_encoding[:, -1, :]
                next_states_encoding = states_encoding[:, -1, :]  

            current_values = self.value_net(states_encoding)
            next_values = self.value_net(next_states_encoding)
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            td_errors = td_targets - current_values
        
        pred_values = self.value_net(states_encoding)
        value_loss = F.mse_loss(pred_values, td_targets.detach())
        
        states_encoding = self.output_fc(states_encoding)
        logits = torch.matmul(states_encoding, self.item_embedding.T) 
        action_probs = F.softmax(logits, dim=-1)

        log_probs = torch.log(action_probs.gather(1, actions))
        
        policy_loss = -(log_probs * td_errors.detach().unsqueeze(1)).mean()
        loss = policy_loss + value_loss 
        
        self.ppo_optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.UniRec_model.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.optimizer.step()
        self.ppo_optimizer.step()
        
        self.memory.reset()
        
        return policy_loss.item(), value_loss.item()

    def learn_PPO(self):
        states, actions, rewards, next_states, dones = self.memory.get()
        if self.add_goal:
            states["goal"] = self.learner_goal
            next_states["goal"] = self.learner_goal

        with torch.no_grad():
            if self.add_goal:
                states_encoding = self.UniRec_model(states)[1][:, -1, :]
                next_states_encoding = self.UniRec_model(next_states)[1][:, -1, :]
            else:
                _, states_encoding = self.UniRec_model(states)

                states_encoding = states_encoding[:, -2, :]
                next_states_encoding = states_encoding[:, -1, :] 
            
            values = self.value_net(states_encoding)
            next_values = self.value_net(next_states_encoding)

            states_encoding = self.output_fc(states_encoding)
            logits = torch.matmul(states_encoding, self.item_embedding.T) 
            old_probs = F.softmax(logits, dim=-1)     

        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.po_epochs):  
            indices = torch.randperm(len(states['item_ids']))
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]
                
                batch_states = {
                    "item_ids": states['item_ids'][batch_idx],
                    "corrects": states['corrects'][batch_idx]
                }
                if self.add_goal:
                    batch_states["goal"] = self.learner_goal
                
                batch_actions = actions[batch_idx]
                batch_old_probs = old_probs[batch_idx].gather(1, batch_actions)
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                _, batch_states_encoding = self.UniRec_model(batch_states)
                batch_states_encoding = self.output_fc(batch_states_encoding[:, -1, :])
                new_probs = F.softmax(torch.matmul(batch_states_encoding, self.item_embedding.T), dim=-1)
                batch_new_probs = new_probs.gather(1, batch_actions)

                ratio = batch_new_probs / (batch_old_probs + 1e-8)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(self.value_net(batch_states_encoding), batch_returns)
                
                loss = policy_loss + value_loss 

                self.ppo_optimizer.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.UniRec_model.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.optimizer.step()
                self.ppo_optimizer.step()

        self.memory.reset()
        return policy_loss.item(), value_loss.item()
    
    def compute_gae(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            reward = rewards[t] if dones[t] else 0
            delta = reward + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns
    
