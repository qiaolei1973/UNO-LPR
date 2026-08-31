import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .agent_utils import RL_Data, Memory, PolicyNet, ValueNet
from LifeLongKT import related_loss, SASRec_KT, GRU4Rec_KT
import numpy as np
import random

class o_GRU4Rec(nn.Module):
    def __init__(self, device, state_dim, action_dim, hidden_dim=300, lr=0.001):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.item_embedding = nn.Embedding(action_dim + 1, state_dim // 2, padding_idx=0)
        self.correct_embedding = nn.Embedding(2, state_dim // 2)

        self.gru_cell = nn.GRUCell(state_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.to(device)

    def embed_interaction(self, item_ids, corrects):
        return torch.cat([
            self.item_embedding(item_ids),
            self.correct_embedding(corrects)
        ], dim=-1)

    def begin_episode(self, user_profile):
        item_ids = torch.as_tensor(user_profile["item_ids"], dtype=torch.long, device=self.device)
        corrects = torch.as_tensor(user_profile["corrects"], dtype=torch.long, device=self.device)
        
        inputs = self.embed_interaction(item_ids[:-1], corrects[:-1])
        targets = item_ids[1:]
        
        self.train()
        h = torch.zeros(1, self.hidden_dim, device=self.device)
        
        hidden_states = []
        for t in range(len(inputs)):
            h = self.gru_cell(inputs[t].unsqueeze(0), h) 
            hidden_states.append(h)
        self.current_h = h

        logits = self.fc(torch.cat(hidden_states).squeeze(1))
        
        loss = self.criterion(logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.init_inference()
        return loss.item()

    def init_inference(self):
        self.eval()
    
    def observe(self, reward, step_done, step_info):
        item_id = step_info.get('item_id')
        correct = step_info.get('correct')
        interaction = self.embed_interaction(
                    torch.as_tensor([item_id], dtype=torch.long, device=self.device),
                    torch.as_tensor([correct], dtype=torch.long, device=self.device)
                )
        print(self.current_h[0,:8])
        self.current_h = self.gru_cell(interaction, self.current_h)
        

    def step(self, observation):
        with torch.no_grad():
            return torch.argmax(self.fc(self.current_h)).item()
        
    def end_episode(self):
        return {
            'actor_loss': 0,
            'critic_loss': 0
        }
    

class GenRec(nn.Module):
    def __init__(self, device, state_dim, action_dim, max_len=300,hidden_dim=300, 
                 gamma=0.98, lr=5e-4, clip_epsilon=0.3, po_epochs=4, 
                 mini_batch_size=64, group_size=8, beta=0.1, seq_beta=0.2,
                   reinforcement=False,temperature=0.1,
                   GenRec_model=None):
        super(GenRec, self).__init__()

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.po_epochs = po_epochs
        self.mini_batch_size = mini_batch_size
        self.group_size = group_size 
        self.beta = beta 
        self.action_dim = action_dim
        self.memory = Memory(device)
        self.max_len = max_len
        self.reinforcement = reinforcement
        self.seq_beta = seq_beta
        self.temperature = temperature

        self.UniRec_model = GenRec_model
        self.output_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(hidden_dim*2, hidden_dim))

        self.criterion = nn.BCEWithLogitsLoss()
        self.kt_optimizer = optim.Adam([
            {'params': self.UniRec_model.parameters(), 'lr': lr*self.seq_beta}])
        
        self.optimizer = optim.Adam([
            {'params': self.UniRec_model.parameters(), 'lr': lr},
            {'params': self.output_fc.parameters(), 'lr': lr}])
    
        self.device = device
        self.to(device)


    def begin_episode(self, learner_profile):
        self.train()
        self.item_embedding = self.UniRec_model.item_embedding.weight
        self.learner_goal = torch.as_tensor(learner_profile['targets'],dtype=torch.long, device=self.device)
        
        item_ids = torch.as_tensor(learner_profile["item_ids"], device=self.device)  
        corrects = torch.as_tensor(learner_profile["corrects"], device=self.device)
        batch = {
            "item_ids": item_ids.unsqueeze(0),
            "corrects": corrects.unsqueeze(0)
        }

        targets = item_ids[1:] 
        self.train()

        question_output = self.UniRec_model(batch)
        self.current_state = batch

        self.hist_items = item_ids.clone() 
        self.hist_corrects = corrects.clone()


        loss = related_loss(question_output[0,:-1,:], targets, 10, self.action_dim, self.UniRec_model.item_embedding, self.device)

        self.kt_optimizer.zero_grad()
        loss.backward()
        self.kt_optimizer.step()
        if not self.reinforcement:
            self.eval()

        return loss.item()

    def end_episode(self):
        if self.reinforcement:
            actor_loss, critic_loss = self.learn()
        else:
            actor_loss, critic_loss = 0, 0
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss
        }

    def step(self, observation=None):
        batch = {
            "item_ids":self.hist_items.unsqueeze(0), 
            "corrects":self.hist_corrects.unsqueeze(0)
        }

        question_output = self.UniRec_model(batch)[0, -1, :]
        state_encoding = self.output_fc(question_output)
        
        logits = torch.matmul(state_encoding, self.item_embedding.T) / self.temperature
        probs = F.softmax(logits,dim=-1)
        action = torch.argmax(probs).item()
        

        while action == 0:
            action = random.randint(1, self.action_dim-1)
            
        return action
    
    def get_padded_batch(self):
        def pad_sequence(seq, max_len, pad_val=0):
            if len(seq) >= max_len:
                return seq[-max_len:]
            else:
                pad_len = max_len - len(seq)
                padding = torch.full((pad_len,), pad_val, dtype=seq.dtype, device=self.device)
                return torch.cat([padding, seq])

        padded_items = pad_sequence(self.hist_items, self.max_len)
        padded_corrects = pad_sequence(self.hist_corrects, self.max_len)

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

        data = RL_Data(last_state, action, max(-1,reward), self.current_state, done)
        self.memory.set(data)

    def learn(self):
        
        states, actions, rewards, next_states, dones = self.memory.get()

        with torch.no_grad():
            question_output = self.UniRec_model(states)[:, -1, :]
            state_encoding = self.output_fc(question_output)
            
            logits = torch.matmul(state_encoding, self.item_embedding.T) / self.temperature
            old_probs = F.softmax(logits,dim=-1)
            
        returns = rewards
        advantages = returns
        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses = []
        for _ in range(self.po_epochs):
            indices = torch.randperm(len(states['item_ids']))
            
            for start in range(0, len(states), self.group_size):
                end = start + self.group_size
                group_idx = indices[start:end]
                
                group_states = {
                    "item_ids": states['item_ids'][group_idx],
                    "corrects": states['corrects'][group_idx]
                }
                
                group_actions = actions[group_idx]
                group_old_probs = old_probs[group_idx].gather(1, group_actions)
                group_advantages = advantages[group_idx]

                group_states_encoding = self.UniRec_model(group_states)[:, -1, :]
                group_states_encoding = self.output_fc(group_states_encoding)

                logits = torch.matmul(group_states_encoding, self.item_embedding.T) / self.temperature
                new_probs = F.softmax(logits,dim=-1)

                group_new_probs = new_probs.gather(1, group_actions)

                ratio = group_new_probs / (group_old_probs + 1e-8)
                
                group_ratio_mean = ratio.mean().detach()
                group_ratio_constraint = F.mse_loss(ratio, group_ratio_mean.expand_as(ratio))
                
                policy_loss = -torch.min(ratio * group_advantages,
                                          torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) 
                                            * group_advantages).mean() 
                + self.beta * group_ratio_constraint

                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()

                policy_losses.append(policy_loss.item())


        self.memory.reset()
        return np.mean(policy_losses), 0
    

class GRU4Rec(GenRec):
    def __init__(self, device, state_dim, action_dim, max_len=300, hidden_dim=300, 
                 n_layers=1, dropout=0.01, gamma=0.98, lr=5e-4, clip_epsilon=0.3, 
                 po_epochs=4, mini_batch_size=64, group_size=8, beta=0.1, 
                 seq_beta=0.2, reinforcement=False, temperature=0.1):

        gru_model = GRU4Rec_KT(action_dim, device, hidden_dim, hidden_dim, dropout, n_layers)
        
        super(GRU4Rec, self).__init__(
            device=device,
            state_dim=state_dim,
            action_dim=action_dim,
            max_len=max_len,
            hidden_dim=hidden_dim,
            gamma=gamma,
            lr=lr,
            clip_epsilon=clip_epsilon,
            po_epochs=po_epochs,
            mini_batch_size=mini_batch_size,
            group_size=group_size,
            beta=beta,
            seq_beta=seq_beta,
            reinforcement=reinforcement,
            temperature=temperature,
            GenRec_model=gru_model
        )
   
    
class SASRec(GenRec):
    def __init__(self, device, state_dim, action_dim, max_len=300,hidden_dim=300, 
                 n_layers=2, n_heads=2, dropout=0.01,
                 gamma=0.98, lr=5e-4, clip_epsilon=0.3, po_epochs=4, 
                 mini_batch_size=64, group_size=8, beta=0.1, seq_beta=0.2, 
                 reinforcement=False, temperature=0.1):

        sasrec_model = SASRec_KT(action_dim, device, hidden_dim, hidden_dim,max_len+10,dropout, n_layers)
        
        super(SASRec, self).__init__(
            device=device,
            state_dim=state_dim,
            action_dim=action_dim,
            max_len=max_len,
            hidden_dim=hidden_dim,
            gamma=gamma,
            lr=lr,
            clip_epsilon=clip_epsilon,
            po_epochs=po_epochs,
            mini_batch_size=mini_batch_size,
            group_size=group_size,
            beta=beta,
            seq_beta=seq_beta,
            reinforcement=reinforcement,
            temperature=temperature,
            GenRec_model=sasrec_model
        )
        

                


    
    