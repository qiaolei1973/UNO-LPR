import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .agent_utils import RL_Data, Memory, PolicyNet, ValueNet
from LifeLongKT import POMDPDKT, related_loss
import numpy as np
import random
        
class UNO(nn.Module):
    def __init__(self, device, state_dim, action_dim, max_len=300,hidden_dim=300,
                 gamma=0.98, lr=5e-4, clip_epsilon=0.3, po_epochs=4,
                 mini_batch_size=64, group_size=8, beta=0.1, add_goal=False,
                 kt_predict=True, seq_predict=False, kt_beta=0.1, seq_beta=0.2,
                 temperature=0.2, reinforcement=True):
        super(UNO, self).__init__()
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.po_epochs = po_epochs
        self.group_size = group_size
        self.beta = beta
        self.action_dim = action_dim
        self.memory = Memory(device)
        self.max_len = max_len
        
        self.temperature = temperature
        self.reinforcement = reinforcement

        self.add_goal = add_goal
        self.kt_predict = kt_predict
        self.seq_predict = seq_predict
        self.kt_beta = kt_beta
        self.seq_beta = seq_beta

        self.UniRec_model = POMDPDKT(action_dim, device, hidden_dim, hidden_dim,max_len,sigmoid_func=True)
        self.output_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(hidden_dim*2, hidden_dim))
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.kt_optimizer = optim.Adam([
            {'params': self.UniRec_model.parameters(), 'lr': lr}])
        
        self.optimizer = optim.Adam([
            {'params': self.UniRec_model.parameters(), 'lr': lr},
            {'params': self.output_fc.parameters(), 'lr': lr}])
    
        self.device = device
        self.to(device)


    def begin_episode(self, learner_profile):
        # KL(kt_loss) and KS(seq_loss):
            
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

        self.step_count = 0
        self.kt_loss = torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)

        if self.kt_predict:
            kt_loss = self.criterion(outputs.squeeze(), labels)
            loss += self.kt_beta * kt_loss
        if self.seq_predict:
            seq_loss = related_loss(question_output[0,:-1,:], targets, 1, self.action_dim, self.UniRec_model.item_embedding, self.device)
            loss += self.seq_beta * seq_loss

        if self.kt_predict or self.seq_predict:
            self.kt_optimizer.zero_grad()
            loss.backward()
            self.kt_optimizer.step()

        return loss.item()

    def end_episode(self):
        
        actor_loss, critic_loss = self.learn()
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss
        }

    def step(self, observation=None):
        # UniLPR model generates action:
            
        self.step_count = self.step_count + 1
        batch = {
            "item_ids":self.hist_items.unsqueeze(0),
            "corrects":self.hist_corrects.unsqueeze(0)
        }
        if self.add_goal:
            batch["goal"]=self.learner_goal
                
        
        outputs, question_output = self.UniRec_model(batch)

        state_encoding = self.output_fc(question_output[0, -1, :])
        logits = torch.matmul(state_encoding, self.item_embedding.T) / self.temperature
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

        data = RL_Data(last_state, action, max(-1,reward), self.current_state, done)
        self.memory.set(data)

    def learn(self):
        # GRPO optimize:
            
        if not self.reinforcement:
            return 0, 0

        states, actions, rewards, next_states, dones = self.memory.get()

        if self.add_goal:
            states["goal"]=self.learner_goal
            next_states["goal"]=self.learner_goal

        with torch.no_grad():
            if self.add_goal:
                states_encoding = self.UniRec_model(states)[1][:, -1, :]
            else:
                _, states_encoding = self.UniRec_model(states)
                states_encoding = states_encoding[:, -1, :]

            states_encoding = self.output_fc(states_encoding)
            logits = torch.matmul(states_encoding, self.item_embedding.T) / self.temperature
            old_probs = F.softmax(logits, dim=-1)

        advantages = rewards

        # Personalized Advantage Estimation:
        if rewards.shape[0] > 1:
            advantages[1:,:] = advantages[1:,:] - advantages[:-1,:]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = torch.clamp(advantages, 1, -1)

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
                if self.add_goal:
                    group_states["goal"] = self.learner_goal
                
                group_actions = actions[group_idx]
                group_old_probs = old_probs[group_idx].gather(1, group_actions)
                group_advantages = advantages[group_idx]

                _, group_states_encoding = self.UniRec_model(group_states)
                group_states_encoding = self.output_fc(group_states_encoding[:, -1, :])

                logits = torch.matmul(group_states_encoding, self.item_embedding.T) / self.temperature
                new_probs = F.softmax(logits, dim=-1)
                group_new_probs = new_probs.gather(1, group_actions)

                ratio = group_new_probs / (group_old_probs + 1e-8)

                policy_loss = -torch.min(ratio * group_advantages,
                                         torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
                                         * group_advantages).mean()

                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()
                policy_losses.append(policy_loss.item())



        self.memory.reset()
        return np.mean(policy_losses), 0

    def get_embeddings(self):
        if hasattr(self.UniRec_model, 'item_embedding'):
            return self.UniRec_model.item_embedding.weight.data
        elif hasattr(self, 'item_embedding'):
            return self.item_embedding.data
        else:
            raise NotImplementedError("UNO Agent doesn't have accessible embeddings")
