import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from collections import deque
from .agent_utils import RL_Data, Memory, PolicyNet, ValueNet
from LifeLongKT import DKT
import yaml
import os
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import random

class CognitiveNavigator:
    def __init__(self, graph, k=2):
        self.graph = graph
        self.k = k
        nx.draw(graph, with_labels=True)

        plt.savefig('graph.png')

        
    def get_candidates(self, central_focus:int, target):
        central_focus = str(central_focus)
        D = set([central_focus])
        Q = deque()
        
        visited = set()
        queue = deque([(central_focus, 0)]) 
        while queue:
            node, hop = queue.popleft()
            if hop >= self.k - 1:
                continue
            if node not in visited:
                visited.add(node)
                for successor in self.graph.successors(node):
                    if successor not in visited:
                        D.add(successor)
                        queue.append((successor, hop + 1))
        
        visited = set()
        queue = deque([(central_focus, 0)])
        
        while queue:
            node, hop = queue.popleft()
            if hop >= self.k - 1:
                continue
            if node not in visited:
                visited.add(node)
                for predecessor in self.graph.predecessors(node):
                    if predecessor not in visited:
                        Q.append(predecessor)
                        queue.append((predecessor, hop + 1))
        
        while Q:
            q = Q.popleft()
            D.add(q)
            for neighbor in self.graph.successors(q):
                D.add(neighbor)
            for neighbor in self.graph.predecessors(q):
                D.add(neighbor)

        valid_candidates = set()
        for target_item in target.tolist():
            for node in D:
                if node not in valid_candidates:
                    if nx.has_path(self.graph, node, str(target_item)) or  nx.has_path(self.graph, str(target_item), node):
                        valid_candidates.add(int(node))

        return list(valid_candidates)

class CSEAL(nn.Module):
    def __init__(self, device, state_dim, action_dim, 
                 hidden_dim=300, gamma=0.98, lr=1e-5, beta=0.1, add_goal=True, 
                 graph_path='./data/processed_data/junyi/prerequest_graph.gexf',
                 model_path='./LifeLongKT/saved_models/DKT/junyi/xxx_pretrained.pth',
                 config_path="./LifeLongKT/configs/DKT.yaml",
                 temperature=0.1):
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        
        
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.action_dim = action_dim
        self.memory = Memory(device)
        self.add_goal = True

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.dkt = DKT(action_dim, device, **config['model']).to(device)
        
        assert model_path and os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=device)
        self.dkt.load_state_dict(state_dict)

        graph = nx.read_gexf(graph_path)
        self.navigator = CognitiveNavigator(graph)
        
        input_dim = state_dim * 2 if add_goal else state_dim
        self.policy_net = PolicyNet(input_dim, hidden_dim, action_dim, temperature).to(device)
        self.value_net = ValueNet(input_dim, hidden_dim).to(device)
        
        self.actor_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=lr*beta)
        self.critic_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=lr)
        


    def begin_episode(self, learner_profile):
        self.learner_goal = torch.as_tensor(learner_profile['targets'],dtype=torch.long, device=self.device) 

        self.learner_goal_embedding = self.dkt.embedding_layer(self.learner_goal.unsqueeze(0)).sum(dim=-2).squeeze()
    
        item_ids = torch.as_tensor(learner_profile["item_ids"], device=self.device)  
        corrects = torch.as_tensor(learner_profile["corrects"], device=self.device)
        batch = {
            "item_ids": item_ids.unsqueeze(0),
            "corrects": corrects.unsqueeze(0)
        }

        self.current_state = self.dkt(batch, True)[0,-1,:]
        
        if self.add_goal:
            self.current_state = torch.concat([self.current_state, self.learner_goal_embedding], dim=-1)

        self.hist_items = item_ids.clone() 
        self.hist_corrects = corrects.clone()

        self.central_focus = int(item_ids[-1].item())
        self.hist_items = item_ids.clone() 
        self.hist_corrects = corrects.clone()
        
    
    def step(self, state):
        candidates = self.navigator.get_candidates(
            self.central_focus, self.learner_goal)
        
        probs = self.policy_net(self.current_state)

    
        if len(candidates) == 0: 
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

        if action == 0:
            action = random.randint(1,self.action_dim-1)
    
        return action

    def observe(self, reward, done, step_info):
        item_id = step_info.get('item_id')
        correct = step_info.get('correct')
        last_state = self.current_state

        self.hist_items = torch.cat([
            self.hist_items,
            torch.as_tensor([item_id], dtype=torch.long, device=self.device)
        ])
        self.hist_corrects = torch.cat([
            self.hist_corrects,
            torch.as_tensor([correct], dtype=torch.long, device=self.device)
        ])
        
        batch = {
            "item_ids":self.hist_items.unsqueeze(0), 
            "corrects":self.hist_corrects.unsqueeze(0)
        }

        self.current_state = self.dkt(batch, True)[0,-1,:]
        if self.add_goal:
            self.current_state = torch.concat([self.current_state, self.learner_goal_embedding], dim=-1)

        action = item_id

        self.central_focus = int(step_info['item_id'])
        data = RL_Data(last_state.detach(), action, reward, self.current_state.detach(), done)
        self.memory.set(data)


    def end_episode(self):
        actor_loss, critic_loss = self.learn()
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss
        }


    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.get()

        returns = torch.zeros_like(rewards)
        G = torch.tensor([0.0], device=states.device)
        for i in reversed(range(len(rewards))):
            G = rewards[i]*dones[i] + self.gamma * G * (1 - dones[i])
            returns[i] = G
        rewards = returns

        td_target = rewards + self.gamma * self.value_net(next_states) * (1 - dones)
        states_value = self.value_net(states)
        critic_loss = torch.mean(
            F.mse_loss(states_value, td_target.detach()))
        
        td_delta = td_target - states_value
        log_probs = torch.log(self.policy_net(states).squeeze(1).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        critic_loss.backward() 
        actor_loss.backward()  

        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        
        self.critic_optimizer.step()  
        self.actor_optimizer.step()  

        self.memory.reset()
       
        return actor_loss.item(), critic_loss.item()