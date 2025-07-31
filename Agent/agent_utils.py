import torch
import torch.nn as nn
import torch.optim as optim
import collections
import torch.nn.functional as F

class RL_Data:
    def __init__(self, state, action, reward, next_state, done,log_prob=None, value=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.log_prob = log_prob
        self.value = value

class Memory:
    def __init__(self, device):
        self.buffer = collections.deque()
        self.device = device

    def set(self, data):
        self.buffer.append(data)

    def get(self):
        if isinstance(self.buffer[0].state, dict):
            states = {
                "item_ids": torch.concat([data.state["item_ids"] for data in self.buffer], dim=0),
                "corrects": torch.concat([data.state["corrects"] for data in self.buffer], dim=0)
            }
            next_states = {
                "item_ids": torch.concat([data.next_state["item_ids"] for data in self.buffer]),
                "corrects": torch.concat([data.next_state["corrects"] for data in self.buffer])
            }
        else:
            states = torch.stack([data.state for data in self.buffer]).to(self.device)
            next_states = torch.stack([data.next_state for data in self.buffer]).to(self.device)

        actions = torch.tensor([data.action for data in self.buffer], 
                            dtype=torch.long, device=self.device).unsqueeze(1)
        
        rewards = torch.tensor([data.reward for data in self.buffer], 
                            dtype=torch.float32, device=self.device).unsqueeze(1)
        
        dones = torch.tensor([data.done for data in self.buffer], 
                            dtype=torch.float32, device=self.device).unsqueeze(1)
        
        if self.buffer[0].log_prob is not None:
            log_probs = torch.stack([data.log_prob for data in self.buffer]).to(self.device)
            return states, actions, rewards, next_states, dones, log_probs
        else:
            return states, actions, rewards, next_states, dones
    
    def reset(self):
        self.buffer.clear()

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, temperature=1.0):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.temperature = 1

    def forward(self, x):
        if torch.isnan(self.fc1.weight).any():
            print("Warning: fc1.weight contains NaN values!")
            t=t
        x = F.relu(self.fc1(x))
        fc2_output = self.fc2(x) / self.temperature
        softmax_output = F.softmax(fc2_output, dim=-1)
        return softmax_output
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)