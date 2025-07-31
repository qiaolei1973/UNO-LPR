import torch
import torch.nn as nn
from LifeLongKT import featureEmbedding
from .agent_utils import RL_Data, Memory
from .AC import AC
from .PPO import PPO
import networkx as nx
import numpy as np
from tqdm import tqdm
import os

class HRL(nn.Module):
    def __init__(self, device, state_dim, action_dim, hidden_dim=300, 
                 gamma=0.98, lr=1e-4, clip_epsilon=0.2, 
                 ppo_epochs=4, mini_batch_size=64, 
                 max_sub_step=5,
                 candidates_method='embedding',
                 emb_cons_num=10,
                 graph_path='./data/processed_data/junyi/prerequest_graph.gexf',
                 graph_embedding_path="./data/processed_data/junyi/GraphEmbeddings.npy"):
        super().__init__()

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        if not os.path.exists(graph_embedding_path):
            raise FileNotFoundError(f"Model file not found: {graph_embedding_path}")

        self.device = device
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.max_sub_step = max_sub_step
        self.memory = Memory(device)
        self.embedding_layer = featureEmbedding(num_embeddings=action_dim, type='One-hot')
        
        self.kt = nn.GRU(state_dim*2, state_dim)
        self.output_layer = nn.Sequential(nn.Linear(state_dim, state_dim),
                                          nn.ReLU(),
                                          nn.Linear(state_dim, 1))
                                        
        graph_embedding = np.load(graph_embedding_path, allow_pickle=True)
        self.graph_embedding = torch.from_numpy(graph_embedding).to(device)
        self.candidates_method = candidates_method
        if candidates_method == 'embedding':
            self.emb_cons_num = emb_cons_num
            self._init_embedding_candidates(graph_path)
            
        else:
            self.knowledge_graph = nx.read_gexf(graph_path)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

        self.high_agent = PPO(device=device, state_dim=state_dim*2, action_dim=action_dim, gamma=gamma, lr=lr, clip_epsilon=clip_epsilon,ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size, add_goal=False)
        self.low_agent = AC(device=device, state_dim=state_dim*3, action_dim=action_dim, hidden_dim=hidden_dim, lr=lr, add_goal=False)
        
        self.to(device)


    def embed_interaction(self, item_ids, corrects, train=False):

        if len(item_ids.shape) < 2:
            item_ids = item_ids.unsqueeze(0)
            corrects = corrects.unsqueeze(0)
        
        item_emb = self.embedding_layer(item_ids, corrects)
        output, hidden = self.kt(item_emb)
        h_t = hidden[:,-1]

        if train:
            logits = self.output_layer(output)  # shape: (batch_size, seq_len, 1)
            pred_probs = torch.sigmoid(logits).squeeze(-1)  # shape: (batch_size, seq_len)

            pred_probs = pred_probs[:, :-1]
            target_corrects = corrects[:, 1:]

            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(pred_probs, target_corrects.float())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return h_t, loss
                
        return h_t

    def begin_episode(self, user_profile):
        self.leaner_goal = torch.sum(user_profile['targets_embedding'],dim=-2)
        item_ids = torch.as_tensor(user_profile["item_ids"], dtype=torch.long, device=self.device)
        corrects = torch.as_tensor(user_profile["corrects"], dtype=torch.long, device=self.device)
        
        self.hist_items = item_ids.clone()
        self.hist_corrects = corrects.clone()
        
        self.h_t, loss = self.embed_interaction(item_ids, corrects,True) 
        self.high_current_states = torch.concat([self.h_t, self.leaner_goal],dim=-1)
        
        self.g = self.high_agent.step(self.high_current_states)
        self.g_emb = self.embedding_layer(torch.Tensor([[self.g]]).to(self.device).to(torch.long)).squeeze(0)
        
        self.low_current_states = torch.concat([self.h_t, self.leaner_goal, self.g_emb],dim=-1)

        self.high_agent.begin_episode(user_profile)
        self.low_agent.begin_episode(user_profile)

        self.num_sub_step = 0
        return loss

        
    def observe(self, reward, done, step_info):
        item_id = step_info.get('item_id')
        correct = step_info.get('correct')

        self.hist_items = torch.cat([
            self.hist_items,
            torch.as_tensor([item_id], dtype=torch.long, device=self.device)
        ])
        self.hist_corrects = torch.cat([
            self.hist_corrects,
            torch.as_tensor([correct], dtype=torch.long, device=self.device)
        ])

        self.low_last_states = self.low_current_states
        self.h_t = self.embed_interaction(self.hist_items, self.hist_corrects)

        if self.num_sub_step % self.max_sub_step == 0 or done:
            self.high_last_states = self.high_current_states
            self.high_current_states = torch.concat([self.h_t, self.leaner_goal],dim=-1)

            high_reward = 1 if done else 0
            high_step_info = {
                'last_state': self.high_last_states.clone().detach().squeeze(),
                'current_state': self.high_current_states.clone().detach().squeeze(),
                'item_id': self.g
            }
            self.high_agent.observe(high_reward, done, high_step_info) 
            self.g = self.high_agent.step(self.high_current_states)
            self.g_emb = self.embedding_layer(torch.Tensor([[self.g]]).to(self.device).to(torch.long)).squeeze(0)
        
        self.low_current_states = torch.concat([self.h_t, self.leaner_goal, self.g_emb],dim=-1)

        if self.h_t[0,self.g] > 0.9 or  \
            done or self.num_sub_step % self.max_sub_step == 0:
            low_reward,low_done = 1, True
        else:
            low_reward,low_done = 0, False
        
        low_reward = low_reward + reward if done else low_reward

        low_step_info = {
            'last_state': self.low_last_states.clone().detach().squeeze(),
            'current_state': self.low_current_states.clone().detach().squeeze(),
            'item_id': item_id
        }

        self.low_agent.observe(low_reward, low_done, low_step_info)

            
    def step(self, observation=None):

        if self.num_sub_step % self.max_sub_step == (self.max_sub_step - 1):
            action = self.g
        else:
            low_candidates = self.get_low_candidates(self.g)
            
            self.low_current_states = torch.concat([self.h_t, self.leaner_goal, self.g_emb],dim=-1)
            action = self.low_agent.step(self.low_current_states, candidates=low_candidates)
        
        self.num_sub_step += 1
        return action
    
    def learn(self, states=None, actions=None, rewards=None, next_states=None, dones=None):
        low_losses = self.low_agent.learn()
        high_losses = self.high_agent.learn()
        return low_losses, high_losses

    def end_episode(self):
        low_losses, high_losses = self.learn()
        return  {
            'actor_loss': low_losses[0],
            'critic_loss': low_losses[1]
        }
    
    def get_low_candidates(self, sub_goal):
        if self.candidates_method == 'embedding':
            low_candidates = self._get_embedding_candidates(sub_goal)
        else:
            low_candidates = self.get_prerequisite_candidates(sub_goal)
            
        if low_candidates is not None:
            return low_candidates
        
        return None
    
    def _get_embedding_candidates(self, node):
        assert node > 0
        node = node - 1
        return self.embedding_candidates[node].tolist()

    def get_prerequisite_candidates(self, target_node, max_depth=None, verbose=False):
        target_node = str(target_node)
        if not hasattr(self, 'knowledge_graph'):
            if verbose:
                print(f"[Error] Knowledge graph not initialized")
            return None
        
        if target_node not in self.knowledge_graph:
            if verbose:
                print(f"[Warning] Target node {target_node} not in graph")
            return None

        if verbose:
            print(f"\nProcessing target node {target_node}:")
            print(f" - Total nodes in graph: {len(self.knowledge_graph.nodes)}")
            print(f" - Max depth: {'None' if max_depth is None else max_depth}")

        try:
            predecessors = list(nx.dfs_preorder_nodes(
                self.knowledge_graph.reverse(copy=False),
                source=target_node
            ))
            
            prerequisites = [n for n in predecessors if n != target_node]
            
            if max_depth is not None:
                depth_dict = nx.shortest_path_length(
                    self.knowledge_graph.reverse(),
                    source=target_node
                )
                prerequisites = [n for n in prerequisites if depth_dict[n] <= max_depth]
                if verbose:
                    print(f" - Prerequisites after depth filtering: {len(prerequisites)}")
            
            topo_order = list(nx.topological_sort(self.knowledge_graph))
            sorted_prereqs = [n for n in topo_order if n in prerequisites]
            
            result = sorted_prereqs if prerequisites else [target_node]
            
            if verbose:
                print(f" - Found {len(result)} prerequisites:")
                print(f"   First 5: {result[:5]}{'...' if len(result)>5 else ''}")
                if max_depth:
                    print(f"   Max depth: {max(depth_dict[n] for n in result) if result else 0}")
            
            return list(set([int(r) for r in result]))
            
        except Exception as e:
            if verbose:
                print(f"[Error] Processing failed: {str(e)}")
            return None

    def _init_embedding_candidates(self, graph_path):
        cache_dir = os.path.dirname(graph_path)
        cache_path = os.path.join(cache_dir, 'emb_candidates.npy')
        if os.path.exists(cache_path):
            self.embedding_candidates = np.load(cache_path, allow_pickle=True)
        else:
            self._calculate_embedding_distances(graph_path)
            np.save(cache_path, self.embedding_candidates)

    def _calculate_embedding_distances(self, graph_path):
        self.prerequisite_graph = nx.read_gexf(graph_path)
        num_nodes = len(self.graph_embedding)
        dist_matrix = torch.zeros((num_nodes+1, num_nodes+1), device=self.device)
        
        for i in tqdm(range(num_nodes)):
            for j in range(i, num_nodes):
                dist = torch.norm(self.graph_embedding[i] - self.graph_embedding[j], p=2)
                dist_matrix[i+1,j+1] = dist_matrix[j,i] = dist
        
        self.embedding_candidates = []
        for i in range(num_nodes):
            sorted_indices = torch.argsort(dist_matrix[i])
            candidates = sorted_indices[:self.emb_cons_num].cpu().numpy()
            self.embedding_candidates.append(candidates)

    