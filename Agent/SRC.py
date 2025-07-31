import torch
import torch.nn as nn
import torch.nn.functional as F

class ConceptAwareEncoder(nn.Module):
    def __init__(self, action_dim, embedding_dim=300, hidden_dim=300):
        super().__init__()
        self.embedding = nn.Embedding(action_dim, embedding_dim)
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.W_q = nn.Linear(embedding_dim, hidden_dim)
        self.W_k = nn.Linear(embedding_dim, hidden_dim)
        self.W_v = nn.Linear(embedding_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, concept_ids=None):
        if concept_ids is None:
            all_concept_ids = torch.arange(self.action_dim, 
                                        device=self.embedding.weight.device)  # [num_concepts]
            x = self.embedding(all_concept_ids)  # [num_concepts, embedding_dim]
            x = x.unsqueeze(0)  
            
        else:
            x = self.embedding(concept_ids)  # [batch_size, action_dim, embedding_dim]
        
        Q = self.W_q(x)  # [batch_size, action_dim, hidden_dim]
        K = self.W_k(x)
        V = self.W_v(x)
        attn_weights = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5), dim=-1)
        E_s_attn = torch.bmm(attn_weights, V)  # [batch_size, action_dim, hidden_dim]
        
        E_mlp = self.mlp(x)  # [batch_size, action_dim, hidden_dim]
        pooled = E_mlp.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]
        E_s_mlp = E_mlp + pooled
        
        E_s = torch.cat([E_s_attn, E_s_mlp], dim=-1)  # [batch_size, action_dim, hidden_dim*2]
        E_s = self.layer_norm(E_s)
        
        return E_s.squeeze(0)
    
class ConceptAwareDecoder(nn.Module):
    def __init__(self, max_steps, num_concepts, embedding_dim=300, hidden_dim=300, device='cpu'):
        super().__init__()
        self.embedding = nn.Embedding(num_concepts, embedding_dim)
        self.steps = max_steps
        self.device = device
        self.input_fc = nn.ModuleDict({
            'lstm_input': nn.Linear(embedding_dim+ 1, embedding_dim, bias=False),
            'encoder_input':  nn.Linear(embedding_dim*2, embedding_dim, bias=False),
            'd_i': nn.Linear(embedding_dim, 1),
            'kt': nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())
            })

        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_dim)
        
        self.target_fusion = nn.Linear(embedding_dim, hidden_dim)
        
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.kt_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, encoder_output, learner_profile):
        hist_emb = self.embedding(torch.as_tensor(learner_profile["item_ids"], device=self.device).unsqueeze(0))
        hist_corrects = torch.as_tensor(learner_profile["corrects"], device=self.device).unsqueeze(0).unsqueeze(-1).float()
        target_ids = torch.as_tensor(learner_profile['targets'],dtype=torch.long, device=self.device)
        
        lstm_input = self.input_fc['lstm_input'](torch.cat([hist_emb, hist_corrects], dim=-1))

        h_n = torch.zeros(hist_emb.shape[0], self.hidden_dim).to(lstm_input.device)
        c_n = torch.zeros(hist_emb.shape[0], self.hidden_dim).to(lstm_input.device)

        for t in range(lstm_input.size(1)):
            h_n, c_n = self.lstm_cell(lstm_input[:, t, :], (h_n, c_n))
        student_state = h_n
  
        target_emb = self.embedding(target_ids).mean(dim=0) 
        target_rep = self.target_fusion(target_emb)  # [batch_size, hidden_dim]
        
        path_probs = []
        kt_probs = []
        selected_indices = []
        encoder_embedding = self.input_fc['encoder_input'](encoder_output)
        

        for i in range(self.steps):
            d_i = self.input_fc['d_i'](student_state+encoder_embedding+target_rep)
            d_i[0, 0] = 0
            for i, idx in enumerate(selected_indices):
                d_i[i, 0] = 0
            mask = torch.ones_like(d_i)  
            mask[list(selected_indices)] = 0  
            masked_logits = d_i * mask - 1e10 * (1 - mask)  
            step_probs = F.softmax(masked_logits, dim=0).squeeze() 

            selected_idx = step_probs.argmax(dim=-1)  
            path_probs.append(step_probs[selected_idx].unsqueeze(0))

            e_sj = self.input_fc['encoder_input'](encoder_output[selected_idx]).unsqueeze(0)
            selected_indices.append(selected_idx.item())

            kt_v = self.input_fc['kt'](student_state)
            kt_probs.append(kt_v)

            student_state = self.lstm_cell(e_sj,(student_state, student_state))[0]

        
        return selected_indices, torch.cat(kt_probs), torch.cat(path_probs)
    

class SRC(nn.Module):
    def __init__(self, device, state_dim, action_dim, hidden_dim=300,
                  gamma=0.98, lr=1e-5, beta=0.1, max_steps=10,
                  temperature=1, add_goal=False):
        super(SRC, self).__init__()
        self.gamma = gamma
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.encoder = ConceptAwareEncoder(action_dim, embedding_dim=hidden_dim, hidden_dim=hidden_dim)
        self.decoder = ConceptAwareDecoder(max_steps, action_dim, embedding_dim=hidden_dim, hidden_dim=hidden_dim, device=device)
        self.beta = beta
        self.device = device
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)  

    def n_step(self):
        encoder_output = self.encoder()
        action_list, kt_probs, path_probs = self.decoder(encoder_output=encoder_output, learner_profile=self.learner_profile)
        self.kt_probs = kt_probs
        self.path_probs = path_probs
        return action_list

    
    def begin_episode(self, learner_profile):
        self.learner_profile = learner_profile
    

    def end_episode(self):
        actor_loss, critic_loss = self.learn()
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss
        }

    def observe(self, correct_list, reward_list):
        ET = reward_list[-1]
        log_probs = torch.log(self.path_probs)
        self.L_theta = -ET * torch.sum(log_probs)

        y_true = torch.Tensor(correct_list).to(self.device).float()
        self.Ly = -torch.mean(
            y_true * torch.log(self.kt_probs + 1e-8) + 
            (1 - y_true) * torch.log(1 - self.kt_probs + 1e-8)
        )


    def learn(self):
        loss = self.L_theta +  self.beta *  self.Ly
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(),0




    
