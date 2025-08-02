import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class featureEmbedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int=None, type:str='One-hot'):
        super(featureEmbedding, self).__init__()
        self.type = type
        self.num_embeddings = num_embeddings
        if type != 'One-hot':
            assert embedding_dim is not None
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.action_embedding = nn.Linear(1, embedding_dim)
        else:
            num_embeddings *= 2
            embedding_dim = num_embeddings
            self.embedding_matrix = nn.Parameter(torch.eye(num_embeddings, embedding_dim), requires_grad=False)

    def forward(self, item_id: torch.Tensor, correct: torch.Tensor=None):

        if self.type != 'One-hot':
            item_embedding = self.embedding(item_id)
            correct_embedding = self.action_embedding(correct.unsqueeze(1))
            return item_embedding + correct_embedding
        else:
            
            if correct is None:
                return self.embedding_matrix[item_id][:, :, :self.num_embeddings]
            item_id = item_id * (1 + correct)
            return self.embedding_matrix[item_id.type(torch.int64)]
        
    
class DKT(nn.Module):
    def __init__(self, num_item, device, hidden_size, emb_dim, dropout, num_layers):
        super().__init__()
        input_size = num_item * 2
        num_skills = num_item
        dropout = dropout
        self.device = device

        self.embedding_layer = featureEmbedding(num_embeddings=num_item, type='One-hot')
        self.fw = nn.Linear(input_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, num_skills)

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch, states=False):
        item_seq = batch['item_ids'].to(self.device) 
        action_seq = batch['corrects'].to(self.device)

        item_emb = self.embedding_layer(item_seq, action_seq)
        item_emb = self.fw(item_emb).permute(1, 0, 2)

        output, _ = self.rnn(item_emb)
        out = self.fc_out(output).permute(1, 0, 2)  
        
        
        if states:
            return out
        
        targets_seq = batch['targets'].to(self.device)
        targets_emb = self.embedding_layer(targets_seq)
        
        result = torch.sum(out * targets_emb, dim=-1) 
        result = torch.sigmoid(result)

        return result

class UniLPR(nn.Module):
    def __init__(self, num_item, device, hidden_size, emb_dim, max_len, dropout=0.1, num_layers=2,sigmoid_func=False):
        super().__init__()
        self.num_item = num_item
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.max_len = max_len

        self.device = device
        self.sigmoid_func = sigmoid_func
        
        
        self.item_pos_embedding = nn.Embedding(max_len+1, emb_dim)
        self.action_pos_embedding = nn.Embedding(max_len+1, emb_dim)
        self.session_pos_embedding = nn.Embedding(max_len+1, emb_dim)

        self.item_embedding = nn.Embedding(num_item, emb_dim)
        self.action_embedding = nn.Embedding(3, emb_dim)
        
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=4,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.transformer_layer,
            num_layers=num_layers
        )
        

        self.fc_out = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2),
                                    nn.ReLU(),
                                    nn.Linear(emb_dim // 2, 1))
        self.dropout = nn.Dropout(dropout)
        
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        self.initializer_range=0.02

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
                
    def create_padding_mask(self, seq):
        return (seq != 0).to(self.device)  

    def create_attention_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(self.device)

    def forward(self, batch):
        item_seq = batch['item_ids'].to(self.device)  
        action_seq = batch['corrects'].to(self.device)
        if 'session_seq' in batch:
            session_seq = batch['session_seq'].to(self.device) 
            session_pos_emb = self.session_pos_embedding(session_seq).repeat_interleave(2, dim=1)
        
        item_emb = self.item_embedding(item_seq) 
        action_emb = self.action_embedding(action_seq) 
    

        # (8) Xu' = Xu || [xT, x[REC]]
        if 'goal' in batch:
            action_seq = torch.concat([action_seq, 2+torch.zeros_like(action_seq)[:, 1].unsqueeze(1)],dim=1)
            action_emb = self.action_embedding(action_seq) 
            goal_emb = torch.mean(self.item_embedding(batch['goal']),dim=0).unsqueeze(0).unsqueeze(1).repeat(item_emb.shape[0],1,1)
            item_emb = torch.concat([item_emb, goal_emb], dim=1)
            
            item_seq = torch.concat([item_seq, torch.ones_like(item_seq)[:, 1].unsqueeze(1).to(torch.long)], dim=1)
            
       
        batch_size, seq_len, _ = item_emb.size()
        
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        item_pos_emb = self.item_pos_embedding(positions)
        action_pos_emb = self.action_pos_embedding(positions)

        # (9) Xp
        pos_emb = torch.cat([item_pos_emb, action_pos_emb], dim=2).view(batch_size, seq_len*2, -1)   
        
        if 'session_seq' in batch: 
            pos_emb = pos_emb + session_pos_emb

        total_emb = torch.cat([item_emb, action_emb], dim=2)  
        total_emb = total_emb.view(batch_size, seq_len*2, -1) 
        total_emb *= self.emb_dim ** 0.5 
        
        # (10) X = Xu' + Xp
        total_emb = total_emb + pos_emb  
        x = self.dropout(total_emb)
        
        # M: causal mask 
        src_key_padding_mask = self.create_padding_mask(item_seq).repeat_interleave(2, dim=1)    
        mask = self.create_attention_mask(seq_len*2)  
        
        x = x * src_key_padding_mask.unsqueeze(-1)  

        # (11)-(13)
        output = self.transformer(
            src=x,
            mask=mask
        )  

        question_output = output[:,1::2,:]
        
        question_result = question_output[:,:-1,:]

        output = output[:,::2,:][:,1:,:]

        logits = self.fc_out(question_result+output)
        if self.sigmoid_func:
            result = torch.sigmoid(logits.squeeze(-1))
        else:
            result = logits.squeeze(-1)
        return (result, question_output)


class SessionEmbedding(nn.Module):

    def __init__(self, pool_strategy='mean', hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_strategy = pool_strategy
        self.hidden_dim = hidden_dim

    def forward(self, seq_emb, sessions):

        batch_size = seq_emb.size(0)
        max_session_id = sessions.max().item()
        session_masks = (
                sessions.unsqueeze(1) == torch.arange(1, max_session_id + 1).view(1, -1, 1).to(seq_emb.device)).float()

        session_lengths = session_masks.sum(dim=2, keepdim=True)
        session_lengths = torch.where(session_lengths == 0, torch.ones_like(session_lengths), session_lengths)

        seq_emb_expanded = seq_emb.unsqueeze(1) * session_masks.unsqueeze(-1)

        session_embeddings = seq_emb_expanded.sum(dim=2) / session_lengths

        valid_mask = session_lengths.squeeze(-1) > 0
        session_embeddings = session_embeddings[valid_mask].reshape(batch_size, -1, self.hidden_dim)
        return session_embeddings
    

class SASRec_KT(nn.Module):
    def __init__(self, num_item, device, hidden_size, emb_dim, max_len, dropout=0.1, num_layers=2):
        super().__init__()
        self.num_item = num_item
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.device = device
        
        
        self.item_embedding = nn.Embedding(num_item, emb_dim)
        self.pos_embedding = nn.Embedding(max_len, emb_dim)
        
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=4,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        initializer_range = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)

    def create_padding_mask(self, seq):
        return (seq != 0).to(self.device)

    def create_attention_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, device=self.device), diagonal=1).bool()

    def forward(self, batch):
        
        item_seq = batch['item_ids'].to(self.device)
        batch_size, seq_len = item_seq.size()
        
        
        pos_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        
        
        item_emb = self.item_embedding(item_seq)  
        pos_emb = self.pos_embedding(pos_ids)     
        
        
        seq_emb = item_emb + pos_emb
        seq_emb = self.dropout(seq_emb)
        
        
        mask = self.create_attention_mask(seq_len)
        
        
        out = seq_emb
        for layer in self.transformer_layers:
            out = layer(out, src_mask=mask)
        
        
        question_output = out
   
        return question_output
    

class GRU4Rec_KT(nn.Module):
    def __init__(self, num_item, device, hidden_size, emb_dim,  dropout=0.1, num_layers=2):
        super().__init__()
        self.num_item = num_item
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.device = device
        
        
        self.item_embedding = nn.Embedding(num_item, emb_dim)
        
        
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
        
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        initializer_range = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)

    def forward(self, batch):
        item_seq = batch['item_ids'].to(self.device)
        
        item_emb = self.item_embedding(item_seq)  
        item_emb = self.dropout(item_emb)
        
        
        gru_out, _ = self.gru(item_emb)  
        
        question_output = gru_out 
        return question_output
