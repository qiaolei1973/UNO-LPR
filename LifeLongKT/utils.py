import torch
import torch.nn as nn

def related_loss(question_emb, item_ids, num_negatives, item_num, item_embedding, device):
    item_ids = item_ids.to(device)
    label_mask = item_ids != -100  
    item_embedding = item_embedding

    def sample_negatives(positive_ids, num_negatives, item_num, label_mask=None):
        if label_mask is not None:
            positive_ids = positive_ids[label_mask]
        
        
        if positive_ids.dim() == 1:
            output_shape = (positive_ids.size(0), num_negatives)
        else:
            output_shape = positive_ids.size() + (num_negatives,)
        
        
        sampled_ids = torch.randint(
            low=1,  
            high=item_num,
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device
        )
        
        
        expanded_positive = positive_ids.unsqueeze(-1)  
        
        
        negative_mask = (sampled_ids != expanded_positive)
        
        
        if positive_ids.dim() == 2:
            
            seq_comparison = (sampled_ids.unsqueeze(2) != positive_ids.unsqueeze(-1))  
            negative_mask = seq_comparison.all(dim=-1)  
        
        return sampled_ids, negative_mask

    sampled_ids, negative_mask = sample_negatives(item_ids, num_negatives, item_num, label_mask)
    positive_ids = item_ids[label_mask]  
    question_emb = question_emb[label_mask]  

    positive_embedding = item_embedding(positive_ids)
    negative_embedding = item_embedding(sampled_ids)

    logits = torch.matmul(positive_embedding.unsqueeze(1), question_emb.unsqueeze(-1)).squeeze(-1)  
    targets = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long, device=logits.device)
    
    negative_logits = torch.matmul(negative_embedding, question_emb.unsqueeze(-1)).squeeze(-1)  
    negative_logits = negative_logits.masked_fill(negative_mask == 0, float('-inf'))
    
    if num_negatives == 1:
        negative_logits = negative_logits.unsqueeze(-1)
        logits = logits.unsqueeze(-1)

    logits = torch.cat((logits, negative_logits), dim=-1)  

    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))

    loss = loss.mean()

    return loss