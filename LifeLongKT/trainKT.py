import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import hydra
from datetime import datetime
from DataLoad import SeqDataset, PaddingCollateFn, POMDPSeqDataset
from DKT import DKT, UniLPR
from omegaconf import DictConfig, OmegaConf
from utils import related_loss

def create_dataloaders(train_ratio:float=0.8, valid_ratio:float=0.1, config=None):
    df = pd.read_csv(f'../data/processed_data/{config.dataset_name}/{config.dataset_name}_processed.csv')
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()
    train_users, test_users = train_test_split(user_ids, test_size=1 - train_ratio, random_state=config.seed)
    valid_users, test_users = train_test_split(test_users, test_size=valid_ratio / (1 - train_ratio), random_state=config.seed)

    train = df[df['user_id'].isin(train_users)]
    validation = df[df['user_id'].isin(valid_users)]
    test = df[df['user_id'].isin(test_users)]
    
    if config.model_name in ['DKT']:
        train_dataset = SeqDataset(train, **config['dataset'])
        eval_dataset = SeqDataset(validation, **config['dataset'])
        test_dataset = SeqDataset(test, **config['dataset'])
        collate_fn = PaddingCollateFn(reverse=True)

    elif config.model_name in ['UniLPR']:
        
        train_dataset = POMDPSeqDataset(train, **config['dataset'])
        eval_dataset = POMDPSeqDataset(validation, **config['dataset'])
        test_dataset = POMDPSeqDataset(test, **config['dataset'])
        collate_fn = PaddingCollateFn(reverse=False)



    train_loader = DataLoader(dataset=train_dataset, batch_size=config.dataloader.batch_size,
                             shuffle=True, num_workers=config.dataloader.num_workers,
                             collate_fn=collate_fn)

    eval_loader = DataLoader(dataset=eval_dataset, batch_size=config.dataloader.batch_size,
                            shuffle=False, num_workers=config.dataloader.num_workers,
                            collate_fn=collate_fn)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.dataloader.batch_size,
                            shuffle=False, num_workers=config.dataloader.num_workers,
                            collate_fn=collate_fn)

    return train_loader, eval_loader, test_loader, len(item_ids) + 1

def evaluate(model, eval_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_outputs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in eval_loader:
            labels = batch['labels'].to(device).float()
            outputs = model(batch)
            masked_labels = labels != -100  
            if type(outputs) is tuple:
                outputs, question_output = outputs
                
            outputs = outputs[masked_labels]
            labels = labels[masked_labels]
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            
            
            preds = (outputs > 0.5).float()  
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    
    avg_loss = total_loss / len(eval_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_outputs)
    
    return avg_loss, accuracy, auc


import random 
def train_kt(config):
    
    num_exp = random.randint(0, 1000000)
    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    train_loader, eval_loader, test_loader, num_item = create_dataloaders(config=config)
    
    
    if config.model_name == 'DKT':
        model = DKT(num_item, device, **config['model']).to(device)
    elif config.model_name == 'POMDPDKT':
        model = POMDPDKT(num_item, device, **config['model']).to(device)
    
    
    criterion =  nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    
    
    best_val_loss = float('inf')
    best_val_auc = 0.0
    patience = config.train.patience
    patience_counter = 0
    best_model_state = None
    
    
    for epoch in range(config.train.num_epochs):
        model.train()
        train_loss = 0.0
        
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.train.num_epochs}")
        
        for index, batch in enumerate(pbar):
            
        
            labels = batch['labels'].to(device).type(torch.float32)
            outputs = model(batch)
            masked_labels = labels != -100  
            if type(outputs) is tuple:
                outputs, question_output= outputs
                loss = related_loss(question_output[:,:-1,:], batch['item_ids'][:,1:], config.train.num_negative, num_item, model.item_embedding, device)
                
                
                
                
            else: loss = 0

            loss += criterion(outputs[masked_labels], labels[masked_labels])
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'train_loss': f"{loss.item():.4f}"})
        
        
        avg_train_loss = train_loss / len(train_loader)
        
        
        val_loss, val_acc, val_auc = evaluate(model, eval_loader, device, criterion)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        
        if val_auc > best_val_auc:  
            best_val_auc = val_auc
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            
            timestamp = datetime.now().strftime("%Y%m%d")
            
            if not os.path.exists(config.train.model_save_path):
                os.makedirs(config.train.model_save_path)
            model_save_path = os.path.join(config.train.model_save_path, f"{config.model_name}/{config.dataset_name}")
            
            if not os.path.exists(os.path.dirname(model_save_path)):
                os.makedirs(os.path.dirname(model_save_path))
            model_save_path = os.path.join(model_save_path, f"{num_exp}_seed_{config.seed}_best.pth")
            torch.save(best_model_state, model_save_path)
            print(f"New best model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    
    test_loss, test_acc, test_auc = evaluate(model, test_loader, device, criterion)
    print(f"Final Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")

@hydra.main(version_base=None, config_path="configs/junyi", config_name="DKT")
def main(config):
    print(OmegaConf.to_yaml(config))
    train_kt(config)

if __name__ == "__main__":
    main()
