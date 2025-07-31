import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class SeqDataset(Dataset):

    def __init__(self, df, max_length,
                 user_col='user_id', item_col='item_id', time_col='timestamp', session_col='session_id', action_col='action'):

        self.max_length = max_length
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.session_col = session_col

        self.data = df.sort_values(by=[session_col, time_col], ascending=[True, True]).groupby(user_col).agg({
            item_col: list,
            session_col: list,
            action_col: list,
            time_col: list
            }).reset_index().set_index(user_col).T.to_dict('list')

        self.user_ids = list(self.data.keys())

    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, idx):
        data_sequence = self.data[self.user_ids[idx]]
        item_seq, session_seq, action, timestamp_seq = np.array(data_sequence[0]), np.array(data_sequence[1]), np.array(data_sequence[2]), np.array(data_sequence[3])

        timestamp_seq = timestamp_seq[:self.max_length+1]
        session_seq = session_seq[:self.max_length+1]
        item_seq = item_seq[:self.max_length+1]
        action = action[:self.max_length+1]

        session_seq = session_seq.max() - session_seq
        session_seq = session_seq.max() - session_seq

        input_seq = item_seq[:-1]
        action_seq = action[:-1]
        session_seq = session_seq[:-1]
        timestamp_seq = timestamp_seq[:-1]

        target_seq = item_seq[1:]
        labels= action[1:]
        
        return {'item_ids': input_seq, 'corrects': action_seq,
                'session_seq': session_seq, 'timestamp_sequence': timestamp_seq,
                'targets': target_seq, 'labels': labels}

class POMDPSeqDataset(SeqDataset):
    def __init__(self, df, max_length,
                 user_col='user_id', item_col='item_id', time_col='timestamp', session_col='session_id', action_col='action'):
        super().__init__(df, max_length, user_col, item_col, time_col, session_col, action_col)

    def __getitem__(self, idx):
        data_sequence = self.data[self.user_ids[idx]]
        item_seq, session_seq, action, timestamp_seq = np.array(data_sequence[0]), np.array(data_sequence[1]), np.array(data_sequence[2]), np.array(data_sequence[3])

        start_index = -self.max_length - 1

        timestamp_seq = timestamp_seq[start_index:]
        session_seq = session_seq[start_index:]
        item_seq = item_seq[start_index:]
        action = action[start_index:]

        input_seq = item_seq
        session_seq = session_seq.max() - session_seq
        session_seq = session_seq.max() - session_seq
        
        action_seq = action
        timestamp_seq = timestamp_seq[:-1]

        labels= action[1:]

        return {'item_ids': input_seq, 'corrects': action_seq,
                'session_seq': session_seq, 'timestamp_sequence': timestamp_seq,
                'labels': labels}

from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

class PaddingCollateFn:
    def __init__(self, padding_value=0, labels_padding_value=-100, session_padding_value=0, reverse=True):
        self.session_padding_value = session_padding_value
        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value
        self.reverse = reverse  
    
    def __call__(self, batch):
        collated_batch = {}

        for key in batch[0].keys():
            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.tensor([example[key] for example in batch])
                continue

            
            if key == 'labels' or key == 'valid_labels':
                padding_value = self.labels_padding_value
            elif 'session_ids' in key:
                padding_value = self.session_padding_value
            else:
                padding_value = self.padding_value
            
            
            values = [torch.tensor(example[key]) for example in batch]
            
            if self.reverse:
                
                reversed_values = [torch.flip(v, dims=(0,)) for v in values]
                padded_reversed = pad_sequence(reversed_values, batch_first=True, padding_value=padding_value)
                collated_batch[key] = torch.flip(padded_reversed, dims=(1,))  
            else:
                
                collated_batch[key] = pad_sequence(values, batch_first=True, padding_value=padding_value)

        
        if 'item_ids' in collated_batch:
            attention_mask = collated_batch['item_ids'] != self.padding_value
            collated_batch['attention_mask'] = attention_mask.to(dtype=torch.float32)
        if 'action_sequence' in collated_batch:
            collated_batch['action_sequence'] = collated_batch['action_sequence'].to(dtype=torch.float32)

        return collated_batch

