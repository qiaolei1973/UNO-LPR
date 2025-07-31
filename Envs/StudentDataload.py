import copy
import matplotlib.pyplot as plt
import pandas as pd
import logging
import random
import os
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Students')

class Student:
    def __init__(self, student_id, initial_items, initial_corrects, learning_targets, session_ids=None):
        self.student_id = student_id
        self.learning_targets = list(set(learning_targets))  
        self.initial_logs = {'item_ids': initial_corrects,
                            'corrects': initial_corrects
                            }

        self.logs = {'item_ids': copy.deepcopy(initial_items).tolist(),
                     'corrects': copy.deepcopy(initial_corrects).tolist()
                    }
        if session_ids is not None:
            self.logs['session_ids'] = copy.deepcopy(session_ids).tolist()

    def add_log(self, new_item_id, new_correct):
        self.logs['item_ids'].append(new_item_id)
        self.logs['corrects'].append(new_correct)

    def get_logs(self):
        batch = {
            'item_ids': torch.Tensor(self.logs['item_ids']).unsqueeze(0).long(),  
            'corrects': torch.Tensor(self.logs['corrects']).unsqueeze(0).long()  
        }
        return batch
    

    @property    
    def profile(self):
        batch = {
            "student_id": self.student_id,
            "item_ids": self.logs['item_ids'],
            "corrects": self.logs['corrects'],
            "targets": self.learning_targets
        }
    
        if  "session_ids" in self.logs.keys():
            batch['session_ids'] = self.logs['session_ids']
        return batch

class StudentGroup:
    def __init__(self, data_path, max_length, num_students=100, learning_pattern='learn', num_goals=5, 
                 user_col='user_id', item_col='item_id', time_col='timestamp', session_col='session_id', action_col='action',
                 seed=None):
        random.seed(seed)
        self.seed = seed
        self.data_path = data_path
        self.max_length = max_length
        self.num_students = num_students
        self.rng = random.Random(seed)
        self.learning_pattern = learning_pattern
        self.num_goals = num_goals
        self.students = []
        self.index = 0
        
        self._create_student_group(user_col, item_col, action_col, session_col, time_col)
    
    def _create_student_group(self, user_col, item_col, action_col, session_col, time_col):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"file not exists: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        user_ids = df[user_col].unique()
        self.global_item_ids = df[item_col].unique()
        self.num_items = len(self.global_item_ids) + 1

        if self.num_students is not None:
            self.num_students = min(self.num_students*2, len(user_ids))
            self.used_student_ids = random.sample(user_ids.tolist(), int(self.num_students))
        else:
            self.used_student_ids = user_ids

        df = df[df[user_col].isin(self.used_student_ids)]

        sorted_df = df.sort_values(by=[session_col, time_col], ascending=[True, True])

        grouped = sorted_df.groupby(user_col).agg({
            item_col: list,
            action_col: list,
            session_col: list
        })

        shuffled_users = grouped.sample(frac=1, random_state=self.seed) 
        self.data = shuffled_users.T.to_dict('list')
        self.eval_used_student_ids = self.used_student_ids[:int(0.2*len(self.used_student_ids))]
        self.used_student_ids = self.used_student_ids[int(0.2*len(self.used_student_ids)):]
        
        
    
    

    
    
    

    
    

    
    

    
    
    

    

    
    
    
    

    

    
    
    
    
    
        
    

    
    def set_used_student_ids(self, student_ids):
        self.used_student_ids = student_ids

    def __iter__(self):
        return self
  

    def __next__(self):

        if self.index >= len(self.used_student_ids): 
            self.rng.shuffle(self.used_student_ids)
            self.index = 0

        student_id = self.used_student_ids[self.index]
        self.index += 1

        data_sequence = self.data[student_id]


        item_seq, action, session_seq = np.array(data_sequence[0]), np.array(data_sequence[1]), np.array(data_sequence[2])

        start_index = -self.max_length - 1

        item_seq = item_seq[start_index:]
        action = action[start_index:]
        session_seq = session_seq[start_index:]

        if self.learning_pattern == 'revise':

            goal_candidtate = np.unique(item_seq).tolist()
            learning_targets = random.sample(goal_candidtate, min(self.num_goals, len(goal_candidtate)))     
            
        elif self.learning_pattern == 'learn':

            goal_candidtate = np.setdiff1d(self.global_item_ids,np.unique(item_seq)).tolist()
            learning_targets = random.sample(goal_candidtate, min(self.num_goals, len(goal_candidtate)))

        elif self.learning_pattern == 'rct':

            l = int(0.8*len(item_seq))
            m = int(0.6*len(item_seq))
            goal_candidtate = item_seq[l:].tolist()
            goal_candidtate = list(goal_candidtate)
            item_seq = item_seq[:m]
            action = action[:m]
            session_seq = session_seq[:m]

            learning_targets = random.sample(goal_candidtate, min(self.num_goals, len(goal_candidtate))) 
        else:
            raise ValueError(f"Unsupported learning pattern: {self.learning_pattern}")
        
        return Student(
            student_id=student_id,
            initial_items=item_seq,
            initial_corrects=action,
            session_ids=session_seq,
            learning_targets=learning_targets
        )
    

