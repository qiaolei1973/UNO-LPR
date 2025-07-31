import torch
import numpy as np
import os
import sys
import random
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('JunyiDKTEnv')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LifeLongKT import DKT
import yaml


class JunyiDKTEnv:
    def __init__(self, student_group, model_path=None, device='cpu', config_path='configs/DKT.yaml'):
        self.device = device
        self.student_group = student_group
        self.learners = iter(student_group)
        self.num_items = student_group.num_items 
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.model = DKT(self.num_items, device, **config['model']).to(device)

        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                t=t

        self.outside_model = None # for AC,PPO, we can set another pretrained KT model

        self._learner = None
        self.current_state_vector = None


    def _exam(self):
        with torch.no_grad():
            self.current_state = self.model(self._learner.get_logs(), True)[:,-1,:].squeeze() 
            if self.outside_model is not None:
                self.outside_current_state = self.outside_model(self._learner.get_logs(), True)[:,-1,:].squeeze() 
            else:
                self.outside_current_state = self.current_state.clone()

        goals_tensor = torch.tensor(self._learner.learning_targets, dtype=torch.long).to(self.device)  
        goal_scores = torch.gather(self.current_state, dim=0, index=goals_tensor)  
        score = (goal_scores > 0.5).sum().item()
        return score

    def begin_episode(self):
        self.initial_score,self.full_score = 0, 0

        while self.initial_score == self.full_score:
            self._learner = next(self.learners)  
            self.initial_score = self._exam()  
            self.current_score = self.initial_score
            self.full_score = len(self._learner.learning_targets) 

        profile = self._learner.profile    
        profile.update({"targets_embedding":self.model.embedding_layer(torch.tensor(self._learner.learning_targets, dtype=torch.long).unsqueeze(0).to(self.device))})
        
        return self.outside_current_state, profile  
    
    def end_episode(self):
        
        final_score = self._exam()
        reward = self._calculate_reward(final_score)
        
        absolute_reward = self._calculate_absolute_reward(final_score)  
        
        info = {
            "learner_id": self._learner.student_id,
            "goals": self._learner.learning_targets,
            "initial_score": self.initial_score,
            "final_score": final_score,
            "absolute_reward": absolute_reward  
        }
        done = self._check_completion(final_score)
        
        self._learner = None
        return self.outside_current_state, reward, done, info
    

    def n_step(self, item_id_list: int):
        correct_list = []
        reward_list = []
        
        for item_id in item_id_list:
            _, reward, _, info = self.step(item_id)
            correct = info["correct"]
            reward_list.append(reward)
            correct_list.append(correct)
        return correct_list, reward_list
        

    def step(self, item_id: int):
        correct = (self.current_state[item_id] > 0.5).item()
        self._learner.add_log(item_id, correct)

        last_state = self.outside_current_state

        last_score = self.current_score
        self.current_score = self._exam()

        done = self._check_completion(self.current_score)
      
        info = {
            "item_id": item_id,
            "correct": correct,

            "last_state": last_state,
            "current_state": self.outside_current_state,

            "last_score": last_score,
            "current_score": self.current_score
        }
        
        return self.outside_current_state, self._calculate_reward(self.current_score), done, info

    def _check_completion(self, current_score) -> bool:

        return current_score == self.full_score
    
    def _calculate_reward(self, last_score):

        delta = last_score - self.initial_score
        normalize_factor = self.full_score - self.initial_score
        return delta / normalize_factor
    
    def _calculate_absolute_reward(self, last_score):
        delta = last_score - self.initial_score
        return delta
    
    def reset(self):
        self._learner = None
        self.learners = iter(self.student_group)
        self.student_group.index = 0
        self.student_group.used_student_ids = self.student_group.eval_used_student_ids
        