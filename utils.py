import numpy as np
from collections import defaultdict
import time
import wandb


class EMAValue:
    def __init__(self):
        self.values = {} 
        self.counts = {}
        self.current = {}
        self.alpha = 0.9
    
    def update(self, metric, value):
        if metric not in self.values:
            self.values[metric] = 0.0
            self.counts[metric] = 0
        
        self.counts[metric] += 1

        if self.counts[metric] == 1:
            self.values[metric] = value
        else:
            self.values[metric] = (value + self.values[metric] * (self.counts[metric]-1)) / (self.counts[metric])
    
    def add(self, metric, value):
        self.current['epoch_'+str(metric)] = value

    def get(self, metric):
        if metric not in self.values:
            raise ValueError(f"Metric '{metric}' not in available metrics")
        return self.values.get(metric, None)


class EpisodeLogger:
    def __init__(self, use_wandb=True):
        self.log_data = defaultdict(list)
        self.start_time = time.time()
        self.use_wandb = use_wandb
        self.values = EMAValue()

    def log_episode(self,is_training, episode, reward, steps, info=None):
        self.log_data['episode'].append(episode)
        self.log_data['reward'].append(reward)
        self.log_data['num_steps'].append(len(steps))
        self.log_data['steps'].append(steps)
        elapsed = time.time() - self.start_time
        self.log_data['time'].append(elapsed)
        self.values.update("train/reward", reward)
        self.values.add("train/reward", reward)

        self.values.update("train/len(steps)", len(steps))
        self.values.add("train/len(steps)", len(steps))

        if info:
            for key, value in info.items():
                self.log_data[key].append(value)
                if 'loss' in key:
                    self.values.add(f"{key}", value)
                    self.values.update(f"{key}", value)
        
        if self.use_wandb:
            if is_training:
                prefix = "train"
            else:
                prefix = "eval"
            wandb.log({
                f"{prefix}/episode_reward": reward,
            })
            for key, value in self.values.values.items():
                wandb.log({f"{prefix}/avg_{key}": value})

    
    def log_evaluation(self, avg_reward, rewards, infos):
        self.values.update("eval/reward", avg_reward)
        if self.use_wandb:
            wandb.log({
                "eval/avg_reward": avg_reward,
                "eval/std_reward": np.std(rewards)
            })
            
            wandb.log({"eval/reward_dist": wandb.Histogram(rewards)})

    def get_logs(self):
        return dict(self.log_data)