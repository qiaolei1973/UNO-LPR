import torch
import numpy as np
from Envs import JunyiDKTEnv
from Agent import SRC,AC,PPO,GRU4Rec,SASRec,UNO,UniLPR_PPO,CSEAL,HRL
import wandb
import os
from tqdm import tqdm
from Envs import StudentGroup
from utils import  EpisodeLogger
from omegaconf import DictConfig, OmegaConf
import hydra
import sys
sys.path.append('.')
import logging


def run_episode(agent, env, config, episode_num, value_logger, is_training=True):
    observation, learner_profile = env.begin_episode()
    seq_kt_loss = agent.begin_episode(learner_profile)
    if seq_kt_loss is not None:
        value_logger.values.update('seq_kt_loss', seq_kt_loss)

    done = False
    step_count = 0
    learning_path = []
    info = {}

    if config.n_step:
        learning_path = agent.n_step()
        correct_list, reward_list = env.n_step(learning_path)
        steps = learning_path
        agent.observe(correct_list, reward_list)
    else:
        steps = []
        for step in range(config.max_steps):
            learning_item = agent.step(observation)
            if learning_item is tuple:
                learning_item = learning_item[0]
            learning_path.append(learning_item)

            observation, reward, done, step_info = env.step(learning_item)

            step_count += 1
            step_done = done or step_count >= config.max_steps
            agent.observe(reward, step_done, step_info)
            steps.append(learning_item)
            info.update(step_info)
            if done or step_count >= config.max_steps:
                break

    observation, reward, done, info = env.end_episode()
    if is_training:
        loss_dict = agent.end_episode()
        if loss_dict:
            info.update(loss_dict)

    value_logger.log_episode(
        is_training=is_training,
        episode=episode_num,
        reward=reward,
        steps=steps,
        info=info
    )
    return reward, info


def evaluate(agent, env, config, num_episodes=1000):
    env.reset()
    rewards = []
    infos = []
    eval_value_logger = EpisodeLogger(use_wandb=config.use_wandb)

    pbar = tqdm(range(1, num_episodes + 1),
            desc=f"Evaluate {num_episodes} episodes: ",
            bar_format="{l_bar}{bar:20}{r_bar}")

    for episode in pbar:
        reward, info = run_episode(agent, env, config, episode, eval_value_logger, is_training=False)
        rewards.append(reward)
        infos.append(info)
        pbar.set_postfix({'eval/reward': f"{sum(rewards)/len(rewards):.4f}" }, refresh=True)

    avg_reward = np.mean(rewards)
    eval_value_logger.log_evaluation(avg_reward, rewards, infos)

    return avg_reward, rewards, infos



def train_with_config(agent, env, config, config_logger):

    value_logger = EpisodeLogger(use_wandb=config.use_wandb)
    best_reward = float('-inf')
    episode_rewards = []
    all_infos = []

    pbar = tqdm(range(1, config.max_episodes + 1), desc="Training",bar_format="{l_bar}{bar:20}{r_bar}")

    checkpoint_interval = max(1, config.max_episodes // 10)
    checkpoint_rewards = []
    training_final_rewards = []

    for episode_idx, episode in enumerate(pbar, 1):
        episode_reward, info = run_episode(agent, env, config, episode, value_logger, is_training=True)

        episode_rewards.append(episode_reward)
        all_infos.append(info)
        checkpoint_rewards.append(episode_reward)

        if value_logger.values.get("train/reward") > best_reward:
            best_reward = value_logger.values.get("train/reward")

        if config.use_wandb:
            wandb.log({
                    "train/best_reward": best_reward
            })

        if episode_idx % checkpoint_interval == 0:
            avg_reward = sum(checkpoint_rewards) / len(checkpoint_rewards)
            config_logger.info(f"\n[Stage {episode_idx//checkpoint_interval}/10] Average Rewards: {avg_reward:.4f}")
            if config.use_wandb:
                wandb.log({
                    f"train/phase_{episode_idx//checkpoint_interval}_avg_reward": avg_reward
                })
            training_final_rewards = checkpoint_rewards
            checkpoint_rewards = []

        bar_print = {}
        for k,v in value_logger.values.values.items():
            bar_print[k] = f"{v:.4f}"
        for k,v in value_logger.values.current.items():
            bar_print[k] = f"{v:.4f}"

        pbar.set_postfix(bar_print, refresh=True)

    print()
    config_logger.info("------------------- Starting evaluation ---------------------\n")

    evaluate_score, _, _ = evaluate(agent=agent, env=env, config=config, num_episodes=1000)

    results = {
        'evaluate_score': evaluate_score,
        'best_score': best_reward,
        'training_ave_rewards': episode_rewards,
        'training_final_rewards': training_final_rewards,
        'training_all_infos': all_infos
    }
    print()
    config_logger.info(f"Training Reward: {np.mean(results['training_ave_rewards'])}")
    config_logger.info(f"Training Final Reward: {np.mean(results['training_final_rewards'])}")
    config_logger.info(f"Evaluate Reward: {results['evaluate_score']}")
    config_logger.info(f"{'='*60}\n\n")

    return results

def setup_environment(config, seed=1) -> JunyiDKTEnv:
    if 'junyi' in config.data_path:
        model_path = './LifeLongKT/saved_models/DKT/junyi/xxx.pth'
        config_path = './LifeLongKT/configs/junyi/DKT.yaml'
    elif 'assist09' in config.data_path:
        model_path = './LifeLongKT/saved_models/DKT/assist09/xxx.pth'
        config_path = './LifeLongKT/configs/assist09/DKT.yaml'
    
    student_group = StudentGroup(
        data_path=config.data_path,
        max_length=config.max_length,
        learning_pattern=config.learning_pattern,
        num_students=config.max_episodes,
        num_goals=config.num_goals,
        seed=seed
    )
    return JunyiDKTEnv(student_group, model_path, torch.device("cuda" if torch.cuda.is_available() else "cpu"), config_path)

def initialize_agent(device, env, config, q_max_length=220):
    if config.agent == 'AC':
        agent = AC(device=device, state_dim=env.num_items, action_dim=env.num_items,**config['agent_config'])
    elif config.agent == 'SRC':
        agent = SRC(device=device, state_dim=env.num_items, action_dim=env.num_items,max_steps=config.max_steps, **config['agent_config'])
    elif config.agent == 'CSEAL':
        agent = CSEAL(device=device, state_dim=env.num_items, action_dim=env.num_items,**config['agent_config'])
    elif config.agent == 'HRL':
        agent = HRL(device=device, state_dim=env.num_items, action_dim=env.num_items,**config['agent_config'])
    elif config.agent == 'GRU4Rec':
        agent = GRU4Rec(device=device, state_dim=env.num_items, action_dim=env.num_items,**config['agent_config'])
    elif config.agent == 'PPO':
        agent = PPO(device=device, state_dim=env.num_items, action_dim=env.num_items,**config['agent_config'])
    elif config.agent == 'UNO':
        agent = UNO(device=device, state_dim=env.num_items, action_dim=env.num_items, max_len=config.max_steps+q_max_length,**config['agent_config'])
    elif config.agent == 'UniLPR_PPO':
        agent = UniLPR_PPO(device=device, state_dim=env.num_items, action_dim=env.num_items, max_len=config.max_steps+q_max_length,**config['agent_config'])
    elif config.agent == 'SASRec':
        agent = SASRec(device=device, state_dim=env.num_items, action_dim=env.num_items, max_len=config.max_steps+q_max_length,**config['agent_config'])

    return agent

from datetime import datetime
def initialize_wandb(config, run_id: int) -> None:
    if config.use_wandb:
        timestamp = datetime.now().strftime("%m%d-%H%M%S")
        wandb.init(
            project=config.wandb_project,
            tags=config.wandb_tags + [f"lr{config.agent_config.lr:.0e}"],
            config={
                "agent": config.agent,
                "learning_pattern": config.learning_pattern,
                "max_episodes": config.max_episodes,
                "max_steps": config.max_steps,
                "lr": config.agent_config.lr,
                "hidden_dim": config.agent_config.hidden_dim,
                "run_id": run_id
            }
        )

def init_logger(dataset, agent, exp_name):
    current_time = datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.join('outputs', dataset, agent, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{current_time}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger()

@hydra.main(version_base=None, config_path="config/junyi", config_name="Random")
def main(config):
    base_config = config['base_config']
    exp_config = config[base_config.exp_name]
    OmegaConf.set_struct(exp_config, False)
    OmegaConf.set_struct(base_config, False)
    config = OmegaConf.merge(base_config , exp_config)

    dataset = 'assist09' if '09' in config.data_path else 'junyi'
    config_logger = init_logger(dataset, base_config.agent, base_config.exp_name)
    config_logger.info(f'experiment settings: \n {OmegaConf.to_yaml(config)}')

    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)
    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')

    all_training_rewards = []
    training_final_rewards = []
    all_evaluate_scores = []

    for run_id in range(1, config.runs + 1):
        config_logger.info(f"=====================Starting run {run_id}/{config.runs} ======================")
        print()
        initialize_wandb(config=config, run_id=run_id)

        env = setup_environment(config=config, seed=config.runs + 1)
        agent = initialize_agent(device=device, env=env, config=config, q_max_length=env.student_group.max_length)

        results = train_with_config(agent=agent, env=env, config=config, config_logger=config_logger)
        all_training_rewards.append(np.mean(results['training_ave_rewards']))
        training_final_rewards.append(np.mean(results['training_final_rewards']))
        all_evaluate_scores.append(results['evaluate_score'])

        if config.use_wandb:
            wandb.finish()

    config_logger.info(f"===========total: {config.runs}: {config.agent} {config.exp_name}===============")
    config_logger.info(f"Training Rewards - Mean: {np.mean(all_training_rewards):.4f}, Std: {np.std(all_training_rewards):.4f}")
    config_logger.info(f"Training Rewards - Mean: {np.mean(training_final_rewards):.4f}, Std: {np.std(all_training_rewards):.4f}")
    config_logger.info(f"Evaluation Scores - Mean: {np.mean(all_evaluate_scores):.4f}, Std: {np.std(all_evaluate_scores):.4f}")
    config_logger.info("="*60 + "\n")

    if config.use_wandb:
        wandb.init(project=config.wandb_project, reinit=True)
        wandb.log({
            "final/training_reward_mean": np.mean(all_training_rewards),
            "final/training_reward_std": np.std(all_training_rewards),
            "final/evaluation_score_mean": np.mean(all_evaluate_scores),
            "final/evaluation_score_std": np.std(all_evaluate_scores)
        })
        wandb.finish()

if __name__ == "__main__":
    main()