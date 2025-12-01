"""Shared utilities for training scripts."""

import os
import time
import numpy as np
from datetime import datetime
from typing import Callable, Dict, Optional, List

from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from gym_env import PacmanEnv, make_masked_pacman_env


# =============================================================================
# Directory Management
# =============================================================================

def create_training_dirs(base_dir: str, prefix: str, layout: str = None):
    """Create timestamped output directories for a training run.
    
    Args:
        base_dir: Base directory (e.g., 'training_output', 'logs', 'models')
        prefix: Prefix for run folder (e.g., 'ppo', 'mixed')
        layout: Optional layout name to include in folder name
    
    Returns:
        Tuple of (run_dir, dirs_dict with 'models' and 'logs' keys)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if layout:
        run_name = f"{prefix}_{layout}_{timestamp}"
    else:
        run_name = f"{prefix}_{timestamp}"
    run_dir = os.path.join(base_dir, run_name)
    
    dirs = {
        'root': run_dir,
        'models': os.path.join(run_dir, 'models') if 'mixed' in prefix else run_dir,
        'logs': os.path.join(run_dir, 'logs') if 'mixed' in prefix else run_dir,
    }
    
    # For PPO, model_dir and log_dir are separate top-level dirs
    if 'ppo' in prefix:
        dirs['models'] = run_dir
        dirs['logs'] = run_dir
    
    for key in ['models', 'logs']:
        os.makedirs(dirs[key], exist_ok=True)
    
    return run_dir, dirs


def create_mixed_dirs(base_dir: str = "training_output"):
    """Create output directories for mixed adversarial training.
    
    Returns:
        Tuple of (run_dir, dirs_dict)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"mixed_{timestamp}")
    dirs = {
        'models': os.path.join(run_dir, 'models'),
        'logs': os.path.join(run_dir, 'logs'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return run_dir, dirs


# =============================================================================
# Model Loading
# =============================================================================

def load_ghost_models(paths: Dict[int, str]) -> Optional[Dict[int, DQN]]:
    """Load ghost models from paths.
    
    Args:
        paths: Dict mapping ghost index to model path, e.g., {1: 'path1.zip', 2: 'path2.zip'}
    
    Returns:
        Dict {idx: model} or None if no valid paths
    """
    if not paths:
        return None
    
    models = {}
    for idx, path in paths.items():
        if path and os.path.exists(path):
            models[idx] = DQN.load(path)
    
    return models if models else None


# =============================================================================
# Environment Creation
# =============================================================================

def create_vec_env(layout_name: str, ghost_type: str, num_envs: int, 
                   max_steps: int = 500, normalize: bool = False, 
                   norm_stats_path: str = None, training: bool = True):
    """Create vectorized training environment.
    
    Args:
        layout_name: Layout to use
        ghost_type: Type of ghost agent ('random' or 'directional')
        num_envs: Number of parallel environments
        max_steps: Max steps per episode
        normalize: Whether to use VecNormalize
        norm_stats_path: Path to load normalization stats from (for eval/resume)
        training: Whether this is for training (updates running stats) or eval (frozen stats)
    
    Returns:
        Vectorized environment
    """
    def make_env():
        return make_masked_pacman_env(layout_name, ghost_type, max_steps=max_steps)
    
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    env = VecMonitor(env)
    
    if normalize:
        if norm_stats_path and os.path.exists(norm_stats_path):
            env = VecNormalize.load(norm_stats_path, env)
            env.training = training
            env.norm_reward = training
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, 
                              clip_obs=10.0, clip_reward=10.0)
    
    return env


# =============================================================================
# Learning Rate Schedules
# =============================================================================

def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    """Linear learning rate schedule from initial to final value.
    
    Args:
        initial_value: Starting learning rate
        final_value: Ending learning rate
    
    Returns:
        Schedule function that takes progress_remaining and returns LR
    """
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule


# =============================================================================
# Callbacks
# =============================================================================

class MetricsCallback(BaseCallback):
    """Logs win/loss metrics during training."""
    
    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.wins = 0
        self.losses = 0
        self.episodes = 0
        self.log_freq = log_freq
        self.recent_wins = []
        
    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                self.episodes += 1
                info = self.locals.get('infos', [{}])[i]
                
                if info.get('win', False):
                    self.wins += 1
                    self.recent_wins.append(1)
                else:
                    self.losses += 1
                    self.recent_wins.append(0)
                
                if len(self.recent_wins) > 100:
                    self.recent_wins.pop(0)
                
                if self.episodes % self.log_freq == 0:
                    win_rate = self.wins / max(1, self.wins + self.losses)
                    recent_rate = sum(self.recent_wins) / max(1, len(self.recent_wins))
                    print(f"\nEp {self.episodes} | Win Rate: {win_rate:.1%} | Recent: {recent_rate:.1%}")
        return True


class NormalizeSyncCallback(BaseCallback):
    """Syncs VecNormalize stats between training and eval envs periodically."""
    
    def __init__(self, eval_env, sync_freq: int = 10000):
        super().__init__()
        self.eval_env = eval_env
        self.sync_freq = sync_freq
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.sync_freq == 0:
            if hasattr(self.training_env, 'obs_rms') and hasattr(self.eval_env, 'obs_rms'):
                self.eval_env.obs_rms = self.training_env.obs_rms
                self.eval_env.ret_rms = self.training_env.ret_rms
        return True


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_pacman(model, layout: str, ghost_models: Dict = None, 
                    ghost_type: str = 'random', n_episodes: int = 50, 
                    render: bool = False, verbose: bool = False) -> Dict:
    """Evaluate Pac-Man model and return statistics.
    
    Args:
        model: Trained MaskablePPO model
        layout: Layout name
        ghost_models: Dict of ghost DQN models {1: model, 2: model} or None for random
        ghost_type: Ghost type if ghost_models is None ('random' or 'directional')
        n_episodes: Number of evaluation episodes
        render: Whether to render games visually
        verbose: Print per-episode results
    
    Returns:
        Dict with 'win_rate', 'wins', 'losses', 'mean_score', 'std_score', 'mean_reward'
    """
    wins = 0
    scores = []
    rewards = []
    render_mode = 'human' if render else None
    
    for ep in range(n_episodes):
        if ghost_models:
            env = PacmanEnv(layout_name=layout, ghost_policies=ghost_models, 
                           max_steps=500, render_mode=render_mode)
        else:
            env = PacmanEnv(layout_name=layout, ghost_type=ghost_type, 
                           max_steps=500, render_mode=render_mode)
        
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            if isinstance(action, np.ndarray):
                action = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
            if render:
                time.sleep(0.05)
        
        win = info.get('win', False)
        score = info.get('score', 0)
        
        if win:
            wins += 1
        scores.append(score)
        rewards.append(ep_reward)
        
        if verbose:
            result = 'WIN' if win else 'LOSE'
            print(f"Ep {ep+1}: {result} | Reward: {ep_reward:.1f}")
        
        env.close()
        
        if render:
            time.sleep(0.5)
    
    return {
        'win_rate': wins / n_episodes,
        'wins': wins,
        'losses': n_episodes - wins,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
    }


def quick_evaluate(model, layout: str, ghost_models: Dict = None, n: int = 50) -> float:
    """Quick evaluation returning just win rate (for mixed training).
    
    Args:
        model: Trained MaskablePPO model
        layout: Layout name
        ghost_models: Dict of ghost DQN models or None for random
        n: Number of episodes
    
    Returns:
        Win rate as float (0.0 to 1.0)
    """
    wins = 0
    for _ in range(n):
        if ghost_models:
            env = PacmanEnv(layout_name=layout, ghost_policies=ghost_models, max_steps=500)
        else:
            env = PacmanEnv(layout_name=layout, ghost_type='random', max_steps=500)
        
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            if isinstance(action, np.ndarray):
                action = int(action.item())
            obs, _, t, tr, info = env.step(action)
            done = t or tr
        
        if info.get('win'):
            wins += 1
        env.close()
    
    return wins / n


def print_eval_summary(results: Dict, n_episodes: int):
    """Print evaluation summary."""
    print(f"\nWin Rate: {results['wins']}/{n_episodes} ({results['win_rate']*100:.1f}%)")
    print(f"Mean Reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
