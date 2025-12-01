"""Shared utilities for training scripts."""

import os
import time
import numpy as np
from datetime import datetime
from typing import Callable

from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from gym_env import PacmanEnv, make_masked_pacman_env


# =============================================================================
# Directory Management
# =============================================================================

def create_training_dirs(base_dir, prefix):
    """Create timestamped output directories for a training run.
    
    Args:
        base_dir: Base directory (e.g., 'training_output', 'logs')
        prefix: Prefix for run folder (e.g., 'ppo', 'mixed')
    
    Returns:
        Tuple of (run_dir, dirs_dict)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
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

def load_ghost_models(ghost1_path, ghost2_path):
    """Load ghost models from paths.
    
    Args:
        ghost1_path: Path to ghost 1 model (or None)
        ghost2_path: Path to ghost 2 model (or None)
    
    Returns:
        Dict {1: model, 2: model} or None if paths not provided
    """
    if ghost1_path and ghost2_path:
        return {
            1: DQN.load(ghost1_path),
            2: DQN.load(ghost2_path)
        }
    return None


# =============================================================================
# Environment Creation
# =============================================================================

def create_vec_env(layout_name, ghost_type, num_envs, max_steps=500, 
                   normalize=False, norm_stats_path=None, training=True):
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
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
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
    
    def __init__(self, log_freq=100):
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
    
    def __init__(self, eval_env, sync_freq=10000):
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

def evaluate_pacman(model, layout, ghost_models=None, ghost_type='random', 
                    n_episodes=50, render=False, verbose=False):
    """Evaluate Pac-Man model and return statistics.
    
    Args:
        model: Trained MaskablePPO model
        layout: Layout name
        ghost_models: Dict of ghost DQN models {0: model, 1: model} or None for random
        ghost_type: Ghost type if ghost_models is None ('random' or 'directional')
        n_episodes: Number of evaluation episodes
        render: Whether to render games visually
        verbose: Print per-episode results
    
    Returns:
        Dict with 'win_rate', 'wins', 'losses', 'mean_score', 'std_score', 'scores'
    """
    wins = 0
    scores = []
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
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            if isinstance(action, np.ndarray):
                action = int(action.item())
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if render:
                time.sleep(0.05)
        
        win = info.get('win', False)
        score = info.get('score', 0)
        
        if win:
            wins += 1
        scores.append(score)
        
        if verbose:
            result = 'WIN!' if win else 'LOSE'
            print(f"Ep {ep+1}: {result} | Score: {score} | Steps: {steps}")
        
        env.close()
        
        if render:
            time.sleep(0.5)
    
    return {
        'win_rate': wins / n_episodes,
        'wins': wins,
        'losses': n_episodes - wins,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores
    }


def print_eval_summary(results, n_episodes):
    """Print evaluation summary."""
    print(f"\nWin Rate: {results['wins']}/{n_episodes} ({results['win_rate']*100:.1f}%)")
    print(f"Mean Score: {results['mean_score']:.1f} ± {results['std_score']:.1f}")


def run_evaluation(args):
    """Common evaluation function for both training scripts.
    
    Args:
        args: Parsed arguments with model_path, layout, episodes, render, ghost1, ghost2
    """
    print(f"\nEvaluating on {args.layout}")
    print(f"  Model: {args.model_path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Render: {args.render}")
    
    model = MaskablePPO.load(args.model_path)
    ghost_models = load_ghost_models(
        getattr(args, 'ghost1', None), 
        getattr(args, 'ghost2', None)
    )
    
    if ghost_models:
        print(f"  Ghost 1: {args.ghost1}")
        print(f"  Ghost 2: {args.ghost2}")
    else:
        ghost_type = getattr(args, 'ghost_type', 'random')
        print(f"  Ghosts: {ghost_type}")
    print()
    
    results = evaluate_pacman(
        model, args.layout,
        ghost_models=ghost_models,
        ghost_type=getattr(args, 'ghost_type', 'random'),
        n_episodes=args.episodes,
        render=args.render,
        verbose=True
    )
    print_eval_summary(results, args.episodes)
