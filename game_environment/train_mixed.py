#!/usr/bin/env python3
"""Mixed Adversarial Training for Pac-Man vs Ghosts.

Alternates training between random and trained ghosts to prevent
catastrophic forgetting while improving against smart opponents.

Features:
- Episode-based training (not timesteps)
- Progress bars with episode counts
- Early stopping when performance plateaus
- JSON metrics export
"""

import argparse
import json
import os
import time
from collections import deque
from datetime import datetime
from typing import Dict

import numpy as np
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from ghost_agent import IndependentGhostEnv
from gym_env import PacmanEnv
from training_utils import (
    create_training_dirs,
    evaluate_pacman,
    run_evaluation,
)


# =============================================================================
# Callbacks
# =============================================================================

class EpisodeProgressCallback(BaseCallback):
    """Track episodes with progress bar and optional early stopping."""
    
    def __init__(
        self, 
        n_episodes: int,
        desc: str = "Training",
        early_stop_patience: int = 0,  # 0 = disabled
        early_stop_delta: float = 0.02,  # 2% improvement threshold
        eval_window: int = 50,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.n_episodes = n_episodes
        self.desc = desc
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        self.eval_window = eval_window
        
        self.episode_count = 0
        self.pbar = None
        
        # For early stopping
        self.recent_wins = deque(maxlen=eval_window)
        self.best_win_rate = 0.0
        self.patience_counter = 0
        self.stopped_early = False
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.n_episodes, desc=self.desc, unit="ep")
    
    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_count += 1
                self.pbar.update(1)
                
                # Track wins for early stopping
                win = 1 if info.get('win', False) else 0
                self.recent_wins.append(win)
                
                # Check early stopping
                if self.early_stop_patience > 0 and len(self.recent_wins) >= self.eval_window:
                    if self.episode_count % self.eval_window == 0:
                        current_wr = np.mean(self.recent_wins)
                        
                        if current_wr > self.best_win_rate + self.early_stop_delta:
                            self.best_win_rate = current_wr
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1
                            
                        self.pbar.set_postfix({
                            'wr': f'{current_wr*100:.1f}%',
                            'best': f'{self.best_win_rate*100:.1f}%',
                            'pat': f'{self.patience_counter}/{self.early_stop_patience}'
                        })
                        
                        if self.patience_counter >= self.early_stop_patience:
                            self.stopped_early = True
                            self.pbar.set_description(f"{self.desc} (early stop)")
                            return False
                
                if self.episode_count >= self.n_episodes:
                    return False
        return True
    
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()


# =============================================================================
# Training Functions
# =============================================================================

def train_ghost(ghost_idx: int, pacman_model, ghost_models: Dict, 
                layout: str, dirs: Dict, episodes: int, version: int,
                early_stop: bool = False) -> DQN:
    """Train a single ghost using DQN."""
    
    num_ghosts = len(ghost_models)
    other_ghosts = {i: ghost_models[i] for i in range(1, num_ghosts + 1) 
                    if i != ghost_idx and ghost_models[i] is not None}
    
    env = IndependentGhostEnv(
        ghost_index=ghost_idx,
        layout_name=layout,
        pacman_policy=pacman_model,
        other_ghost_policies=other_ghosts,
        max_steps=500
    )
    
    prev_path = os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{version-1}.zip")
    tb_log_dir = os.path.join(dirs['logs'], f"ghost_{ghost_idx}")
    
    if os.path.exists(prev_path) and version > 1:
        model = DQN.load(prev_path, env=env, tensorboard_log=tb_log_dir, device='cpu')
        model.learning_rate = 5e-4
    else:
        model = DQN(
            "MlpPolicy", env,
            learning_rate=1e-3,
            buffer_size=100000,
            learning_starts=2000,
            batch_size=128,
            gamma=0.99,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=0,
            tensorboard_log=tb_log_dir,
            device='cpu'  # CPU is faster for small networks
        )
    
    callback = EpisodeProgressCallback(
        n_episodes=episodes,
        desc=f"Ghost {ghost_idx}",
        early_stop_patience=5 if early_stop else 0,
        eval_window=30
    )
    
    model.learn(total_timesteps=int(1e9), callback=callback, progress_bar=False)
    model.save(os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{version}"))
    env.close()
    
    return model


def train_ghosts(pacman_model, ghost_models: Dict, layout: str, 
                 dirs: Dict, episodes: int, version: int,
                 early_stop: bool = False) -> Dict:
    """Train all ghosts."""
    num_ghosts = len(ghost_models)
    
    print(f"\n  Training {num_ghosts} ghosts (v{version}, {episodes} episodes each)")
    
    new_ghost_models = {}
    for ghost_idx in range(1, num_ghosts + 1):
        new_ghost_models[ghost_idx] = train_ghost(
            ghost_idx, pacman_model, ghost_models,
            layout, dirs, episodes, version, early_stop
        )
    
    return new_ghost_models


def train_pacman(pacman_path: str, layout: str, dirs: Dict, 
                 ghost_models: Dict, episodes: int, version: int,
                 n_envs: int = 8, early_stop: bool = False):
    """Train Pac-Man using alternating random/trained ghost phases."""
    
    def make_random_env():
        def _init():
            env = PacmanEnv(layout_name=layout, ghost_type='random', max_steps=500)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init
    
    def make_trained_env():
        def _init():
            env = PacmanEnv(layout_name=layout, ghost_policies=ghost_models, max_steps=500)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init

    # Curriculum: 60% -> 40% random ghosts over rounds
    random_ratio = max(0.4, 0.6 - 0.05 * (version - 1))
    random_episodes = int(episodes * random_ratio)
    trained_episodes = episodes - random_episodes
    log_dir = os.path.join(dirs['logs'], "pacman")
    
    # Phase 1: Train on random ghosts
    print(f"\n  Training Pac-Man (v{version})")
    env_rand = VecMonitor(DummyVecEnv([make_random_env() for _ in range(n_envs)]))
    model = MaskablePPO.load(pacman_path, env=env_rand, tensorboard_log=log_dir)
    model.learning_rate = 1e-4
    model.clip_range = lambda _: 0.1
    
    callback1 = EpisodeProgressCallback(
        n_episodes=random_episodes,
        desc=f"vs Random ({random_ratio*100:.0f}%)",
        early_stop_patience=5 if early_stop else 0
    )
    model.learn(total_timesteps=int(1e9), callback=callback1, progress_bar=False)
    env_rand.close()
    
    # Phase 2: Train on trained ghosts
    env_trained = VecMonitor(DummyVecEnv([make_trained_env() for _ in range(n_envs)]))
    model2 = MaskablePPO.load(pacman_path, env=env_trained, tensorboard_log=log_dir)
    model2.policy.load_state_dict(model.policy.state_dict())
    model2.learning_rate = 1e-4
    model2.clip_range = lambda _: 0.1
    
    callback2 = EpisodeProgressCallback(
        n_episodes=trained_episodes,
        desc=f"vs Trained ({(1-random_ratio)*100:.0f}%)",
        early_stop_patience=5 if early_stop else 0
    )
    model2.learn(total_timesteps=int(1e9), callback=callback2, progress_bar=False, reset_num_timesteps=False)
    env_trained.close()
    
    # Save
    save_path = os.path.join(dirs['models'], f"pacman_v{version}")
    model2.save(save_path)
    return model2, save_path + ".zip"


# =============================================================================
# Metrics Tracking
# =============================================================================

class MetricsTracker:
    """Track and save training metrics."""
    
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.metrics = {
            'rounds': [],
            'timestamps': [],
            'pacman_vs_random': [],
            'pacman_vs_trained': [],
            'combined_score': [],
            'best_version': 0,
            'best_score': 0.0
        }
    
    def add_round(self, round_num: int, wr_random: float, wr_trained: float):
        combined = 0.7 * wr_trained + 0.3 * wr_random
        
        self.metrics['rounds'].append(round_num)
        self.metrics['timestamps'].append(datetime.now().isoformat())
        self.metrics['pacman_vs_random'].append(wr_random)
        self.metrics['pacman_vs_trained'].append(wr_trained)
        self.metrics['combined_score'].append(combined)
        
        if combined > self.metrics['best_score']:
            self.metrics['best_score'] = combined
            self.metrics['best_version'] = round_num
            return True
        return False
    
    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def print_summary(self):
        print("\n" + "─" * 40)
        print("TRAINING PROGRESS")
        print("─" * 40)
        for i, (r, wr_rand, wr_train) in enumerate(zip(
            self.metrics['rounds'],
            self.metrics['pacman_vs_random'],
            self.metrics['pacman_vs_trained']
        )):
            marker = "★" if r == self.metrics['best_version'] else " "
            print(f"{marker} Round {r}: {wr_rand*100:.1f}% (rand) | {wr_train*100:.1f}% (trained)")


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args):
    """Run mixed adversarial training."""
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"MIXED ADVERSARIAL TRAINING")
    print(f"{'='*60}")
    print(f"  Layout: {args.layout}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Ghost episodes/round: {args.ghost_episodes}")
    print(f"  Pac-Man episodes/round: {args.pacman_episodes}")
    print(f"  Early stopping: {'ON' if args.early_stop else 'OFF'}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
        print(f"  Starting round: {args.start_round}")
    print(f"{'='*60}")
    
    # Setup directories
    if args.resume:
        run_dir = args.resume
        dirs = {
            'models': os.path.join(run_dir, 'models'),
            'logs': os.path.join(run_dir, 'logs')
        }
        print(f"\nResuming in: {run_dir}")
    else:
        run_dir, dirs = create_training_dirs("training_output", "mixed")
        print(f"\nOutput: {run_dir}")
    
    # Initialize metrics tracker
    metrics = MetricsTracker(os.path.join(run_dir, 'metrics.json'))
    
    # Load Pac-Man model
    if args.resume and args.start_round > 1:
        prev_pacman_version = args.start_round - 1
        pacman_path = os.path.join(dirs['models'], f"pacman_v{prev_pacman_version}.zip")
        print(f"Loading: {pacman_path}")
        pacman_model = MaskablePPO.load(pacman_path)
    else:
        print(f"Loading: {args.model_path}")
        pacman_model = MaskablePPO.load(args.model_path)
        pacman_path = args.model_path
    
    # Baseline (only for new runs)
    if not args.resume:
        baseline = evaluate_pacman(pacman_model, args.layout, n_episodes=50)['win_rate']
        print(f"Baseline vs random: {baseline*100:.1f}%")
        pacman_model.save(os.path.join(dirs['models'], "pacman_v0"))
    
    # Get ghost count
    temp_env = PacmanEnv(layout_name=args.layout)
    num_ghosts = temp_env.num_ghosts
    temp_env.close()
    print(f"Ghosts: {num_ghosts}")
    
    ghost_models = {i: None for i in range(1, num_ghosts + 1)}
    
    # Load existing ghost models if resuming
    if args.resume and args.start_round > 1:
        prev_ghost_version = args.start_round
        for ghost_idx in range(1, num_ghosts + 1):
            ghost_path = os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{prev_ghost_version}.zip")
            if os.path.exists(ghost_path):
                ghost_models[ghost_idx] = DQN.load(ghost_path)
                print(f"  Loaded: ghost_{ghost_idx}_v{prev_ghost_version}")
    
    # Track best model
    best_wr = 0.0
    best_version = 0
    
    # Training loop
    for round_num in range(args.start_round, args.rounds + 1):
        round_start = time.time()
        
        print(f"\n{'═'*60}")
        print(f"  ROUND {round_num}/{args.rounds}")
        print(f"{'═'*60}")
        
        # Train ghosts
        ghost_models = train_ghosts(
            pacman_model, ghost_models, args.layout, dirs,
            args.ghost_episodes, round_num, args.early_stop
        )
        
        # Train Pac-Man
        pacman_model, pacman_path = train_pacman(
            pacman_path, args.layout, dirs, ghost_models,
            args.pacman_episodes, round_num, early_stop=args.early_stop
        )
        
        # Evaluate
        print("\n  Evaluating...")
        wr_random = evaluate_pacman(pacman_model, args.layout, n_episodes=50)['win_rate']
        wr_trained = evaluate_pacman(pacman_model, args.layout, ghost_models=ghost_models, n_episodes=30)['win_rate']
        
        round_time = time.time() - round_start
        print(f"\n  Results: {wr_random*100:.1f}% vs random | {wr_trained*100:.1f}% vs trained")
        print(f"  Round time: {round_time/60:.1f} min")
        
        # Track metrics
        is_best = metrics.add_round(round_num, wr_random, wr_trained)
        metrics.save()
        
        if is_best:
            best_wr = 0.7 * wr_trained + 0.3 * wr_random
            best_version = round_num
            pacman_model.save(os.path.join(dirs['models'], "pacman_best"))
            print(f"  ★ New best! Saved pacman_best.zip (v{round_num})")
    
    # Final results
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    
    metrics.print_summary()
    
    final_random = evaluate_pacman(pacman_model, args.layout, n_episodes=100)['win_rate']
    final_trained = evaluate_pacman(pacman_model, args.layout, ghost_models=ghost_models, n_episodes=50)['win_rate']
    
    print(f"\nFinal Performance:")
    print(f"  vs Random:  {final_random*100:.1f}%")
    print(f"  vs Trained: {final_trained*100:.1f}%")
    print(f"  Best model: pacman_v{best_version} (saved as pacman_best.zip)")
    print(f"\nTotal time: {total_time/60:.1f} min")
    
    print(f"\n{'='*60}")
    print(f"TENSORBOARD")
    print(f"{'='*60}")
    print(f"  tensorboard --logdir={dirs['logs']}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Mixed Adversarial Training / Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--layout', default='mediumClassic', help='Game layout')
    parser.add_argument('--model-path', type=str, help='Pac-Man model path')
    
    # Training args
    parser.add_argument('--rounds', type=int, default=10, help='Training rounds')
    parser.add_argument('--ghost-episodes', type=int, default=300, 
                        help='Ghost training episodes per round')
    parser.add_argument('--pacman-episodes', type=int, default=600, 
                        help='Pac-Man training episodes per round')
    parser.add_argument('--early-stop', action='store_true',
                        help='Enable early stopping when performance plateaus')
    parser.add_argument('--resume', type=str, help='Resume from run directory')
    parser.add_argument('--start-round', type=int, default=1, 
                        help='Starting round (for resume)')
    
    # Eval args
    parser.add_argument('--ghost1', type=str, help='Ghost 1 model path')
    parser.add_argument('--ghost2', type=str, help='Ghost 2 model path')
    parser.add_argument('--episodes', type=int, default=100, help='Eval episodes')
    parser.add_argument('--render', action='store_true', help='Render during eval')
    
    args = parser.parse_args()
    
    if not args.model_path:
        parser.error("--model-path is required")
    
    if args.eval:
        run_evaluation(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
