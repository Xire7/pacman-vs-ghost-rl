#!/usr/bin/env python3
"""
Adversarial Training v2 - Improved Self-Play for Pac-Man vs Ghosts

Key improvements based on OpenAI Hide-and-Seek & AlphaStar research:
1. Population-based training - maintain diverse opponent pool
2. Prioritized self-play - sample harder opponents more often  
3. Stronger ghost agents - larger networks, better rewards
4. Much longer training - 1M+ steps for meaningful emergence
5. Interleaved training - train pacman and ghosts more frequently

Usage:
    python train_adversarial_v2.py --pacman models/.../best_model.zip --total-steps 1000000
"""

import argparse
import os
import random
import numpy as np
from datetime import datetime
from collections import deque
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from ghost_agent import IndependentGhostEnv
from gym_env import PacmanEnv


class OpponentPool:
    """
    Population-based opponent pool.
    
    Maintains a history of past model checkpoints to train against.
    This prevents overfitting to just the latest opponent and creates
    a more robust agent (similar to AlphaStar's league training).
    """
    
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.pool = deque(maxlen=max_size)
        self.win_rates = {}  # Track win rate against each opponent
        
    def add(self, model_path, version):
        """Add a new model to the pool."""
        self.pool.append({'path': model_path, 'version': version})
        self.win_rates[model_path] = 0.5  # Initialize at 50%
        
    def sample(self, prioritize_hard=True):
        """
        Sample an opponent from the pool.
        
        If prioritize_hard=True, sample opponents we struggle against more often.
        This creates natural curriculum where we focus on weaknesses.
        """
        if not self.pool:
            return None
            
        if not prioritize_hard or len(self.pool) == 1:
            return random.choice(list(self.pool))
        
        # Prioritize opponents with lower win rate (harder for us)
        weights = []
        for entry in self.pool:
            wr = self.win_rates.get(entry['path'], 0.5)
            # Lower win rate = higher weight
            weight = max(0.1, 1.0 - wr)
            weights.append(weight)
        
        # Normalize
        total = sum(weights)
        probs = [w / total for w in weights]
        
        idx = np.random.choice(len(self.pool), p=probs)
        return self.pool[idx]
    
    def update_win_rate(self, model_path, wins, games):
        """Update win rate for an opponent."""
        if model_path in self.win_rates:
            # Exponential moving average
            old_wr = self.win_rates[model_path]
            new_wr = wins / games
            self.win_rates[model_path] = 0.7 * old_wr + 0.3 * new_wr
    
    def get_latest(self):
        """Get the most recent model."""
        if self.pool:
            return self.pool[-1]
        return None
    
    def __len__(self):
        return len(self.pool)


def create_dirs(base_dir="training_output"):
    """Create output directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"adversarial_v2_{timestamp}")
    dirs = {
        'models': os.path.join(run_dir, 'models'),
        'logs': os.path.join(run_dir, 'logs'),
        'checkpoints': os.path.join(run_dir, 'checkpoints'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return run_dir, dirs


def evaluate_quick(model, layout, ghost_models=None, n=50):
    """Quick evaluation for training loop."""
    wins = 0
    total_score = 0
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
        total_score += info.get('score', 0)
        env.close()
    return wins / n, total_score / n


def evaluate_full(model, layout, ghost_models=None, n=500):
    """Full evaluation with more games."""
    wins = 0
    total_score = 0
    for i in range(n):
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
        total_score += info.get('score', 0)
        env.close()
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n} games... ({wins/(i+1)*100:.1f}%)")
    return wins / n, total_score / n


def train_ghost_improved(ghost_idx, pacman_model, ghost_models, layout, dirs, 
                         timesteps, version, use_larger_network=True):
    """
    Train a ghost with improved DQN settings.
    
    Changes from v1:
    - Larger network (256, 256 instead of default 64, 64)
    - More exploration early on
    - Better reward shaping already in IndependentGhostEnv
    """
    print(f"\n  Training Ghost {ghost_idx} (v{version})...")
    
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
    
    # Network architecture - larger for better learning
    policy_kwargs = dict(
        net_arch=[256, 256]  # Larger than default [64, 64]
    ) if use_larger_network else None
    
    if os.path.exists(prev_path) and version > 1:
        model = DQN.load(prev_path, env=env, tensorboard_log=tb_log_dir)
        # Decay learning rate over versions
        model.learning_rate = max(1e-4, 5e-4 * (0.9 ** (version - 1)))
    else:
        model = DQN(
            "MlpPolicy", env,
            learning_rate=1e-3,
            buffer_size=100000,  # Larger buffer
            learning_starts=2000,
            batch_size=128,  # Larger batch
            gamma=0.99,
            target_update_interval=500,  # More frequent updates
            exploration_fraction=0.4,  # More exploration
            exploration_final_eps=0.02,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=tb_log_dir
        )
    
    model.learn(total_timesteps=timesteps, progress_bar=True)
    save_path = os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{version}.zip")
    model.save(save_path)
    env.close()
    return model, save_path


def train_pacman_with_pool(pacman_path, layout, dirs, ghost_pool, latest_ghost_paths,
                           timesteps, version, n_envs=8):
    """
    Train Pac-Man against a population of ghost opponents.
    
    Key insight: Training against diverse opponents (not just the latest)
    prevents overfitting and creates more robust policies.
    
    Note: We pass ghost PATHS instead of models to avoid pickle issues with SubprocVecEnv.
    For DummyVecEnv with trained ghosts, we load models inside the factory function.
    """
    print(f"\n  Training Pac-Man (v{version})...")
    
    # Mix of training environments:
    # - 30% random ghosts (maintain basic skills, easy baseline)
    # - 30% latest trained ghosts (learn current counter-strategies)  
    # - 40% sampled from pool (diverse, robust learning)
    
    def make_random_env():
        def _init():
            env = PacmanEnv(layout_name=layout, ghost_type='random', max_steps=500)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init
    
    def make_trained_env_from_paths(ghost_paths):
        """Create env factory that loads ghost models from paths (avoids pickle issues)."""
        def _init():
            # Load ghost models inside the factory to avoid pickle issues
            ghost_models = {}
            for gidx, path in ghost_paths.items():
                if path and os.path.exists(path):
                    ghost_models[gidx] = DQN.load(path)
            env = PacmanEnv(layout_name=layout, ghost_policies=ghost_models, max_steps=500)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init
    
    log_dir = os.path.join(dirs['logs'], "pacman")
    
    # Calculate splits
    random_ratio = max(0.2, 0.4 - 0.02 * version)  # Decrease over time
    latest_ratio = 0.4
    pool_ratio = 1.0 - random_ratio - latest_ratio
    
    random_steps = int(timesteps * random_ratio)
    latest_steps = int(timesteps * latest_ratio)
    pool_steps = timesteps - random_steps - latest_steps
    
    # Load base model
    model = MaskablePPO.load(pacman_path, tensorboard_log=log_dir)
    model.learning_rate = max(5e-5, 1e-4 * (0.95 ** (version - 1)))
    model.clip_range = lambda _: max(0.05, 0.15 - 0.01 * version)
    
    # Phase 1: Random ghosts (use DummyVecEnv for consistency)
    if random_steps > 0:
        print(f"    Phase 1: {random_steps:,} steps vs RANDOM ({random_ratio*100:.0f}%)")
        env = VecMonitor(DummyVecEnv([make_random_env() for _ in range(n_envs)]))
        model.set_env(env)
        model.learn(total_timesteps=random_steps, progress_bar=True, reset_num_timesteps=False)
        env.close()
    
    # Phase 2: Latest ghosts (use DummyVecEnv to avoid pickle issues with DQN models)
    if latest_steps > 0 and latest_ghost_paths:
        print(f"    Phase 2: {latest_steps:,} steps vs LATEST ({latest_ratio*100:.0f}%)")
        env = VecMonitor(DummyVecEnv([make_trained_env_from_paths(latest_ghost_paths) for _ in range(n_envs)]))
        model.set_env(env)
        model.learn(total_timesteps=latest_steps, progress_bar=True, reset_num_timesteps=False)
        env.close()
    
    # Phase 3: Pool sampling (if we have a pool)
    if pool_steps > 0 and len(ghost_pool) > 0:
        print(f"    Phase 3: {pool_steps:,} steps vs POOL ({pool_ratio*100:.0f}%)")
        
        # Sample multiple opponents from pool
        n_opponents = min(3, len(ghost_pool))
        steps_per_opponent = pool_steps // n_opponents
        
        for i in range(n_opponents):
            opponent = ghost_pool.sample(prioritize_hard=True)
            if opponent:
                print(f"      Opponent {i+1}: v{opponent['version']}")
                # Get opponent ghost paths
                pool_ghost_paths = {}
                for gidx in latest_ghost_paths.keys():
                    opp_path = opponent['path'].replace('ghost_1', f'ghost_{gidx}')
                    if os.path.exists(opp_path):
                        pool_ghost_paths[gidx] = opp_path
                    else:
                        pool_ghost_paths[gidx] = latest_ghost_paths[gidx]
                
                # Use same n_envs for consistency with other phases
                env = VecMonitor(DummyVecEnv([make_trained_env_from_paths(pool_ghost_paths) for _ in range(n_envs)]))
                model.set_env(env)
                model.learn(total_timesteps=steps_per_opponent, progress_bar=True, reset_num_timesteps=False)
                env.close()
    
    # Save
    save_path = os.path.join(dirs['models'], f"pacman_v{version}")
    model.save(save_path)
    return model, save_path + ".zip"


def main():
    parser = argparse.ArgumentParser(description='Adversarial Training v2')
    parser.add_argument('--pacman', type=str, required=True, help='Pretrained Pac-Man path')
    parser.add_argument('--total-steps', type=int, default=1000000, 
                        help='Total training steps for Pac-Man')
    parser.add_argument('--layout', type=str, default='mediumClassic')
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--checkpoint-freq', type=int, default=100000,
                        help='Steps between checkpoints')
    parser.add_argument('--ghost-train-freq', type=int, default=50000,
                        help='Train ghosts every N pacman steps')
    parser.add_argument('--ghost-steps', type=int, default=30000,
                        help='Steps per ghost training session')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"ADVERSARIAL TRAINING v2 - Population-Based Self-Play")
    print(f"{'='*70}")
    print(f"  Layout:       {args.layout}")
    print(f"  Total Steps:  {args.total_steps:,}")
    print(f"  Checkpoint:   every {args.checkpoint_freq:,} steps")
    print(f"  Ghost Train:  every {args.ghost_train_freq:,} steps ({args.ghost_steps:,} each)")
    print(f"{'='*70}")
    
    run_dir, dirs = create_dirs()
    print(f"Output: {run_dir}\n")
    
    # Load pretrained Pac-Man
    print(f"Loading: {args.pacman}")
    pacman_model = MaskablePPO.load(args.pacman)
    pacman_path = args.pacman
    
    # Baseline
    baseline_wr, baseline_score = evaluate_quick(pacman_model, args.layout, n=100)
    print(f"Baseline vs random: {baseline_wr*100:.1f}% (avg score: {baseline_score:.0f})")
    
    # Get ghost count
    temp_env = PacmanEnv(layout_name=args.layout)
    num_ghosts = temp_env.num_ghosts
    temp_env.close()
    print(f"Ghosts: {num_ghosts}")
    
    # Initialize
    ghost_models = {i: None for i in range(1, num_ghosts + 1)}
    ghost_paths = {i: None for i in range(1, num_ghosts + 1)}  # Track paths for pickle-safe loading
    ghost_pool = OpponentPool(max_size=10)
    
    # Save initial Pac-Man
    pacman_model.save(os.path.join(dirs['models'], "pacman_v0"))
    
    # Training state
    total_pacman_steps = 0
    version = 0
    best_vs_trained = 0.0
    best_model_path = None
    
    # Calculate number of training iterations
    n_iterations = args.total_steps // args.ghost_train_freq
    steps_per_iter = args.ghost_train_freq
    
    print(f"\nTraining plan: {n_iterations} iterations x {steps_per_iter:,} steps")
    
    for iteration in range(1, n_iterations + 1):
        version += 1
        print(f"\n{'─'*70}")
        print(f"ITERATION {iteration}/{n_iterations} (version {version})")
        print(f"  Total Pac-Man steps so far: {total_pacman_steps:,}")
        print(f"{'─'*70}")
        
        # === PHASE 1: Train Ghosts ===
        print(f"\n[Phase 1] Training Ghosts (v{version})")
        for ghost_idx in range(1, num_ghosts + 1):
            ghost_models[ghost_idx], ghost_path = train_ghost_improved(
                ghost_idx, pacman_model, ghost_models,
                args.layout, dirs, args.ghost_steps, version
            )
            ghost_paths[ghost_idx] = ghost_path  # Store path for pickle-safe env creation
        
        # Add to pool
        ghost_pool.add(
            os.path.join(dirs['models'], f"ghost_1_v{version}.zip"),
            version
        )
        
        # === PHASE 2: Train Pac-Man ===
        print(f"\n[Phase 2] Training Pac-Man (v{version})")
        pacman_model, pacman_path = train_pacman_with_pool(
            pacman_path, args.layout, dirs, ghost_pool, ghost_paths,  # Pass paths, not models
            steps_per_iter, version, args.n_envs
        )
        total_pacman_steps += steps_per_iter
        
        # === PHASE 3: Evaluate ===
        print(f"\n[Phase 3] Evaluation")
        wr_random, score_random = evaluate_quick(pacman_model, args.layout, n=50)
        wr_trained, score_trained = evaluate_quick(pacman_model, args.layout, ghost_models, n=50)
        
        print(f"  vs Random:  {wr_random*100:.1f}% (score: {score_random:.0f})")
        print(f"  vs Trained: {wr_trained*100:.1f}% (score: {score_trained:.0f})")
        
        # Update pool with win rate info
        latest = ghost_pool.get_latest()
        if latest:
            ghost_pool.update_win_rate(latest['path'], wr_trained * 50, 50)
        
        # Track best
        if wr_trained > best_vs_trained:
            best_vs_trained = wr_trained
            best_model_path = pacman_path
            pacman_model.save(os.path.join(dirs['models'], "pacman_best"))
            print(f"  ★ New best! Saved pacman_best.zip")
        
        # Checkpoint
        if total_pacman_steps % args.checkpoint_freq == 0:
            ckpt_path = os.path.join(dirs['checkpoints'], f"pacman_step{total_pacman_steps}")
            pacman_model.save(ckpt_path)
            print(f"  📁 Checkpoint saved: {ckpt_path}")
    
    # === FINAL EVALUATION ===
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION (500 games each)")
    print(f"{'='*70}")
    
    # Load best model
    if best_model_path:
        pacman_model = MaskablePPO.load(os.path.join(dirs['models'], "pacman_best"))
    
    print("\nvs Random ghosts:")
    final_random, score_random = evaluate_full(pacman_model, args.layout, n=500)
    
    print("\nvs Trained ghosts:")
    final_trained, score_trained = evaluate_full(pacman_model, args.layout, ghost_models, n=500)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  vs Random:  {final_random*100:.1f}% (avg score: {score_random:.0f})")
    print(f"  vs Trained: {final_trained*100:.1f}% (avg score: {score_trained:.0f})")
    print(f"  Improvement over baseline: {(final_random - baseline_wr)*100:+.1f}%")
    print(f"{'='*70}")
    
    print(f"\nModels saved in: {dirs['models']}")
    print(f"TensorBoard: tensorboard --logdir={dirs['logs']}")


if __name__ == '__main__':
    main()
