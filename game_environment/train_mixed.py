#!/usr/bin/env python3
"""
Mixed Adversarial Training for Pac-Man vs Ghosts.

Alternates training between random and trained ghosts to prevent
catastrophic forgetting while improving against smart opponents.
"""

import argparse
import glob
import os
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from ghost_agent import IndependentGhostEnv
from gym_env import PacmanEnv
from training_utils import create_mixed_dirs, quick_evaluate, evaluate_pacman, print_eval_summary


def train_ghost(ghost_idx, pacman_model, ghost_models, layout, dirs, timesteps, version):
    """Train a ghost using DQN.
    
    Args:
        ghost_idx: Index of ghost to train (1-based)
        pacman_model: Current Pac-Man model to train against
        ghost_models: Dict of all ghost models
        layout: Layout name
        dirs: Directory paths dict
        timesteps: Training timesteps
        version: Version number for this training round
    
    Returns:
        Trained DQN model
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
    
    if os.path.exists(prev_path) and version > 1:
        model = DQN.load(prev_path, env=env, tensorboard_log=tb_log_dir)
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
            tensorboard_log=tb_log_dir
        )
    
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{version}"))
    env.close()
    return model


def train_pacman(pacman_path, layout, dirs, ghost_models, timesteps, version, n_envs=8):
    """Train Pac-Man using alternating random/trained ghost phases.
    
    Args:
        pacman_path: Path to current Pac-Man model
        layout: Layout name
        dirs: Directory paths dict
        ghost_models: Dict of trained ghost models
        timesteps: Total training timesteps
        version: Version number for this training round
        n_envs: Number of parallel environments
    
    Returns:
        Tuple of (trained model, path to saved model)
    """
    print(f"\n  Training Pac-Man (v{version})...")
    
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

    # Curriculum: 60% -> 40% random ghosts (decreases by 5% each version)
    random_ratio = max(0.4, 0.6 - 0.05 * (version - 1))
    random_steps = int(timesteps * random_ratio)
    trained_steps = timesteps - random_steps
    log_dir = os.path.join(dirs['logs'], "pacman")
    
    # Phase 1: Train on random ghosts
    print(f"    Phase 1: {random_steps:,} steps vs random ghosts ({random_ratio*100:.0f}%)")
    env_rand = VecMonitor(DummyVecEnv([make_random_env() for _ in range(n_envs)]))
    model = MaskablePPO.load(pacman_path, env=env_rand, tensorboard_log=log_dir)
    model.learning_rate = 1e-4
    model.clip_range = lambda _: 0.1
    model.learn(total_timesteps=random_steps, progress_bar=True)
    env_rand.close()
    
    # Phase 2: Train on trained ghosts
    print(f"    Phase 2: {trained_steps:,} steps vs trained ghosts ({(1-random_ratio)*100:.0f}%)")
    env_trained = VecMonitor(DummyVecEnv([make_trained_env() for _ in range(n_envs)]))
    model2 = MaskablePPO.load(pacman_path, env=env_trained, tensorboard_log=log_dir)
    model2.policy.load_state_dict(model.policy.state_dict())
    model2.learning_rate = 1e-4
    model2.clip_range = lambda _: 0.1
    model2.learn(total_timesteps=trained_steps, progress_bar=True, reset_num_timesteps=False)
    env_trained.close()
    
    # Save
    save_path = os.path.join(dirs['models'], f"pacman_v{version}")
    model2.save(save_path)
    return model2, save_path + ".zip"


def evaluate_mixed(args):
    """Evaluate Pac-Man model against trained ghosts."""
    print(f"\n{'='*60}")
    print(f"EVALUATION MODE")
    print(f"{'='*60}")
    print(f"  Pac-Man: {args.pacman}")
    print(f"  Layout: {args.layout}")
    print(f"  Episodes: {args.episodes}")
    if args.run_dir:
        print(f"  Run dir: {args.run_dir}")
    print(f"  Render: {args.render}")
    print(f"{'='*60}\n")
    
    # Load Pac-Man model
    pacman_model = MaskablePPO.load(args.pacman)
    
    # Load ghost models if run_dir provided
    ghost_models = None
    if args.run_dir:
        models_dir = os.path.join(args.run_dir, 'models')
        if os.path.isdir(models_dir):
            # Find latest ghost versions
            ghost_models = {}
            for ghost_file in glob.glob(os.path.join(models_dir, "ghost_*_v*.zip")):
                basename = os.path.basename(ghost_file)
                # Parse ghost_1_v5.zip -> idx=1, version=5
                parts = basename.replace('.zip', '').split('_')
                idx = int(parts[1])
                ver = int(parts[2][1:])
                if idx not in ghost_models or ver > ghost_models[idx][1]:
                    ghost_models[idx] = (DQN.load(ghost_file), ver)
            # Convert to just models
            ghost_models = {idx: model for idx, (model, ver) in ghost_models.items()}
            print(f"Loaded {len(ghost_models)} trained ghost models")
    
    # Evaluate
    results = evaluate_pacman(
        pacman_model, args.layout,
        ghost_models=ghost_models,
        n_episodes=args.episodes,
        render=args.render,
        verbose=True
    )
    
    print_eval_summary(results, args.episodes)


def main():
    parser = argparse.ArgumentParser(description='Mixed Adversarial Training')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--rounds', type=int, default=10, help='Training rounds')
    parser.add_argument('--layout', type=str, default='mediumClassic', help='Game layout')
    parser.add_argument('--pacman', type=str, help='Pac-Man model path')
    parser.add_argument('--ghost-steps', type=int, default=50000, help='Ghost training timesteps per round')
    parser.add_argument('--pacman-steps', type=int, default=100000, help='Pac-Man training timesteps per round')
    parser.add_argument('--resume', type=str, default=None, help='Resume from a previous run directory')
    parser.add_argument('--start-round', type=int, default=1, help='Starting round (for resume)')
    # Eval-specific args
    parser.add_argument('--episodes', type=int, default=100, help='Eval episodes')
    parser.add_argument('--render', action='store_true', help='Render during eval')
    parser.add_argument('--run-dir', type=str, default=None, help='Run directory with trained ghost models (for eval)')
    args = parser.parse_args()
    
    # Check mode
    if args.eval:
        if not args.pacman:
            parser.error("--pacman required for evaluation")
        evaluate_mixed(args)
        return
    
    # Training mode requires --pacman
    if not args.pacman:
        parser.error("--pacman required for training")
    
    # Print header
    print(f"\n{'='*60}")
    print(f"MIXED ADVERSARIAL TRAINING")
    print(f"  Layout: {args.layout}")
    print(f"  Rounds: {args.rounds}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
        print(f"  Starting round: {args.start_round}")
    print(f"{'='*60}")
    
    # Handle resume vs new run
    if args.resume:
        run_dir = args.resume
        dirs = {
            'models': os.path.join(run_dir, 'models'),
            'logs': os.path.join(run_dir, 'logs')
        }
        print(f"Resuming in: {run_dir}\n")
    else:
        run_dir, dirs = create_mixed_dirs()
        print(f"Output: {run_dir}\n")
    
    # Load pretrained Pac-Man
    print(f"Loading: {args.pacman}")
    pacman_model = MaskablePPO.load(args.pacman)
    pacman_path = args.pacman
    
    # Baseline evaluation (only if not resuming)
    if not args.resume:
        baseline = quick_evaluate(pacman_model, args.layout)
        print(f"Baseline vs random: {baseline*100:.1f}%")
        pacman_model.save(os.path.join(dirs['models'], "pacman_v0"))
    
    # Get ghost count from layout
    temp_env = PacmanEnv(layout_name=args.layout)
    num_ghosts = temp_env.num_ghosts
    temp_env.close()
    print(f"Ghosts: {num_ghosts}")
    
    # Initialize ghost models dict
    ghost_models = {i: None for i in range(1, num_ghosts + 1)}
    
    # Load existing ghost models if resuming
    if args.resume and args.start_round > 1:
        prev_ghost_version = args.start_round
        for ghost_idx in range(1, num_ghosts + 1):
            ghost_path = os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{prev_ghost_version}.zip")
            if os.path.exists(ghost_path):
                ghost_models[ghost_idx] = DQN.load(ghost_path)
                print(f"  Loaded: ghost_{ghost_idx}_v{prev_ghost_version}")
    
    # Main training loop
    for round_num in range(args.start_round, args.rounds + 1):
        print(f"\n{'─'*60}")
        print(f"ROUND {round_num}/{args.rounds}")
        print(f"{'─'*60}")
        
        # Phase 1: Train ghosts
        ghost_version = round_num
        print(f"\nPhase: Train Ghosts (v{ghost_version})")
        ghost_order = list(range(1, num_ghosts + 1))
        np.random.shuffle(ghost_order)
        for ghost_idx in ghost_order:
            ghost_models[ghost_idx] = train_ghost(
                ghost_idx, pacman_model, ghost_models,
                args.layout, dirs, args.ghost_steps, ghost_version
            )
        
        # Phase 2: Train Pac-Man
        pacman_version = round_num
        print(f"\nPhase: Train Pac-Man (v{pacman_version})")
        pacman_model, pacman_path = train_pacman(
            pacman_path, args.layout, dirs, ghost_models,
            args.pacman_steps, pacman_version
        )
        
        # Evaluate progress
        wr_random = quick_evaluate(pacman_model, args.layout)
        wr_trained = quick_evaluate(pacman_model, args.layout, ghost_models, n=30)
        print(f"\n  Results: {wr_random*100:.1f}% vs random, {wr_trained*100:.1f}% vs trained")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    
    final_random = quick_evaluate(pacman_model, args.layout, n=100)
    final_trained = quick_evaluate(pacman_model, args.layout, ghost_models, n=50)
    
    print(f"  vs Random:  {final_random*100:.1f}%")
    print(f"  vs Trained: {final_trained*100:.1f}%")
    
    # Save best model
    pacman_model.save(os.path.join(dirs['models'], "pacman_best"))
    print(f"\nSaved: {dirs['models']}/pacman_best.zip")
    
    # Print tensorboard instructions
    print(f"\n{'='*60}")
    print(f"VIEW TENSORBOARD")
    print(f"{'='*60}")
    print(f"Run this command:")
    print(f"  tensorboard --logdir={dirs['logs']}")
    print(f"\nThen open: http://localhost:6006")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
