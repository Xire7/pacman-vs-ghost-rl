#!/usr/bin/env python3
"""
Unified Training Script for Pac-Man vs Ghosts.

Training Modes:
  - pacman: Train Pac-Man against random/directional ghosts
  - adversarial: Alternating Pac-Man (PPO) and Ghost (DQN) training

Usage:
  python train.py --mode pacman --timesteps 500000
  python train.py --mode adversarial --iterations 10
  python train.py --eval --training-dir <path>
  python train.py --render --training-dir <path> --games 5
"""

import argparse
import os
import sys
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from gym_env import PacmanEnv
from ghost_agent import IndependentGhostEnv, ActionMaskingWrapper
from training_utils import (
    linear_schedule,
    WinRateCallback,
    CatchRateCallback,
    evaluate_with_ghosts,
    render_game,
)


class Trainer:
    """Unified trainer for Pac-Man and Ghost agents."""
    
    def __init__(
        self,
        layout_name: str = 'mediumClassic',
        output_dir: str = None,
        device: str = 'auto',
    ):
        self.layout_name = layout_name
        self.device = device
        
        # Get ghost count from layout
        temp_env = PacmanEnv(layout_name=layout_name)
        self.num_ghosts = temp_env.num_ghosts
        temp_env.close()
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"training/{layout_name}_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'pacman').mkdir(exist_ok=True)
        (self.output_dir / 'ghosts').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Models
        self.pacman_model = None
        self.pacman_vec_normalize = None
        self.ghost_models: Dict[int, DQN] = {}
        
    def train_pacman(
        self,
        timesteps: int = 500000,
        num_envs: int = 8,
        ghost_type: str = 'random',
        use_trained_ghosts: bool = False,
        trained_policy_prob: float = 0.7,
    ) -> MaskablePPO:
        """Train Pac-Man agent."""
        print(f"\n{'='*60}")
        print(f"Training Pac-Man | {timesteps:,} timesteps | {num_envs} envs")
        print(f"{'='*60}")
        
        if use_trained_ghosts and self.ghost_models:
            print(f"Ghosts: trained policies (prob={trained_policy_prob:.0%}) + random ({1-trained_policy_prob:.0%})")
        else:
            print(f"Ghosts: {ghost_type}")
        
        # Create environments
        def make_env(rank: int):
            def _init():
                if use_trained_ghosts and self.ghost_models:
                    env = PacmanEnv(
                        layout_name=self.layout_name,
                        ghost_policies=self.ghost_models,
                        trained_policy_prob=trained_policy_prob,
                        max_steps=500
                    )
                else:
                    env = PacmanEnv(
                        layout_name=self.layout_name,
                        ghost_type=ghost_type,
                        max_steps=500
                    )
                env = ActionMasker(env, lambda e: e.action_masks())
                env = Monitor(env)
                return env
            return _init
        
        env_fns = [make_env(i) for i in range(num_envs)]
        
        train_env = DummyVecEnv(env_fns)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        
        eval_env = DummyVecEnv([make_env(9999)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        
        # Model setup
        lr_schedule = linear_schedule(3e-4, 1e-5)
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
        )
        
        if self.pacman_model is not None:
            # Continue training
            model = MaskablePPO(
                'MlpPolicy', train_env,
                learning_rate=lr_schedule,
                n_steps=512, batch_size=256, n_epochs=4,
                gamma=0.995, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, policy_kwargs=policy_kwargs,
                tensorboard_log=str(self.output_dir / 'logs'),
                device=self.device, verbose=1
            )
            model.policy.load_state_dict(self.pacman_model.policy.state_dict())
        else:
            model = MaskablePPO(
                'MlpPolicy', train_env,
                learning_rate=lr_schedule,
                n_steps=512, batch_size=256, n_epochs=4,
                gamma=0.995, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, policy_kwargs=policy_kwargs,
                tensorboard_log=str(self.output_dir / 'logs'),
                device=self.device, verbose=1
            )
        
        # Callbacks
        callbacks = CallbackList([
            EvalCallback(eval_env, eval_freq=max(10000 // num_envs, 500),
                        n_eval_episodes=20, deterministic=True, verbose=1),
            WinRateCallback(print_freq=50),
        ])
        
        # Train
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
        
        # Save
        model.save(str(self.output_dir / 'pacman' / 'model'))
        train_env.save(str(self.output_dir / 'pacman' / 'vecnormalize.pkl'))
        
        self.pacman_model = model
        self.pacman_vec_normalize = train_env
        
        train_env.close()
        eval_env.close()
        
        return model
    
    def train_ghost(
        self,
        ghost_index: int,
        timesteps: int = 100000,
        num_envs: int = 4,
        warmup_mode: bool = False,
    ) -> DQN:
        """Train a single ghost agent."""
        print(f"\n{'='*60}")
        if warmup_mode:
            print(f"WARMUP Training Ghost {ghost_index} vs RANDOM Pac-Man | {timesteps:,} timesteps")
        else:
            print(f"Training Ghost {ghost_index} vs TRAINED Pac-Man | {timesteps:,} timesteps")
        print(f"{'='*60}")
        
        pacman_policy = None if warmup_mode else self.pacman_model
        
        if not warmup_mode and self.pacman_model is None:
            raise ValueError("Must train Pac-Man first (or use warmup_mode=True)")
        
        # Get normalization stats for Pac-Man observations (only if using trained Pac-Man)
        pacman_obs_rms = None
        if not warmup_mode and self.pacman_vec_normalize is not None:
            pacman_obs_rms = self.pacman_vec_normalize.obs_rms
        
        # Other trained ghosts
        other_ghosts = {i: m for i, m in self.ghost_models.items() if i != ghost_index}
        
        # Create environments with ActionMaskingWrapper
        def make_env(rank: int):
            def _init():
                env = IndependentGhostEnv(
                    ghost_index=ghost_index,
                    layout_name=self.layout_name,
                    pacman_policy=pacman_policy,
                    other_ghost_policies=other_ghosts,
                    pacman_obs_rms=pacman_obs_rms,
                    max_steps=500
                )
                env = ActionMaskingWrapper(env)  # Handle illegal actions
                env = Monitor(env)
                return env
            return _init
        
        train_env = DummyVecEnv([make_env(i) for i in range(num_envs)])
        eval_env = DummyVecEnv([make_env(9999)])
        
        # Model setup - optimized for ghost chasing task
        policy_kwargs = dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU)
        
        # If we already have a model for this ghost, continue training
        if ghost_index in self.ghost_models:
            model = DQN(
                'MlpPolicy', train_env,
                learning_rate=5e-4,
                buffer_size=50000,
                learning_starts=500,
                batch_size=128,
                gamma=0.95,
                target_update_interval=500,
                exploration_fraction=0.5,
                exploration_final_eps=0.1,
                train_freq=4,
                gradient_steps=2,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(self.output_dir / 'logs'),
                device=self.device, verbose=1
            )
            # Load weights from previous model
            model.q_net.load_state_dict(self.ghost_models[ghost_index].q_net.state_dict())
            model.q_net_target.load_state_dict(self.ghost_models[ghost_index].q_net_target.state_dict())
        else:
            model = DQN(
                'MlpPolicy', train_env,
                learning_rate=5e-4,
                buffer_size=50000,
                learning_starts=500,
                batch_size=128,
                gamma=0.95,
                target_update_interval=500,
                exploration_fraction=0.5,
                exploration_final_eps=0.1,
                train_freq=4,
                gradient_steps=2,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(self.output_dir / 'logs'),
                device=self.device, verbose=1
            )
        
        # Callbacks
        callbacks = CallbackList([
            EvalCallback(eval_env, eval_freq=max(10000 // num_envs, 500),
                        n_eval_episodes=10, deterministic=True, verbose=1),
            CatchRateCallback(print_freq=50),
        ])
        
        # Train
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
        
        # Save
        model.save(str(self.output_dir / 'ghosts' / f'ghost_{ghost_index}'))
        self.ghost_models[ghost_index] = model
        
        train_env.close()
        eval_env.close()
        
        return model
    
    def train_adversarial(
        self,
        iterations: int = 5,
        pacman_warmup_timesteps: int = 250000,
        ghost_warmup_timesteps: int = 150000,
        pacman_timesteps: int = 100000,
        ghost_timesteps: int = 100000,
        num_envs: int = 8,
        initial_trained_prob: float = 0.4,
        final_trained_prob: float = 0.9,
    ):
        """Run adversarial training with symmetric warmup and curriculum learning.
        
        Phases:
        1. Pac-Man warmup vs random ghosts
        2. Ghost warmup vs random Pac-Man (symmetric!)
        3. Adversarial iterations with curriculum
        """
        print(f"\n{'='*60}")
        print(f"ADVERSARIAL TRAINING (Symmetric Warmup + Curriculum)")
        print(f"{'='*60}")
        print(f"Layout: {self.layout_name} ({self.num_ghosts} ghosts)")
        print(f"Iterations: {iterations}")
        print(f"Pac-Man warmup: {pacman_warmup_timesteps:,} | Ghost warmup: {ghost_warmup_timesteps:,}/ghost")
        print(f"Per-iter Pac-Man: {pacman_timesteps:,} | Per-iter Ghost: {ghost_timesteps:,}")
        print(f"Trained ghost prob: {initial_trained_prob:.0%} → {final_trained_prob:.0%}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Phase 1a: Warmup Pac-Man vs random ghosts
        if pacman_warmup_timesteps > 0:
            print("\n*** PHASE 1a: PAC-MAN WARMUP (vs random ghosts) ***")
            self.train_pacman(pacman_warmup_timesteps, num_envs, ghost_type='random')
        
        # Phase 1b: Warmup Ghosts vs random Pac-Man (SYMMETRIC!)
        if ghost_warmup_timesteps > 0:
            print("\n*** PHASE 1b: GHOST WARMUP (vs random Pac-Man) ***")
            for ghost_idx in range(1, self.num_ghosts + 1):
                self.train_ghost(ghost_idx, ghost_warmup_timesteps, num_envs // 2 or 1, warmup_mode=True)
        
        # Phase 2: Adversarial iterations with curriculum learning
        for iteration in range(1, iterations + 1):
            # Compute annealed trained_policy_prob for this iteration
            if iterations > 1:
                progress = (iteration - 1) / (iterations - 1)  # 0.0 to 1.0
            else:
                progress = 1.0
            trained_policy_prob = initial_trained_prob + progress * (final_trained_prob - initial_trained_prob)
            
            print(f"\n*** ITERATION {iteration}/{iterations} (trained_prob={trained_policy_prob:.0%}) ***")
            
            # Train ghosts against trained Pac-Man
            for ghost_idx in range(1, self.num_ghosts + 1):
                self.train_ghost(ghost_idx, ghost_timesteps, num_envs // 2 or 1, warmup_mode=False)
            
            # Train Pac-Man against mixed ghosts (curriculum learning)
            self.train_pacman(
                pacman_timesteps, num_envs,
                use_trained_ghosts=True,
                trained_policy_prob=trained_policy_prob
            )
            
            # Save iteration checkpoint
            self.save(suffix=f'_iter{iteration}')
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        
        # Final evaluation
        self.evaluate()
    
    def save(self, suffix: str = ''):
        """Save all models."""
        if self.pacman_model:
            self.pacman_model.save(str(self.output_dir / 'pacman' / f'model{suffix}'))
            if self.pacman_vec_normalize:
                self.pacman_vec_normalize.save(str(self.output_dir / 'pacman' / f'vecnormalize{suffix}.pkl'))
        
        for idx, model in self.ghost_models.items():
            model.save(str(self.output_dir / 'ghosts' / f'ghost_{idx}{suffix}'))
    
    def load(self, training_dir: str, iteration: int = None):
        """Load models from a training directory."""
        training_dir = Path(training_dir)
        
        # Find latest iteration if not specified
        if iteration is None:
            pacman_files = list((training_dir / 'pacman').glob('model_iter*.zip'))
            if pacman_files:
                iterations = [int(f.stem.split('iter')[-1]) for f in pacman_files]
                iteration = max(iterations)
                print(f"Using iteration {iteration}")
        
        suffix = f'_iter{iteration}' if iteration else ''
        
        # Load Pac-Man
        pacman_path = training_dir / 'pacman' / f'model{suffix}.zip'
        if not pacman_path.exists():
            pacman_path = training_dir / 'pacman' / 'model.zip'
        
        if pacman_path.exists():
            self.pacman_model = MaskablePPO.load(str(pacman_path), device=self.device)
            print(f"Loaded Pac-Man: {pacman_path}")
            
            # Load VecNormalize
            norm_path = pacman_path.with_name(pacman_path.stem.replace('model', 'vecnormalize') + '.pkl')
            if norm_path.exists():
                def make_dummy():
                    env = PacmanEnv(layout_name=self.layout_name)
                    env = ActionMasker(env, lambda e: e.action_masks())
                    return Monitor(env)
                dummy_env = DummyVecEnv([make_dummy])
                self.pacman_vec_normalize = VecNormalize.load(str(norm_path), dummy_env)
                print(f"Loaded VecNormalize")
        
        # Load Ghosts
        ghost_dir = training_dir / 'ghosts'
        for ghost_idx in range(1, self.num_ghosts + 1):
            ghost_path = ghost_dir / f'ghost_{ghost_idx}{suffix}.zip'
            if not ghost_path.exists():
                ghost_path = ghost_dir / f'ghost_{ghost_idx}.zip'
            
            if ghost_path.exists():
                self.ghost_models[ghost_idx] = DQN.load(str(ghost_path), device=self.device)
                print(f"Loaded Ghost {ghost_idx}")
        
        # Re-randomize after loading (models contain saved seeds)
        np.random.seed(None)
        random.seed(None)
        torch.seed()  # Re-randomize PyTorch RNG
    
    def evaluate(self, episodes: int = 100, vs_random: bool = True, vs_trained: bool = True):
        """Evaluate Pac-Man against ghosts."""
        if self.pacman_model is None:
            raise ValueError("No Pac-Man model loaded")
        
        results = {}
        
        if vs_random:
            print(f"\nEvaluating vs RANDOM ghosts ({episodes} episodes)...")
            results['vs_random'] = evaluate_with_ghosts(
                self.pacman_model, self.layout_name, self.pacman_vec_normalize,
                ghost_models=None, episodes=episodes
            )
            print(f"Win Rate: {results['vs_random']['win_rate']*100:.1f}%")
        
        if vs_trained and self.ghost_models:
            print(f"\nEvaluating vs TRAINED ghosts ({episodes} episodes)...")
            results['vs_trained'] = evaluate_with_ghosts(
                self.pacman_model, self.layout_name, self.pacman_vec_normalize,
                ghost_models=self.ghost_models, episodes=episodes
            )
            print(f"Win Rate: {results['vs_trained']['win_rate']*100:.1f}%")
        
        # Save results
        with open(self.output_dir / 'eval_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def render(self, games: int = 3, vs_trained: bool = True, delay: float = 0.05):
        """Render games visually."""
        if self.pacman_model is None:
            raise ValueError("No Pac-Man model loaded")
        
        ghost_models = self.ghost_models if vs_trained else None
        ghost_type = "TRAINED" if ghost_models else "RANDOM"
        
        print(f"\nRendering {games} game(s) vs {ghost_type} ghosts\n")
        
        for game in range(games):
            print(f"--- Game {game + 1}/{games} ---")
            result = render_game(
                self.pacman_model, self.layout_name, self.pacman_vec_normalize,
                ghost_models=ghost_models, delay=delay
            )
            outcome = "PACMAN WINS!" if result['win'] else "GHOSTS WIN!"
            print(f"{outcome} Score: {result['score']}, Steps: {result['steps']}\n")


def main():
    parser = argparse.ArgumentParser(description='Pac-Man Training')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['pacman', 'adversarial'],
                       help='Training mode')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--render', action='store_true', help='Render mode')
    
    # Common
    parser.add_argument('--layout', type=str, default='mediumClassic')
    parser.add_argument('--training-dir', type=str, help='Training directory (for eval/render)')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['auto', 'cpu', 'cuda'])
    
    # Pac-Man training
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--ghost-type', type=str, default='random',
                       choices=['random', 'directional'])
    
    # Adversarial training (symmetric warmup + curriculum learning)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--pacman-warmup-timesteps', type=int, default=250000,
                       help='Pac-Man warmup timesteps vs random ghosts (0 to skip)')
    parser.add_argument('--ghost-warmup-timesteps', type=int, default=150000,
                       help='Ghost warmup timesteps per ghost vs random Pac-Man (0 to skip)')
    parser.add_argument('--pacman-timesteps', type=int, default=200000,
                       help='Pac-Man timesteps per adversarial iteration')
    parser.add_argument('--ghost-timesteps', type=int, default=150000,
                       help='Ghost timesteps per adversarial iteration (per ghost)')
    parser.add_argument('--initial-trained-prob', type=float, default=0.4,
                       help='Initial trained ghost probability (default 0.4 = 60%% random)')
    parser.add_argument('--final-trained-prob', type=float, default=0.9,
                       help='Final trained ghost probability (default 0.9 = 10%% random)')
    
    # Eval/Render
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--games', type=int, default=3)
    parser.add_argument('--delay', type=float, default=0.05)
    parser.add_argument('--vs-random', action='store_true')
    parser.add_argument('--vs-trained', action='store_true')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(
        layout_name=args.layout,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    # Handle modes
    if args.eval or args.render:
        if not args.training_dir:
            parser.error("--training-dir required for eval/render")
        
        trainer.load(args.training_dir)
        
        if args.eval:
            vs_random = args.vs_random or not args.vs_trained
            vs_trained = args.vs_trained or not args.vs_random
            trainer.evaluate(args.episodes, vs_random=vs_random, vs_trained=vs_trained)
        
        if args.render:
            trainer.render(args.games, vs_trained=not args.vs_random, delay=args.delay)
    
    elif args.mode == 'pacman':
        trainer.train_pacman(
            timesteps=args.timesteps,
            num_envs=args.num_envs,
            ghost_type=args.ghost_type,
        )
        trainer.evaluate(episodes=50, vs_random=True, vs_trained=False)
    
    elif args.mode == 'adversarial':
        trainer.train_adversarial(
            iterations=args.iterations,
            pacman_warmup_timesteps=args.pacman_warmup_timesteps,
            ghost_warmup_timesteps=args.ghost_warmup_timesteps,
            pacman_timesteps=args.pacman_timesteps,
            ghost_timesteps=args.ghost_timesteps,
            num_envs=args.num_envs,
            initial_trained_prob=args.initial_trained_prob,
            final_trained_prob=args.final_trained_prob,
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
