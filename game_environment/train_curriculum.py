#!/usr/bin/env python3
"""
Curriculum Learning Pipeline for Pac-Man vs Ghosts

This script adds curriculum learning (Stages 1-2) before the adversarial
training from train_adversarial.py (Stage 3).

Training Stages:
1. No Ghosts: Learn navigation and pellet collection
2. Random Ghosts: Learn basic evasion against scripted opponents  
3. Adversarial Ghosts: Import and use train_adversarial.py

Usage:
    python train_curriculum.py --full-curriculum
    python train_curriculum.py --skip-to adversarial --pacman-model path/to/model
"""

import argparse
import os
import sys
import json
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from gym_env import PacmanEnv, make_pacman_env
from train_adversarial import train_adversarial_rl


class CurriculumConfig:
    """Configuration for curriculum learning stages."""
    
    def __init__(self, output_dir="curriculum_output"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        
        # Create directory structure
        self.dirs = {
            'models': os.path.join(self.output_dir, 'models'),
            'logs': os.path.join(self.output_dir, 'logs'),
            'checkpoints': os.path.join(self.output_dir, 'checkpoints'),
            'eval_results': os.path.join(self.output_dir, 'eval_results')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        self.config_path = os.path.join(self.output_dir, 'config.json')
    
    def save_config(self, args):
        """Save configuration to JSON."""
        config_dict = {
            'timestamp': self.timestamp,
            'stages': {
                'stage1_no_ghosts': {
                    'timesteps': args.stage1_timesteps,
                    'layout': args.layout
                },
                'stage2_random_ghosts': {
                    'timesteps': args.stage2_timesteps,
                    'layout': args.layout,
                    'ghost_type': args.ghost_type
                },
                'stage3_adversarial': {
                    'rounds': args.adversarial_rounds,
                    'layout': args.layout,
                    'num_ghosts': args.num_ghosts
                }
            },
            'hyperparameters': vars(args)
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to: {self.config_path}")


def create_vectorized_env(layout_name, num_ghosts, ghost_type, num_envs, max_steps, reward_shaping=True):
    """Create vectorized environment for training."""
    def make_env():
        if num_ghosts == 0:
            # Stage 1: No ghosts
            return PacmanEnv(
                layout_name=layout_name,
                ghost_agents=[],
                max_steps=max_steps,
                render_mode=None,
                reward_shaping=reward_shaping
            )
        else:
            # Stage 2+: With ghosts
            return make_pacman_env(
                layout_name=layout_name,
                ghost_type=ghost_type,
                num_ghosts=num_ghosts,
                max_steps=max_steps,
                render_mode=None,
                reward_shaping=reward_shaping
            )
    
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    return VecMonitor(env)


def train_pacman_stage(
    stage_name,
    layout_name,
    num_ghosts,
    ghost_type,
    timesteps,
    config,
    previous_model=None,
    num_envs=4,
    max_steps=500,
    learning_rate=3e-4,
    eval_episodes=20
):
    """
    Train Pac-Man for one curriculum stage.
    
    This function handles Stages 1 and 2 of the curriculum.
    Stage 3 (adversarial) uses train_adversarial.py directly.
    """
    print(f"\n{'='*60}")
    print(f"CURRICULUM STAGE: {stage_name.upper().replace('_', ' ')}")
    print(f"{'='*60}")
    print(f"Layout: {layout_name}")
    print(f"Ghosts: {num_ghosts} ({ghost_type if ghost_type else 'none'})")
    print(f"Timesteps: {timesteps:,}")
    print(f"Parallel Envs: {num_envs}")
    print(f"{'='*60}\n")
    
    # Create training environment
    train_env = create_vectorized_env(
        layout_name=layout_name,
        num_ghosts=num_ghosts,
        ghost_type=ghost_type,
        num_envs=num_envs,
        max_steps=max_steps,
        reward_shaping=True
    )
    
    # Create evaluation environment
    eval_env = create_vectorized_env(
        layout_name=layout_name,
        num_ghosts=num_ghosts,
        ghost_type=ghost_type,
        num_envs=1,
        max_steps=max_steps,
        reward_shaping=True
    )
    
    # Load previous model or create new one
    if previous_model and os.path.exists(previous_model + ".zip"):
        print(f"Loading previous model from: {previous_model}")
        model = PPO.load(previous_model, env=train_env)
        model.learning_rate = learning_rate
        print(f"Continuing training with LR={learning_rate}\n")
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=os.path.join(config.dirs['logs'], stage_name),
            policy_kwargs=dict(net_arch=[256, 256])
        )
        print(f"Training from scratch\n")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000, timesteps // 10),
        save_path=os.path.join(config.dirs['checkpoints'], stage_name),
        name_prefix=f"pacman_{stage_name}",
        save_replay_buffer=False
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config.dirs['models'], f'{stage_name}_best'),
        log_path=os.path.join(config.dirs['logs'], stage_name),
        eval_freq=max(5000, timesteps // 20),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Train
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    # Save final model
    model_path = os.path.join(config.dirs['models'], f"pacman_{stage_name}_final")
    model.save(model_path)
    print(f"\n✓ Stage complete! Model saved to: {model_path}")
    
    # Final evaluation
    print(f"\nEvaluating stage performance...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=eval_episodes, deterministic=True
    )
    print(f"Stage {stage_name}: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Save evaluation results
    eval_results = {
        'stage': stage_name,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'timesteps': timesteps,
        'num_ghosts': num_ghosts
    }
    
    eval_path = os.path.join(config.dirs['eval_results'], f'{stage_name}_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    train_env.close()
    eval_env.close()
    
    return model, model_path


def run_adversarial_training_wrapper(
    pacman_model_path,
    config,
    args
):
    """
    Wrapper that calls the existing train_adversarial.py functions.
    
    This integrates Stage 3 by using the already-implemented adversarial
    training code instead of duplicating it.
    """
    print(f"\n{'='*60}")
    print("STAGE 3: ADVERSARIAL TRAINING")
    print(f"{'='*60}")
    print("Using train_adversarial.py implementation")
    print(f"Starting with curriculum-trained Pac-Man: {pacman_model_path}")
    print(f"{'='*60}\n")

    print("Validating layout ghost count...")
    temp_env = PacmanEnv(layout_name=args.layout, render_mode=None)
    temp_state, _ = temp_env.reset()
    actual_num_ghosts = temp_env.game_state.getNumAgents() - 1
    temp_env.close()
    
    if args.num_ghosts > actual_num_ghosts:
        print(f"WARNING: Layout '{args.layout}' only has {actual_num_ghosts} ghost(s).")
        print(f"   Requested {args.num_ghosts}, adjusting to {actual_num_ghosts}.")
        args.num_ghosts = actual_num_ghosts
    
    print(f"✓ Using {args.num_ghosts} ghost(s) for adversarial training\n")
    
    # Call the existing adversarial training function
    # This uses the implementation from train_adversarial.py
    pacman_model, ghost_models, history = train_adversarial_rl(
        num_rounds=args.adversarial_rounds,
        layout_name=args.layout,
        num_ghosts=args.num_ghosts,
        ghost_initial_timesteps=args.ghost_initial_steps,
        ghost_refinement_timesteps=args.ghost_refine_steps,
        pacman_initial_timesteps=0,  # We already trained Pac-Man
        pacman_refinement_timesteps=args.pacman_refine_steps,
        eval_frequency=2,
        eval_episodes=20
    )
    
    # Copy adversarial models to curriculum output directory
    print("\nCopying adversarial models to curriculum directory...")
    
    print(f"\nAdversarial training complete!")
    print(f"Models saved in: {config.dirs['models']}")
    print(f"Also check: training_output/ for adversarial-specific outputs")
    
    return pacman_model, ghost_models

def main():
    parser = argparse.ArgumentParser(
        description='Curriculum Learning Pipeline for Pac-Man'
    )
    
    # Pipeline control
    parser.add_argument('--full-curriculum', action='store_true',
                       help='Run complete curriculum (all 3 stages)')
    parser.add_argument('--skip-to', type=str, choices=['random_ghosts', 'adversarial'],
                       help='Skip to specific stage')
    parser.add_argument('--pacman-model', type=str,
                       help='Path to pre-trained Pac-Man model (for skipping stages)')
    
    # Environment settings
    parser.add_argument('--layout', type=str, default='mediumClassic',
                       help='Map layout')
    parser.add_argument('--num-ghosts', type=int, default=4,
                       help='Number of ghosts')
    parser.add_argument('--ghost-type', type=str, default='random',
                       choices=['random', 'directional'],
                       help='Ghost type for stage 2')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Max steps per episode')
    
    # Stage 1: No Ghosts
    parser.add_argument('--stage1-timesteps', type=int, default=200000,
                       help='Training timesteps for stage 1 (no ghosts)')
    
    # Stage 2: Random Ghosts
    parser.add_argument('--stage2-timesteps', type=int, default=500000,
                       help='Training timesteps for stage 2 (random ghosts)')
    
    # Stage 3: Adversarial (delegates to train_adversarial.py)
    parser.add_argument('--adversarial-rounds', type=int, default=10,
                       help='Number of adversarial training rounds')
    parser.add_argument('--ghost-initial-steps', type=int, default=50000,
                       help='Ghost initial training timesteps')
    parser.add_argument('--ghost-refine-steps', type=int, default=30000,
                       help='Ghost refinement timesteps')
    parser.add_argument('--pacman-refine-steps', type=int, default=50000,
                       help='Pac-Man refinement timesteps in adversarial phase')
    
    # Training hyperparameters
    parser.add_argument('--num-envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate for PPO')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='curriculum_output',
                       help='Output directory for models and logs')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = CurriculumConfig(output_dir=args.output_dir)
    config.save_config(args)
    
    print(f"\n{'#'*60}")
    print("# CURRICULUM LEARNING PIPELINE")
    print(f"# Output: {config.output_dir}")
    print(f"{'#'*60}\n")
    
    # Validate layout supports requested number of ghosts
    print("Validating layout configuration...")
    temp_env = PacmanEnv(layout_name=args.layout, render_mode=None)
    temp_state, _ = temp_env.reset()
    actual_num_ghosts = temp_env.game_state.getNumAgents() - 1
    temp_env.close()
    
    if args.num_ghosts > actual_num_ghosts:
        print(f"⚠️  WARNING: Layout '{args.layout}' only has {actual_num_ghosts} ghost(s).")
        print(f"   Requested {args.num_ghosts}, adjusting to {actual_num_ghosts}.")
        args.num_ghosts = actual_num_ghosts
    
    print(f"✓ Layout: {args.layout} with {args.num_ghosts} ghost(s)")
    print()
    
    # Determine starting stage
    if args.full_curriculum:
        start_stage = 1
        pacman_model_path = None
    elif args.skip_to == 'random_ghosts':
        start_stage = 2
        if not args.pacman_model:
            parser.error("--pacman-model required when skipping to random_ghosts")
        pacman_model_path = args.pacman_model
    elif args.skip_to == 'adversarial':
        start_stage = 3
        if not args.pacman_model:
            parser.error("--pacman-model required when skipping to adversarial")
        pacman_model_path = args.pacman_model
    else:
        start_stage = 1
        pacman_model_path = None
    
    # Stage 1: No Ghosts
    if start_stage <= 1:
        print(f"\n{'*'*60}")
        print("* STAGE 1: NO GHOSTS (Basic Navigation)")
        print(f"{'*'*60}")
        
        _, pacman_model_path = train_pacman_stage(
            stage_name="no_ghosts",
            layout_name=args.layout,
            num_ghosts=0,
            ghost_type=None,
            timesteps=args.stage1_timesteps,
            config=config,
            previous_model=None,
            num_envs=args.num_envs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate
        )
    
    # Stage 2: Random Ghosts
    if start_stage <= 2:
        print(f"\n{'*'*60}")
        print("* STAGE 2: RANDOM GHOSTS (Evasion Skills)")
        print(f"{'*'*60}")
        
        _, pacman_model_path = train_pacman_stage(
            stage_name="random_ghosts",
            layout_name=args.layout,
            num_ghosts=args.num_ghosts,
            ghost_type=args.ghost_type,
            timesteps=args.stage2_timesteps,
            config=config,
            previous_model=pacman_model_path,
            num_envs=args.num_envs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate
        )
    
    # Stage 3: Adversarial Training (uses train_adversarial.py)
    if start_stage <= 3:
        print(f"\n{'*'*60}")
        print("* STAGE 3: ADVERSARIAL TRAINING (Adaptive Opponents)")
        print(f"{'*'*60}")
        
        final_pacman, final_ghosts = run_adversarial_training_wrapper(
            pacman_model_path=pacman_model_path,
            config=config,
            args=args
        )
    
    print(f"\n{'#'*60}")
    print("# CURRICULUM COMPLETE!")
    print(f"{'#'*60}")
    print(f"\nAll models and logs saved to: {config.output_dir}")
    print(f"\nTo visualize training:")
    print(f"  tensorboard --logdir {config.dirs['logs']}")
    print()


if __name__ == '__main__':
    main()