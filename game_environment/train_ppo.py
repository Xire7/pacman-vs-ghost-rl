#!/usr/bin/env python3
"""
PPO Training Script for Pac-Man Agent

This script trains a Pac-Man agent using Proximal Policy Optimization (PPO)
from stable-baselines3.

Usage:
    python train_ppo.py                          # Train with defaults
    python train_ppo.py --layout mediumClassic   # Train on specific layout
    python train_ppo.py --timesteps 500000       # Train for more steps
    python train_ppo.py --eval                   # Evaluate a trained model
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from gym_env import PacmanEnv, make_pacman_env
import ghostAgents


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to log additional training metrics.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.losses = 0
        self.episodes = 0
        
    def _on_step(self) -> bool:
        # Check for episode end
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                self.episodes += 1
                info = self.locals.get('infos', [{}])[i]
                
                if info.get('win', False):
                    self.wins += 1
                elif info.get('lose', False):
                    self.losses += 1
                
                # Log every 100 episodes
                if self.episodes % 100 == 0:
                    win_rate = self.wins / max(1, self.wins + self.losses)
                    if self.verbose > 0:
                        print(f"Episodes: {self.episodes}, "
                              f"Win Rate: {win_rate:.2%}, "
                              f"Wins: {self.wins}, Losses: {self.losses}")
                    
                    # Log to tensorboard
                    self.logger.record('custom/win_rate', win_rate)
                    self.logger.record('custom/total_episodes', self.episodes)
                    self.logger.record('custom/wins', self.wins)
                    self.logger.record('custom/losses', self.losses)
        
        return True


def create_training_env(layout_name, ghost_type, num_envs, max_steps):
    """Create vectorized training environment."""
    
    def make_env(rank):
        def _init():
            env = make_pacman_env(
                layout_name=layout_name,
                ghost_type=ghost_type,
                max_steps=max_steps,
                render_mode=None,  # No rendering during training
                reward_shaping=True
            )
            return env
        return _init
    
    if num_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        # Use SubprocVecEnv for parallel training (faster but more memory)
        env = DummyVecEnv([make_env(i) for i in range(num_envs)])
        # Note: SubprocVecEnv has issues with Tkinter, so we use DummyVecEnv
    
    # Wrap with VecMonitor for logging
    env = VecMonitor(env)
    
    return env


def create_eval_env(layout_name, ghost_type, max_steps):
    """Create evaluation environment."""
    env = make_pacman_env(
        layout_name=layout_name,
        ghost_type=ghost_type,
        max_steps=max_steps,
        render_mode=None,
        reward_shaping=True
    )
    return DummyVecEnv([lambda: env])


def train(args):
    """Main training function."""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_{args.layout}_{timestamp}"
    log_dir = os.path.join(args.log_dir, run_name)
    model_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 60)
    print("PPO Training for Pac-Man")
    print("=" * 60)
    print(f"Layout: {args.layout}")
    print(f"Ghost Type: {args.ghost_type}")
    print(f"Total Timesteps: {args.timesteps:,}")
    print(f"Parallel Environments: {args.num_envs}")
    print(f"Log Directory: {log_dir}")
    print(f"Model Directory: {model_dir}")
    print("=" * 60)
    
    # Check environment compatibility first
    print("\nChecking environment compatibility...")
    test_env = make_pacman_env(
        layout_name=args.layout,
        ghost_type=args.ghost_type,
        max_steps=args.max_steps
    )
    try:
        check_env(test_env, warn=True)
        print("Environment check passed!\n")
    except Exception as e:
        print(f"Environment check failed: {e}")
        print("Continuing anyway...\n")
    test_env.close()
    
    # Create training environment
    print("Creating training environment...")
    env = create_training_env(
        layout_name=args.layout,
        ghost_type=args.ghost_type,
        num_envs=args.num_envs,
        max_steps=args.max_steps
    )
    
    # Create evaluation environment
    eval_env = create_eval_env(
        layout_name=args.layout,
        ghost_type=args.ghost_type,
        max_steps=args.max_steps
    )
    
    # Define PPO hyperparameters
    # These are tuned for the Pac-Man domain
    ppo_params = {
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': args.ent_coef,  # Entropy coefficient for exploration
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'verbose': 1,
        'tensorboard_log': log_dir,
        'device': 'auto',
    }
    
    # Network architecture
    policy_kwargs = {
        'net_arch': dict(
            pi=[256, 256],  # Policy network
            vf=[256, 256]   # Value network
        )
    }
    
    # Create or load model
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = PPO.load(args.resume, env=env, **ppo_params)
    else:
        print("Creating new PPO model...")
        model = PPO(
            policy='MlpPolicy',
            env=env,
            policy_kwargs=policy_kwargs,
            **ppo_params
        )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=model_dir,
        name_prefix='ppo_pacman',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, 'best'),
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    
    metrics_callback = TrainingMetricsCallback(verbose=1)
    
    callbacks = CallbackList([
        checkpoint_callback,
        eval_callback,
        metrics_callback,
    ])
    
    # Train!
    print("\nStarting training...")
    print("Press Ctrl+C to stop training early (model will be saved)\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    # Save final model
    final_path = os.path.join(model_dir, 'final_model')
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Final Performance: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print("\nTraining complete!")
    print(f"Best model saved in: {os.path.join(model_dir, 'best')}")
    print(f"Tensorboard logs in: {log_dir}")
    print(f"\nTo view training curves: tensorboard --logdir {log_dir}")
    
    return model


def evaluate(args):
    """Evaluate a trained model."""
    
    print("=" * 60)
    print("Evaluating PPO Pac-Man Agent")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Layout: {args.layout}")
    print(f"Episodes: {args.eval_episodes}")
    print(f"Render: {args.render}")
    if args.render:
        print(f"Frame time: {args.frame_time}s")
    print("=" * 60)
    
    # Create environment
    render_mode = 'human' if args.render else None
    env = make_pacman_env(
        layout_name=args.layout,
        ghost_type=args.ghost_type,
        max_steps=args.max_steps,
        render_mode=render_mode,
        reward_shaping=True
    )
    
    # Load model - strip .zip extension if present (SB3 adds it automatically)
    model_path = args.model_path
    if model_path.endswith('.zip'):
        model_path = model_path[:-4]
    model = PPO.load(model_path)
    
    # Run evaluation
    wins = 0
    losses = 0
    total_rewards = []
    total_scores = []
    
    import time
    for episode in range(args.eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if args.render:
                env.render()
                time.sleep(args.frame_time)  # Slow down for visibility
        
        if info.get('win', False):
            wins += 1
            result = "WIN"
        elif info.get('lose', False):
            losses += 1
            result = "LOSE"
        else:
            result = "TIMEOUT"
        
        total_rewards.append(episode_reward)
        total_scores.append(info.get('raw_score', 0))
        
        print(f"Episode {episode + 1}: {result}, "
              f"Reward: {episode_reward:.2f}, Score: {info.get('raw_score', 0):.0f}")
    
    env.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes: {args.eval_episodes}")
    print(f"Wins: {wins} ({100 * wins / args.eval_episodes:.1f}%)")
    print(f"Losses: {losses} ({100 * losses / args.eval_episodes:.1f}%)")
    print(f"Timeouts: {args.eval_episodes - wins - losses}")
    print(f"Mean Reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"Mean Score: {np.mean(total_scores):.2f} +/- {np.std(total_scores):.2f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for Pac-Man')
    
    # Mode selection
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate a trained model instead of training')
    
    # Environment settings
    parser.add_argument('--layout', type=str, default='smallGrid',
                       choices=['smallGrid', 'mediumGrid', 'smallClassic', 
                               'mediumClassic', 'testClassic', 'trickyClassic'],
                       help='Layout/map to train on')
    parser.add_argument('--ghost-type', type=str, default='random',
                       choices=['random', 'directional'],
                       help='Type of ghost behavior')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    
    # Training settings
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps')
    parser.add_argument('--num-envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to model to resume training from')
    
    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Number of steps per update')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Minibatch size')
    parser.add_argument('--n-epochs', typie=int, default=10,
                       help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help='Entropy coefficient for exploration')
    
    # Evaluation and checkpointing
    parser.add_argument('--eval-freq', type=int, default=10000,
                       help='Evaluate every N steps')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--save-freq', type=int, default=50000,
                       help='Save checkpoint every N steps')
    
    # Directories
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Directory for saved models')
    
    # Evaluation mode arguments
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for evaluation')
    parser.add_argument('--render', action='store_true',
                       help='Render during evaluation')
    parser.add_argument('--frame-time', type=float, default=0.1,
                       help='Delay between frames in seconds (default: 0.1, use higher for slower)')
    
    args = parser.parse_args()
    
    if args.eval:
        if args.model_path is None:
            parser.error("--model-path is required for evaluation mode")
        evaluate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
