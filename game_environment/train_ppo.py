#!/usr/bin/env python3
"""MaskablePPO training script for Pac-Man."""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

from gym_env import make_masked_pacman_env


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


def create_env(layout_name, ghost_type, num_envs, max_steps):
    """Create vectorized training environment."""
    def make_env():
        return make_masked_pacman_env(layout_name, ghost_type, max_steps=max_steps)
    
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    return VecMonitor(env)


def train(args):
    """Train a MaskablePPO agent."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_{args.layout}_{timestamp}"
    log_dir = os.path.join(args.log_dir, run_name)
    model_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training PPO on {args.layout}")
    print(f"Timesteps: {args.timesteps:,} | Envs: {args.num_envs}")
    print(f"{'='*60}\n")
    
    env = create_env(args.layout, args.ghost_type, args.num_envs, args.max_steps)
    eval_env = create_env(args.layout, args.ghost_type, 1, args.max_steps)
    
    policy_kwargs = {
        'net_arch': dict(pi=[256, 128], vf=[256, 128]),
        'activation_fn': torch.nn.Tanh,
    }
    
    if args.resume:
        print(f"Loading model from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env, tensorboard_log=log_dir)
    else:
        model = MaskablePPO(
            'MlpPolicy', env,
            learning_rate=2.5e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            clip_range=0.1,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
        )
    
    callbacks = CallbackList([
        CheckpointCallback(save_freq=50000 // args.num_envs, save_path=model_dir, name_prefix='ppo'),
        MaskableEvalCallback(eval_env, best_model_save_path=os.path.join(model_dir, 'best'),
                            log_path=log_dir, eval_freq=25000 // args.num_envs, n_eval_episodes=20),
        MetricsCallback(log_freq=50),
    ])
    
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    
    model.save(os.path.join(model_dir, 'final_model'))
    print(f"\nModel saved to {model_dir}")
    
    env.close()
    eval_env.close()


def evaluate(args):
    """Evaluate a trained model."""
    print(f"\nEvaluating {args.model_path} on {args.layout}")
    print(f"Episodes: {args.episodes}\n")
    
    render_mode = 'human' if args.render else None
    env = make_masked_pacman_env(args.layout, args.ghost_type, max_steps=args.max_steps, render_mode=render_mode)
    model = MaskablePPO.load(args.model_path)
    
    wins, total_reward = 0, []
    
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
            if args.render:
                env.render()
                time.sleep(0.05)
        
        if info.get('win'):
            wins += 1
        total_reward.append(ep_reward)
        print(f"Ep {ep+1}: {'WIN' if info.get('win') else 'LOSE'} | Reward: {ep_reward:.1f}")
    
    env.close()
    print(f"\nWin Rate: {wins}/{args.episodes} ({100*wins/args.episodes:.1f}%)")
    print(f"Mean Reward: {np.mean(total_reward):.1f} ± {np.std(total_reward):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Train/Evaluate PPO Pac-Man')
    parser.add_argument('--eval', action='store_true', help='Evaluate mode')
    parser.add_argument('--layout', default='mediumClassic', help='Game layout')
    parser.add_argument('--ghost-type', default='random', choices=['random', 'directional'])
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--ent-coef', type=float, default=0.05, help='Entropy coefficient')
    parser.add_argument('--resume', type=str, help='Resume from model path')
    parser.add_argument('--log-dir', default='./logs')
    parser.add_argument('--model-dir', default='./models')
    parser.add_argument('--model-path', type=str, help='Model path for evaluation')
    parser.add_argument('--episodes', type=int, default=100, help='Eval episodes')
    parser.add_argument('--render', action='store_true', help='Render during eval')
    
    args = parser.parse_args()
    
    if args.eval:
        if not args.model_path:
            parser.error("--model-path required for evaluation")
        evaluate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
