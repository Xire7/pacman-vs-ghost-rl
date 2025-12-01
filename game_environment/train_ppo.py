#!/usr/bin/env python3
"""MaskablePPO training script for Pac-Man."""

import argparse
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from gym_env import make_masked_pacman_env
from training_utils import (
    create_vec_env,
    linear_schedule,
    MetricsCallback,
    NormalizeSyncCallback,
    evaluate_pacman,
    print_eval_summary,
)


def train(args):
    """Train a MaskablePPO agent."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_{args.layout}_{timestamp}"
    log_dir = os.path.join(args.log_dir, run_name)
    model_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Handle normalization stats for resume
    norm_stats_path = None
    if args.resume and args.normalize:
        resume_dir = os.path.dirname(args.resume)
        candidate = os.path.join(resume_dir, 'vecnormalize.pkl')
        if os.path.exists(candidate):
            norm_stats_path = candidate
            print(f"Found normalization stats: {norm_stats_path}")
    
    device = 'cpu' if args.cpu else 'auto'
    
    # Print training config
    print(f"\n{'='*60}")
    print(f"Training PPO on {args.layout}")
    print(f"Timesteps: {args.timesteps:,} | Envs: {args.num_envs}")
    print(f"Device: {device} | Normalize: {args.normalize}")
    print(f"n_steps: {args.n_steps} | batch_size: {args.batch_size} | clip_range: {args.clip_range}")
    print(f"n_epochs: {args.n_epochs} | gamma: {args.gamma} | target_kl: {args.target_kl}")
    print(f"{'='*60}\n")
    
    # Create environments using shared utility
    env = create_vec_env(args.layout, args.ghost_type, args.num_envs, args.max_steps,
                         normalize=args.normalize, norm_stats_path=norm_stats_path, training=True)
    
    if args.normalize:
        eval_env = create_vec_env(args.layout, args.ghost_type, 1, args.max_steps,
                                  normalize=True, norm_stats_path=norm_stats_path, training=False)
        if hasattr(env, 'obs_rms') and hasattr(eval_env, 'obs_rms'):
            eval_env.obs_rms = env.obs_rms
            eval_env.ret_rms = env.ret_rms
    else:
        eval_env = create_vec_env(args.layout, args.ghost_type, 1, args.max_steps,
                                  normalize=False, training=False)
    
    # Policy configuration
    policy_kwargs = {
        'net_arch': dict(pi=args.net_arch, vf=args.net_arch),
        'activation_fn': torch.nn.Tanh,
        'ortho_init': True,
    }
    
    # Create or load model
    if args.resume:
        print(f"Loading model from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env, tensorboard_log=log_dir, device=device)
        print(f"Network: {model.policy.net_arch}")
        
        if args.lr_decay:
            lr_schedule = linear_schedule(args.lr, args.lr_final)
            model.learning_rate = lr_schedule
            model.lr_schedule = lr_schedule
            print(f"LR schedule: {args.lr} → {args.lr_final} (linear decay)")
        else:
            model.learning_rate = args.lr
            model.lr_schedule = lambda _: args.lr
            print(f"LR: {args.lr} (constant)")
    else:
        if args.lr_decay:
            lr = linear_schedule(args.lr, args.lr_final)
            print(f"LR schedule: {args.lr} → {args.lr_final} (linear decay)")
        else:
            lr = args.lr
            print(f"LR: {args.lr} (constant)")
        
        model = MaskablePPO(
            'MlpPolicy', env,
            learning_rate=lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=0.5,
            target_kl=args.target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
            device=device,
        )
        print(f"Network: {args.net_arch}")
    
    # Setup callbacks
    callback_list = [
        CheckpointCallback(save_freq=50000 // args.num_envs, save_path=model_dir, name_prefix='ppo'),
        MaskableEvalCallback(eval_env, best_model_save_path=os.path.join(model_dir, 'best'),
                             log_path=log_dir, eval_freq=25000 // args.num_envs, n_eval_episodes=20),
        MetricsCallback(log_freq=50),
    ]
    
    if args.normalize:
        callback_list.append(NormalizeSyncCallback(eval_env, sync_freq=10000))
    
    callbacks = CallbackList(callback_list)
    
    # Train
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    
    # Save final model
    model.save(os.path.join(model_dir, 'final_model'))
    
    if args.normalize and hasattr(env, 'save'):
        env.save(os.path.join(model_dir, 'vecnormalize.pkl'))
        print(f"VecNormalize stats saved to {model_dir}/vecnormalize.pkl")
    
    print(f"\nModel saved to {model_dir}")
    
    env.close()
    eval_env.close()


def evaluate(args):
    """Evaluate a trained model."""
    print(f"\nEvaluating {args.model_path} on {args.layout}")
    print(f"Episodes: {args.episodes}\n")
    
    model = MaskablePPO.load(args.model_path)
    
    results = evaluate_pacman(
        model, args.layout,
        ghost_type=args.ghost_type,
        n_episodes=args.episodes,
        render=args.render,
        verbose=True
    )
    
    print_eval_summary(results, args.episodes)


def main():
    parser = argparse.ArgumentParser(description='Train/Evaluate PPO Pac-Man')
    parser.add_argument('--eval', action='store_true', help='Evaluate mode')
    parser.add_argument('--layout', default='mediumClassic', help='Game layout')
    parser.add_argument('--ghost-type', default='random', choices=['random', 'directional'])
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--num-envs', type=int, default=8)
    
    # Learning rate settings
    parser.add_argument('--lr-decay', action='store_true', help='Enable linear LR decay')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Initial learning rate')
    parser.add_argument('--lr-final', type=float, default=1e-5, help='Final LR when using decay')
    
    # PPO hyperparameters
    parser.add_argument('--n-steps', type=int, default=256, help='Steps per env per update')
    parser.add_argument('--batch-size', type=int, default=64, help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='PPO epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.1, help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--target-kl', type=float, default=0.02, help='Target KL for early stopping')
    
    # Network architecture
    parser.add_argument('--net-arch', type=int, nargs='+', default=[256, 256],
                        help='Network architecture (e.g., --net-arch 256 256)')
    
    # Other settings
    parser.add_argument('--normalize', action='store_true', 
                        help='Use VecNormalize for obs/reward normalization')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
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
