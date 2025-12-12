#!/usr/bin/env python3
"""MaskablePPO training script for Pac-Man"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Callable

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

from gym_env import make_masked_pacman_env


def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule


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

def create_env(layout_name, ghost_type, num_envs, max_steps, normalize=False, norm_stats_path=None, training=True):
    """Create vectorized training environment."""
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


def train(args):
    """Train a MaskablePPO agent."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_{args.layout}_{timestamp}"
    log_dir = os.path.join(args.log_dir, run_name)
    model_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    norm_stats_path = None
    if args.resume and args.normalize:
        resume_dir = os.path.dirname(args.resume)
        candidate = os.path.join(resume_dir, 'vecnormalize.pkl')
        if os.path.exists(candidate):
            norm_stats_path = candidate
            print(f"Found normalization stats: {norm_stats_path}")
    
    device = 'cpu' if args.cpu else 'auto'
    
    print(f"\n{'='*60}")
    print(f"Training PPO on {args.layout}")
    print(f"Timesteps: {args.timesteps:,} | Envs: {args.num_envs}")
    print(f"Device: {device} | Normalize: {args.normalize}")
    print(f"TensorBoard log: {log_dir}")  
    print(f"{'='*60}\n")
    
    env = create_env(args.layout, args.ghost_type, args.num_envs, args.max_steps, 
                     normalize=args.normalize, norm_stats_path=norm_stats_path, training=True)
    
    if args.normalize:
        eval_env = create_env(args.layout, args.ghost_type, 1, args.max_steps, 
                              normalize=True, norm_stats_path=norm_stats_path, training=False)
        if hasattr(env, 'obs_rms') and hasattr(eval_env, 'obs_rms'):
            eval_env.obs_rms = env.obs_rms
            eval_env.ret_rms = env.ret_rms
    else:
        eval_env = create_env(args.layout, args.ghost_type, 1, args.max_steps, 
                              normalize=False, training=False)
    
    policy_kwargs = {
        'net_arch': dict(pi=args.net_arch, vf=args.net_arch),
        'activation_fn': torch.nn.Tanh,
        'ortho_init': True,
    }
    
    if args.resume:
        print(f"Loading model from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env, device=device)
        
        model.num_timesteps = 0
        model._num_timesteps_at_start = 0
        
        model.tensorboard_log = log_dir
        
        print(f"Reset timestep counter to 0")
        print(f"New TensorBoard directory: {log_dir}")
        
        # update hyperparameters
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
    
    callback_list = [
        CheckpointCallback(save_freq=50000 // args.num_envs, save_path=model_dir, name_prefix='ppo'),
        MaskableEvalCallback(eval_env, best_model_save_path=os.path.join(model_dir, 'best'),
                            log_path=log_dir, eval_freq=25000 // args.num_envs, n_eval_episodes=20),
        MetricsCallback(log_freq=50),
    ]
    
    if args.normalize:
        callback_list.append(NormalizeSyncCallback(eval_env, sync_freq=10000))
    
    callbacks = CallbackList(callback_list)
    
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    
    model.save(os.path.join(model_dir, 'final_model'))
    
    if args.normalize and hasattr(env, 'save'):
        env.save(os.path.join(model_dir, 'vecnormalize.pkl'))
        print(f"VecNormalize stats saved to {model_dir}/vecnormalize.pkl")
    
    print(f"\nModel saved to {model_dir}")
    print(f"TensorBoard logs saved to {log_dir}")
    print(f"\nView with: tensorboard --logdir={log_dir}")
    
    env.close()
    eval_env.close()


def evaluate(args):
    """Evaluate a trained model."""
    print(f"\nEvaluating {args.model_path} on {args.layout}")
    print(f"Episodes: {args.episodes}\n")
    
    model_dir = os.path.dirname(args.model_path)
    vecnorm_path = os.path.join(model_dir, 'vecnormalize.pkl')
    
    if not os.path.exists(vecnorm_path):
        parent_dir = os.path.dirname(model_dir)
        parent_vecnorm = os.path.join(parent_dir, 'vecnormalize.pkl')
        if os.path.exists(parent_vecnorm):
            vecnorm_path = parent_vecnorm
    
    has_vecnorm = os.path.exists(vecnorm_path)
    
    if has_vecnorm:
        print(f"Found VecNormalize: {vecnorm_path}\n")
    else:
        print(f"No VecNormalize found (model trained without --normalize)\n")
    
    render_mode = 'human' if args.render else None
    
    env = make_masked_pacman_env(args.layout, args.ghost_type, 
                                 max_steps=args.max_steps, render_mode=render_mode)
    
    env = DummyVecEnv([lambda: env])
    
    if has_vecnorm:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print("VecNormalize loaded\n")
    
    model = MaskablePPO.load(args.model_path, env=env)
    
    wins, total_reward = 0, []
    
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action_masks = env.env_method('action_masks')[0]
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            
            if isinstance(action, np.ndarray):
                action = int(action.item())

            obs, reward, done, info = env.step([action])
            
            ep_reward += reward[0]
            done = done[0]
            
            if args.render:
                time.sleep(0.05)
        
        info = info[0]
        
        if info.get('win'):
            wins += 1
        total_reward.append(ep_reward)
        
        result = 'WIN' if info.get('win') else 'LOSE'
        score = info.get('raw_score', 0)
        print(f"Ep {ep+1:3d}: {result} | Score: {score:4.0f} | Reward: {ep_reward:6.1f}")
    
    env.close()
    
    print(f"\n{'='*60}")
    print(f"Win Rate: {wins}/{args.episodes} ({100*wins/args.episodes:.1f}%)")
    print(f"Mean Reward: {np.mean(total_reward):.1f} ± {np.std(total_reward):.1f}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='Train/Evaluate PPO Pac-Man')
    parser.add_argument('--eval', action='store_true', help='Evaluate mode')
    parser.add_argument('--layout', default='mediumClassic', help='Game layout')
    parser.add_argument('--ghost-type', default='random', choices=['random', 'directional'])
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--num-envs', type=int, default=16)
    
    # learning rate settings
    parser.add_argument('--lr-decay', action='store_true', help='Enable linear LR decay')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Initial learning rate')
    parser.add_argument('--lr-final', type=float, default=1e-5, help='Final LR when using decay')
    
    # PPO hyperparameters
    parser.add_argument('--n-steps', type=int, default=512, help='Steps per env per update')
    parser.add_argument('--batch-size', type=int, default=128, help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of PPO epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.1, help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--target-kl', type=float, default=0.02, help='Target KL divergence')
    
    parser.add_argument('--net-arch', type=int, nargs='+', default=[256, 256], 
                       help='Network architecture')
    
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