#!/usr/bin/env python3
"""
Optimized MaskablePPO training script for Pac-Man.

This script provides:
- Best-practice PPO hyperparameters for Pac-Man
- Proper learning rate scheduling  
- Comprehensive callbacks for tracking and saving
- Clean evaluation mode
- Reproducible training with seeds
"""

import argparse
import os
import numpy as np
import torch
from datetime import datetime
from collections import deque

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, BaseCallback
)

from training_utils import (
    create_vec_env,
    linear_schedule,
    run_evaluation,
)


# =============================================================================
# Optimized Hyperparameters
# =============================================================================

# These are tuned for Pac-Man's reward structure and episode length
DEFAULT_HYPERPARAMS = {
    # Rollout settings
    'n_steps': 512,         # Larger buffer for stable updates
    'batch_size': 128,      # Larger batches for better gradients
    'n_epochs': 10,         # Standard PPO epochs
    
    # Discount and GAE
    'gamma': 0.995,         # High discount for long-term planning
    'gae_lambda': 0.95,     # Standard GAE
    
    # PPO clipping
    'clip_range': 0.2,      # Standard clip range
    'max_grad_norm': 0.5,   # Gradient clipping
    
    # Exploration vs exploitation
    'ent_coef': 0.01,       # Entropy for exploration
    'target_kl': 0.02,      # Early stopping threshold
    
    # Value function
    'vf_coef': 0.5,         # Value loss weight
    
    # Network
    'net_arch': [256, 256], # Hidden layers
}


# =============================================================================
# Custom Callbacks
# =============================================================================

class EpisodeProgressCallback(BaseCallback):
    """Track episodes with progress bar and optional early stopping."""
    
    def __init__(
        self, 
        n_episodes: int,
        save_path: str = None,
        early_stop_patience: int = 0,  # 0 = disabled
        early_stop_delta: float = 0.02,  # 2% improvement threshold
        eval_window: int = 100,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.n_episodes = n_episodes
        self.save_path = save_path
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        self.eval_window = eval_window
        
        self.episode_count = 0
        self.win_count = 0
        self.pbar = None
        
        # For early stopping
        self.recent_wins = deque(maxlen=eval_window)
        self.best_win_rate = 0.0
        self.patience_counter = 0
        self.stopped_early = False
    
    def _on_training_start(self):
        from tqdm import tqdm
        self.pbar = tqdm(total=self.n_episodes, desc="Training", unit="ep")
    
    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_count += 1
                self.pbar.update(1)
                
                # Track wins
                win = 1 if info.get('win', False) else 0
                self.recent_wins.append(win)
                if win:
                    self.win_count += 1
                
                # Update progress bar
                if len(self.recent_wins) >= 50:
                    current_wr = np.mean(self.recent_wins)
                    self.pbar.set_postfix({
                        'wr': f'{current_wr*100:.1f}%',
                        'best': f'{self.best_win_rate*100:.1f}%'
                    })
                    
                    # Save best model
                    if self.save_path and current_wr > self.best_win_rate:
                        self.best_win_rate = current_wr
                        path = os.path.join(self.save_path, 'best_winrate')
                        self.model.save(path)
                
                # Check early stopping
                if self.early_stop_patience > 0 and len(self.recent_wins) >= self.eval_window:
                    if self.episode_count % self.eval_window == 0:
                        current_wr = np.mean(self.recent_wins)
                        
                        if current_wr > self.best_win_rate + self.early_stop_delta:
                            self.best_win_rate = current_wr
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1
                        
                        if self.patience_counter >= self.early_stop_patience:
                            self.stopped_early = True
                            self.pbar.set_description("Training (early stop)")
                            return False
                
                if self.episode_count >= self.n_episodes:
                    return False
        return True
    
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
        print(f"\nCompleted {self.episode_count} episodes | "
              f"Win rate: {self.best_win_rate*100:.1f}%")


# =============================================================================
# Training Function
# =============================================================================

def train(args):
    """Train a MaskablePPO agent with optimized settings."""
    
    # Set seeds for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_{args.layout}_{timestamp}"
    log_dir = os.path.join(args.log_dir, run_name)
    model_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Device selection (CPU is faster for small networks like ours)
    device = 'cuda' if args.gpu else 'cpu'
    
    # Load normalization stats if resuming
    norm_stats_path = None
    if args.resume and args.normalize:
        candidate = os.path.join(os.path.dirname(args.resume), 'vecnormalize.pkl')
        if os.path.exists(candidate):
            norm_stats_path = candidate
            print(f"Loading normalization stats: {norm_stats_path}")
    
    # Print configuration
    print(f"\n{'='*65}")
    print(f"PPO Training: {args.layout}")
    print(f"{'='*65}")
    print(f"Episodes: {args.episodes:,} | Envs: {args.num_envs} | Device: {device}")
    print(f"Seed: {args.seed or 'None'} | Normalize: {args.normalize}")
    print(f"\nHyperparameters:")
    print(f"  n_steps: {args.n_steps} | batch_size: {args.batch_size} | n_epochs: {args.n_epochs}")
    print(f"  gamma: {args.gamma} | gae_lambda: {args.gae_lambda} | clip_range: {args.clip_range}")
    print(f"  ent_coef: {args.ent_coef} | target_kl: {args.target_kl}")
    print(f"  network: {args.net_arch}")
    
    # Learning rate info
    if args.lr_decay:
        lr_info = f"{args.lr} → {args.lr_final} (linear decay)"
    else:
        lr_info = f"{args.lr} (constant)"
    print(f"  learning_rate: {lr_info}")
    print(f"{'='*65}\n")
    
    # Create environments
    env = create_vec_env(
        args.layout, args.ghost_type, args.num_envs, args.max_steps,
        normalize=args.normalize, norm_stats_path=norm_stats_path, training=True
    )
    
    eval_env = create_vec_env(
        args.layout, args.ghost_type, 1, args.max_steps,
        normalize=args.normalize, norm_stats_path=norm_stats_path, training=False
    )
    
    # Sync normalization if applicable
    if args.normalize and hasattr(env, 'obs_rms') and hasattr(eval_env, 'obs_rms'):
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms
    
    # Policy configuration
    policy_kwargs = {
        'net_arch': dict(pi=args.net_arch, vf=args.net_arch),
        'activation_fn': torch.nn.Tanh,
        'ortho_init': True,
    }
    
    # Learning rate schedule
    lr = linear_schedule(args.lr, args.lr_final) if args.lr_decay else args.lr
    
    # Create or load model
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = MaskablePPO.load(
            args.resume, env=env, 
            tensorboard_log=log_dir, 
            device=device
        )
        model.learning_rate = lr
        if args.lr_decay:
            model.lr_schedule = lr
    else:
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
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
            device=device,
            seed=args.seed,
        )
    
    # Setup callbacks
    # Checkpoint every ~1000 episodes worth of steps
    checkpoint_freq = max(1000 * 150 // args.num_envs, 1)
    eval_freq = max(500 * 150 // args.num_envs, 1)
    
    episode_callback = EpisodeProgressCallback(
        n_episodes=args.episodes,
        save_path=model_dir,
        early_stop_patience=10 if args.early_stop else 0,
        early_stop_delta=0.02,
        eval_window=100
    )
    
    callbacks = [
        episode_callback,
        CheckpointCallback(
            save_freq=checkpoint_freq, 
            save_path=model_dir, 
            name_prefix='checkpoint'
        ),
        MaskableEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, 'best'),
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=20,
            deterministic=True,
        ),
    ]
    
    # Train until episode callback stops us
    try:
        model.learn(
            total_timesteps=int(1e9),  # Large number, callback will stop us
            callback=CallbackList(callbacks),
            progress_bar=False  # Using our own progress bar
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    # Save final model
    final_path = os.path.join(model_dir, 'final_model')
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}.zip")
    
    if args.normalize and hasattr(env, 'save'):
        norm_path = os.path.join(model_dir, 'vecnormalize.pkl')
        env.save(norm_path)
        print(f"Normalization stats saved to: {norm_path}")
    
    print(f"\n{'='*65}")
    print(f"Training complete! Output: {model_dir}")
    print(f"{'='*65}\n")
    
    env.close()
    eval_env.close()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train or evaluate PPO Pac-Man agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode
    parser.add_argument('--eval', action='store_true', 
                        help='Run evaluation mode')
    
    # Environment
    parser.add_argument('--layout', default='mediumClassic',
                        help='Game layout name')
    parser.add_argument('--ghost-type', default='random', 
                        choices=['random', 'directional'],
                        help='Ghost AI type')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Max steps per episode')
    
    # Training duration
    parser.add_argument('--episodes', type=int, default=10_000,
                        help='Total training episodes')
    parser.add_argument('--num-envs', type=int, default=8,
                        help='Number of parallel environments')
    
    # Learning rate
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate')
    parser.add_argument('--lr-decay', action='store_true',
                        help='Enable linear LR decay')
    parser.add_argument('--lr-final', type=float, default=1e-5,
                        help='Final LR when using decay')
    
    # PPO hyperparameters
    hp = DEFAULT_HYPERPARAMS
    parser.add_argument('--n-steps', type=int, default=hp['n_steps'],
                        help='Steps per env before update')
    parser.add_argument('--batch-size', type=int, default=hp['batch_size'],
                        help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=hp['n_epochs'],
                        help='PPO epochs per update')
    parser.add_argument('--gamma', type=float, default=hp['gamma'],
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=hp['gae_lambda'],
                        help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=hp['clip_range'],
                        help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=hp['ent_coef'],
                        help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=hp['vf_coef'],
                        help='Value function coefficient')
    parser.add_argument('--target-kl', type=float, default=hp['target_kl'],
                        help='Target KL for early stopping')
    parser.add_argument('--max-grad-norm', type=float, default=hp['max_grad_norm'],
                        help='Max gradient norm')
    
    # Network
    parser.add_argument('--net-arch', type=int, nargs='+', default=hp['net_arch'],
                        help='Network hidden layers')
    
    # Training control
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (None for random)')
    parser.add_argument('--early-stop', action='store_true',
                        help='Enable early stopping when win rate plateaus')
    parser.add_argument('--normalize', action='store_true',
                        help='Use observation/reward normalization')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU (CPU is default, faster for small networks)')
    parser.add_argument('--resume', type=str,
                        help='Resume from model path')
    
    # Directories
    parser.add_argument('--log-dir', default='./logs',
                        help='Tensorboard log directory')
    parser.add_argument('--model-dir', default='./models',
                        help='Model save directory')
    
    # Evaluation mode arguments
    parser.add_argument('--model-path', type=str,
                        help='Model path for evaluation')
    parser.add_argument('--ghost1', type=str,
                        help='Ghost 1 model path (for adversarial eval)')
    parser.add_argument('--ghost2', type=str,
                        help='Ghost 2 model path (for adversarial eval)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (eval mode)')
    parser.add_argument('--render', action='store_true',
                        help='Render games during evaluation')
    
    args = parser.parse_args()
    
    # Handle None seed
    if args.seed == 0:
        args.seed = None
    
    if args.eval:
        if not args.model_path:
            parser.error("--model-path required for evaluation mode")
        # Use eval_episodes for evaluation mode
        args.episodes = args.eval_episodes
        run_evaluation(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
