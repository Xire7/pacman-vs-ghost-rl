"""Training utilities for Pac-Man RL agents."""

import time
import zipfile
import io
from pathlib import Path
import numpy as np
import torch
from typing import Callable, Dict

from stable_baselines3.common.callbacks import BaseCallback


# =============================================================================
# Learning Rate Schedules
# =============================================================================

def linear_schedule(initial: float, final: float = 1e-5) -> Callable[[float], float]:
    """Linear learning rate decay from initial to final."""
    def schedule(progress_remaining: float) -> float:
        return final + progress_remaining * (initial - final)
    return schedule


# =============================================================================
# Cross-Platform Model Loading
# =============================================================================

def load_maskable_ppo_cross_platform(
    model_path: Path,
    layout_name: str,
    device: str = 'auto',
):
    """Load MaskablePPO model trained on different platform (e.g., Windows -> Linux).
    
    This extracts just the neural network weights and loads them into a fresh model,
    avoiding pickle compatibility issues across Python versions/platforms.
    
    Args:
        model_path: Path to the .zip model file
        layout_name: Name of the layout (for creating dummy env)
        device: Device to load model on ('auto', 'cpu', 'cuda')
    
    Returns:
        MaskablePPO model with loaded weights
    """
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from gym_env import PacmanEnv
    
    print(f"Cross-platform loading from {model_path}...")
    
    # First, inspect the saved weights to determine architecture
    with zipfile.ZipFile(model_path, 'r') as z:
        if 'policy.pth' not in z.namelist():
            raise ValueError("No policy.pth found in model zip")
        
        with z.open('policy.pth') as f:
            buffer = io.BytesIO(f.read())
            policy_state = torch.load(buffer, map_location='cpu', weights_only=False)
    
    # Detect network architecture from saved weights
    # Look at mlp_extractor.policy_net.0.weight shape to determine hidden size
    if 'mlp_extractor.policy_net.0.weight' in policy_state:
        hidden_size = policy_state['mlp_extractor.policy_net.0.weight'].shape[0]
        print(f"Detected hidden layer size: {hidden_size}")
    else:
        hidden_size = 64  # Default
        print(f"Using default hidden layer size: {hidden_size}")
    
    # Create a fresh model with the matching architecture
    def make_env():
        env = PacmanEnv(layout_name=layout_name)
        env = ActionMasker(env, lambda e: e.action_masks())
        return Monitor(env)
    
    dummy_env = DummyVecEnv([make_env])
    model = MaskablePPO(
        "MlpPolicy", 
        dummy_env, 
        device=device,
        policy_kwargs={"net_arch": [hidden_size, hidden_size]}  # Match saved architecture
    )
    
    # Load the policy weights
    model.policy.load_state_dict(policy_state)
    print(f"Loaded policy weights via cross-platform method")
    
    # Try to load optimizer state (optional, for continued training)
    with zipfile.ZipFile(model_path, 'r') as z:
        if 'policy.optimizer.pth' in z.namelist():
            try:
                with z.open('policy.optimizer.pth') as f:
                    buffer = io.BytesIO(f.read())
                    optim_state = torch.load(buffer, map_location=device, weights_only=False)
                model.policy.optimizer.load_state_dict(optim_state)
                print(f"Loaded optimizer state")
            except Exception as e:
                print(f"Warning: Could not load optimizer state ({e}), will use fresh optimizer")
    
    dummy_env.close()
    print(f"Cross-platform load successful!")
    return model


# =============================================================================
# Callbacks
# =============================================================================

class WinRateCallback(BaseCallback):
    """Track win rate during Pac-Man training."""
    
    def __init__(self, print_freq: int = 50):
        super().__init__()
        self.wins = 0
        self.episodes = 0
        self.recent = []
        self.print_freq = print_freq
    
    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                self.episodes += 1
                info = self.locals.get('infos', [{}])[i]
                win = 1 if info.get('win', False) else 0
                self.wins += win
                self.recent.append(win)
                if len(self.recent) > 100:
                    self.recent.pop(0)
                
                if self.episodes % self.print_freq == 0:
                    rate = self.wins / self.episodes * 100
                    recent = sum(self.recent) / len(self.recent) * 100
                    print(f"\nEp {self.episodes} | Win: {rate:.1f}% | Recent: {recent:.1f}%")
        return True
    
    def get_win_rate(self) -> float:
        return self.wins / max(1, self.episodes)


class CatchRateCallback(BaseCallback):
    """Track catch rate during Ghost training."""
    
    def __init__(self, print_freq: int = 50):
        super().__init__()
        self.catches = 0
        self.episodes = 0
        self.recent = []
        self.print_freq = print_freq
    
    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                self.episodes += 1
                info = self.locals.get('infos', [{}])[i]
                catch = 1 if info.get('ghost_won', False) else 0
                self.catches += catch
                self.recent.append(catch)
                if len(self.recent) > 100:
                    self.recent.pop(0)
                
                if self.episodes % self.print_freq == 0:
                    rate = self.catches / self.episodes * 100
                    recent = sum(self.recent) / len(self.recent) * 100
                    print(f"\nEp {self.episodes} | Catch: {rate:.1f}% | Recent: {recent:.1f}%")
        return True
    
    def get_catch_rate(self) -> float:
        return self.catches / max(1, self.episodes)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_with_ghosts(
    pacman_model,
    layout_name: str,
    vec_normalize=None,
    ghost_models: Dict = None,
    episodes: int = 100,
) -> Dict:
    """Evaluate Pac-Man against ghosts.
    
    Args:
        pacman_model: Trained MaskablePPO model
        layout_name: Layout name
        vec_normalize: VecNormalize with obs_rms (optional)
        ghost_models: Dict of ghost DQN models {1: model, 2: model} or None for random
        episodes: Number of evaluation episodes
    
    Returns:
        Dict with 'win_rate', 'wins', 'mean_score', 'mean_steps'
    """
    from gym_env import PacmanEnv
    
    wins = 0
    scores = []
    steps_list = []
    
    for ep in range(episodes):
        if ghost_models:
            env = PacmanEnv(layout_name=layout_name, ghost_policies=ghost_models, max_steps=500)
        else:
            env = PacmanEnv(layout_name=layout_name, ghost_type='random', max_steps=500)
        
        obs, _ = env.reset()
        
        # Normalize observation if needed
        if vec_normalize is not None and hasattr(vec_normalize, 'obs_rms'):
            obs = (obs - vec_normalize.obs_rms.mean) / np.sqrt(vec_normalize.obs_rms.var + 1e-8)
            obs = np.clip(obs, -10.0, 10.0)
        
        done = False
        steps = 0
        
        while not done:
            action_masks = env.action_masks()
            action, _ = pacman_model.predict(obs, action_masks=action_masks)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
            
            if vec_normalize is not None and hasattr(vec_normalize, 'obs_rms'):
                obs = (obs - vec_normalize.obs_rms.mean) / np.sqrt(vec_normalize.obs_rms.var + 1e-8)
                obs = np.clip(obs, -10.0, 10.0)
        
        if info.get('win', False):
            wins += 1
        scores.append(info.get('score', 0))
        steps_list.append(steps)
        
        env.close()
        
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{episodes}: Win Rate so far: {wins/(ep+1)*100:.1f}%")
    
    return {
        'win_rate': wins / episodes,
        'wins': wins,
        'losses': episodes - wins,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'mean_steps': float(np.mean(steps_list)),
    }


def render_game(
    pacman_model,
    layout_name: str,
    vec_normalize=None,
    ghost_models: Dict = None,
    delay: float = 0.05,
) -> Dict:
    """Render a single game.
    
    Returns:
        Dict with 'win', 'score', 'steps'
    """
    from gym_env import PacmanEnv
    
    if ghost_models:
        env = PacmanEnv(layout_name=layout_name, ghost_policies=ghost_models,
                       max_steps=500, render_mode='human')
    else:
        env = PacmanEnv(layout_name=layout_name, ghost_type='random',
                       max_steps=500, render_mode='human')
    
    obs, _ = env.reset()
    
    if vec_normalize is not None and hasattr(vec_normalize, 'obs_rms'):
        obs = (obs - vec_normalize.obs_rms.mean) / np.sqrt(vec_normalize.obs_rms.var + 1e-8)
        obs = np.clip(obs, -10.0, 10.0)
    
    done = False
    steps = 0
    
    while not done:
        action_masks = env.action_masks()
        action, _ = pacman_model.predict(obs, action_masks=action_masks)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        steps += 1
        time.sleep(delay)
        
        if vec_normalize is not None and hasattr(vec_normalize, 'obs_rms'):
            obs = (obs - vec_normalize.obs_rms.mean) / np.sqrt(vec_normalize.obs_rms.var + 1e-8)
            obs = np.clip(obs, -10.0, 10.0)
    
    result = {
        'win': info.get('win', False),
        'score': info.get('score', 0),
        'steps': steps,
    }
    
    env.close()
    return result
