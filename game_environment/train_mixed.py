"""
Mixed Adversarial Training for Pac-Man vs Ghosts

Training approach: Alternate between training phases
- Phase 1: Train on random ghosts (learn general skills)
- Phase 2: Train on trained ghosts (learn counter-strategies)

This prevents catastrophic forgetting while improving against smart opponents.
"""

import argparse
import os
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from ghost_agent import IndependentGhostEnv
from gym_env import PacmanEnv


def create_dirs(base_dir="training_output"):
    """Create output directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"mixed_{timestamp}")
    dirs = {
        'models': os.path.join(run_dir, 'models'),
        'logs': os.path.join(run_dir, 'logs'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return run_dir, dirs


def evaluate(model, layout, ghost_models=None, n=50):
    """Evaluate Pac-Man win rate."""
    wins = 0
    for _ in range(n):
        if ghost_models:
            env = PacmanEnv(layout_name=layout, ghost_policies=ghost_models, max_steps=500)
        else:
            env = PacmanEnv(layout_name=layout, ghost_type='random', max_steps=500)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, _, t, tr, info = env.step(action)
            done = t or tr
        if info.get('win'):
            wins += 1
        env.close()
    return wins / n


def train_ghost(ghost_idx, pacman_model, ghost_models, layout, dirs, timesteps, version):
    """Train a ghost using DQN."""
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
    if os.path.exists(prev_path) and version > 1:
        model = DQN.load(prev_path, env=env)
        model.learning_rate = 5e-4
    else:
        model = DQN(
            "MlpPolicy", env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            verbose=0
        )
    
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{version}"))
    env.close()
    return model


def train_pacman(pacman_path, layout, dirs, ghost_models, timesteps, version, n_envs=8):
    """Train Pac-Man using alternating random/trained ghost phases."""
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
    
    half_steps = timesteps // 2
    
    # Phase 1: Train on random ghosts
    print(f"    Phase 1: {half_steps:,} steps vs random ghosts...")
    env_rand = VecMonitor(DummyVecEnv([make_random_env() for _ in range(n_envs)]))
    model = MaskablePPO.load(pacman_path, env=env_rand)
    model.learning_rate = 1e-4
    model.clip_range = lambda _: 0.1
    model.learn(total_timesteps=half_steps, progress_bar=True)
    env_rand.close()
    
    # Phase 2: Train on trained ghosts
    print(f"    Phase 2: {half_steps:,} steps vs trained ghosts...")
    env_trained = VecMonitor(DummyVecEnv([make_trained_env() for _ in range(n_envs)]))
    
    # Transfer policy to new env
    model2 = MaskablePPO.load(pacman_path, env=env_trained)
    model2.policy.load_state_dict(model.policy.state_dict())
    model2.learning_rate = 1e-4
    model2.clip_range = lambda _: 0.1
    model2.learn(total_timesteps=half_steps, progress_bar=True, reset_num_timesteps=False)
    env_trained.close()
    
    # Save
    save_path = os.path.join(dirs['models'], f"pacman_v{version}")
    model2.save(save_path)
    return model2, save_path + ".zip"


def main():
    parser = argparse.ArgumentParser(description='Mixed Adversarial Training')
    parser.add_argument('--rounds', type=int, default=6, help='Training rounds')
    parser.add_argument('--layout', type=str, default='mediumClassic')
    parser.add_argument('--pacman', type=str, required=True, help='Pretrained Pac-Man path')
    parser.add_argument('--ghost-steps', type=int, default=40000)
    parser.add_argument('--pacman-steps', type=int, default=50000)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"MIXED ADVERSARIAL TRAINING")
    print(f"  Layout: {args.layout}")
    print(f"  Rounds: {args.rounds}")
    print(f"{'='*60}")
    
    run_dir, dirs = create_dirs()
    print(f"Output: {run_dir}\n")
    
    # Load pretrained Pac-Man
    print(f"Loading: {args.pacman}")
    pacman_model = MaskablePPO.load(args.pacman)
    pacman_path = args.pacman
    
    # Baseline
    baseline = evaluate(pacman_model, args.layout)
    print(f"Baseline vs random: {baseline*100:.1f}%")
    
    # Save as v0
    pacman_model.save(os.path.join(dirs['models'], "pacman_v0"))
    
    # Get ghost count
    temp_env = PacmanEnv(layout_name=args.layout)
    num_ghosts = temp_env.num_ghosts
    temp_env.close()
    print(f"Ghosts: {num_ghosts}")
    
    ghost_models = {i: None for i in range(1, num_ghosts + 1)}
    
    # Training loop
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'─'*60}")
        print(f"ROUND {round_num}/{args.rounds}")
        print(f"{'─'*60}")
        
        # Train ghosts
        ghost_version = round_num
        print(f"\nPhase: Train Ghosts (v{ghost_version})")
        for ghost_idx in range(1, num_ghosts + 1):
            ghost_models[ghost_idx] = train_ghost(
                ghost_idx, pacman_model, ghost_models,
                args.layout, dirs, args.ghost_steps, ghost_version
            )
        
        # Train Pac-Man
        pacman_version = round_num
        print(f"\nPhase: Train Pac-Man (v{pacman_version})")
        pacman_model, pacman_path = train_pacman(
            pacman_path, args.layout, dirs, ghost_models,
            args.pacman_steps, pacman_version
        )
        
        # Evaluate
        wr_random = evaluate(pacman_model, args.layout)
        wr_trained = evaluate(pacman_model, args.layout, ghost_models, n=30)
        print(f"\n  Results: {wr_random*100:.1f}% vs random, {wr_trained*100:.1f}% vs trained")
    
    # Final
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    
    final_random = evaluate(pacman_model, args.layout, n=100)
    final_trained = evaluate(pacman_model, args.layout, ghost_models, n=50)
    
    print(f"  vs Random:  {final_random*100:.1f}%")
    print(f"  vs Trained: {final_trained*100:.1f}%")
    
    pacman_model.save(os.path.join(dirs['models'], "pacman_best"))
    print(f"\nSaved: {dirs['models']}/pacman_best.zip")


if __name__ == '__main__':
    main()
