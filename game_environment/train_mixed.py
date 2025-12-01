"""
Mixed Adversarial Training for Pac-Man vs Ghosts

Training approach: Alternate between training phases
- Phase 1: Train on random ghosts (learn general skills)
- Phase 2: Train on trained ghosts (learn counter-strategies)

This prevents catastrophic forgetting while improving against smart opponents.
"""

import argparse
import os
import numpy as np
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
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


def load_pretrained_pacman(model_path):
    """
    Load pretrained Pac-Man model with VecNormalize if available.
    
    Returns:
        tuple: (model, vecnorm_path or None)
    """
    print(f"\nLoading pretrained Pac-Man from: {model_path}")
    
    # Check for VecNormalize file
    model_dir = os.path.dirname(model_path)
    vecnorm_path = os.path.join(model_dir, 'vecnormalize.pkl')

    # If not found and we're in a subdirectory (like "best/"), check parent
    if not os.path.exists(vecnorm_path):
        parent_dir = os.path.dirname(model_dir)
        parent_vecnorm = os.path.join(parent_dir, 'vecnormalize.pkl')
        if os.path.exists(parent_vecnorm):
            vecnorm_path = parent_vecnorm
    
    if os.path.exists(vecnorm_path):
        print(f"✓ Found VecNormalize stats: {vecnorm_path}")
        return model_path, vecnorm_path
    else:
        print(f"ℹ No VecNormalize found (model trained without --normalize)")
        return model_path, None


def evaluate(model, layout, ghost_models=None, n=50, vecnorm_stats=None):
    """
    Evaluate Pac-Man win rate.
    
    Args:
        model: Trained MaskablePPO model
        layout: Map layout name
        ghost_models: Dict of ghost models (or None for random)
        n: Number of episodes
        vecnorm_stats: Path to vecnormalize.pkl (or None)
    """
    wins = 0
    
    for _ in range(n):
        # Create environment
        if ghost_models:
            env = PacmanEnv(layout_name=layout, ghost_policies=ghost_models, max_steps=500)
        else:
            env = PacmanEnv(layout_name=layout, ghost_type='random', max_steps=500)
        
        # Wrap with ActionMasker
        env = ActionMasker(env, lambda e: e.action_masks())
        
        # Wrap in DummyVecEnv for VecNormalize compatibility
        env = DummyVecEnv([lambda: env])
        
        # Apply VecNormalize if available
        if vecnorm_stats:
            env = VecNormalize.load(vecnorm_stats, env)
            env.training = False
            env.norm_reward = False
        
        obs = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True, 
                                     action_masks=env.env_method('action_masks')[0])
            if isinstance(action, np.ndarray):
                action = int(action.item())
            obs, _, done, info = env.step([action])
            done = done[0]
        
        if info[0].get('win'):
            wins += 1
        
        env.close()
    
    return wins / n


def train_ghost(ghost_idx, pacman_model, ghost_models, layout, dirs, timesteps, version, vecnorm_stats=None):
    """
    Train a ghost using DQN.
    
    Args:
        vecnorm_stats: Path to VecNormalize stats for Pac-Man policy
    """
    print(f"\n  Training Ghost {ghost_idx} (v{version})...")
    
    num_ghosts = len(ghost_models)
    other_ghosts = {i: ghost_models[i] for i in range(1, num_ghosts + 1) 
                    if i != ghost_idx and ghost_models[i] is not None}
    
    # Create ghost training environment
    env = IndependentGhostEnv(
        ghost_index=ghost_idx,
        layout_name=layout,
        pacman_policy=pacman_model,
        other_ghost_policies=other_ghosts,
        max_steps=500
    )
    
    # NOTE: Ghost environment handles VecNormalize internally when calling Pac-Man policy
    # We need to pass vecnorm_stats to the environment
    if vecnorm_stats:
        # Modify IndependentGhostEnv to handle this (see below)
        env.vecnorm_stats = vecnorm_stats
    
    prev_path = os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{version-1}.zip")
    tb_log_dir = os.path.join(dirs['logs'], f"ghost_{ghost_idx}")
    
    if os.path.exists(prev_path) and version > 1:
        model = DQN.load(prev_path, env=env, tensorboard_log=tb_log_dir)
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
            verbose=0,
            tensorboard_log=tb_log_dir
        )
    
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{version}"))
    env.close()
    return model


def train_pacman(pacman_path, layout, dirs, ghost_models, timesteps, version, n_envs=8, vecnorm_stats=None):
    """
    Train Pac-Man using alternating random/trained ghost phases.
    
    Args:
        vecnorm_stats: Path to VecNormalize stats (will be updated during training)
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

    # Gradual shift: more random early, more trained later
    random_ratio = max(0.5, 0.9 - 0.05 * (version - 1))
    random_steps = int(timesteps * random_ratio)
    trained_steps = timesteps - random_steps
    
    log_dir = os.path.join(dirs['logs'], f"pacman")
    
    # Phase 1: Train on random ghosts
    print(f"    Phase 1: {random_steps:,} steps vs random ghosts ({random_ratio*100:.0f}%)")
    env_rand = DummyVecEnv([make_random_env() for _ in range(n_envs)])
    env_rand = VecMonitor(env_rand)
    
    # Apply VecNormalize
    if vecnorm_stats and os.path.exists(vecnorm_stats):
        print(f"    Loading VecNormalize stats from: {vecnorm_stats}")
        env_rand = VecNormalize.load(vecnorm_stats, env_rand)
        env_rand.training = True  # Update statistics during training
        env_rand.norm_reward = True
    else:
        print(f"    Creating new VecNormalize")
        env_rand = VecNormalize(env_rand, norm_obs=True, norm_reward=True, 
                                clip_obs=10.0, clip_reward=10.0)
    
    # Load model
    model = MaskablePPO.load(pacman_path, env=env_rand, tensorboard_log=log_dir)
    model.learning_rate = 1e-4
    model.clip_range = lambda _: 0.1
    model.learn(total_timesteps=random_steps, progress_bar=True)
    
    # Save updated VecNormalize stats
    vecnorm_save_path = os.path.join(dirs['models'], 'vecnormalize.pkl')
    env_rand.save(vecnorm_save_path)
    print(f"    Saved VecNormalize to: {vecnorm_save_path}")
    
    env_rand.close()
    
    # Phase 2: Train on trained ghosts
    print(f"    Phase 2: {trained_steps:,} steps vs trained ghosts ({(1-random_ratio)*100:.0f}%)")
    env_trained = DummyVecEnv([make_trained_env() for _ in range(n_envs)])
    env_trained = VecMonitor(env_trained)
    
    # Load the updated VecNormalize stats
    env_trained = VecNormalize.load(vecnorm_save_path, env_trained)
    env_trained.training = True
    env_trained.norm_reward = True
    
    # Transfer policy to new env
    model2 = MaskablePPO.load(pacman_path, env=env_trained, tensorboard_log=log_dir)
    model2.policy.load_state_dict(model.policy.state_dict())
    model2.learning_rate = 1e-4
    model2.clip_range = lambda _: 0.1
    model2.learn(total_timesteps=trained_steps, progress_bar=True, reset_num_timesteps=False)
    
    # Save final VecNormalize stats
    env_trained.save(vecnorm_save_path)
    env_trained.close()
    
    # Save model
    save_path = os.path.join(dirs['models'], f"pacman_v{version}")
    model2.save(save_path)
    
    return model2, save_path + ".zip", vecnorm_save_path


def main():
    parser = argparse.ArgumentParser(description='Mixed Adversarial Training')
    parser.add_argument('--rounds', type=int, default=10, help='Training rounds')
    parser.add_argument('--layout', type=str, default='mediumClassic')
    parser.add_argument('--pacman', type=str, required=True, help='Pretrained Pac-Man path (.zip file)')
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
    
    # Load pretrained Pac-Man with VecNormalize
    pacman_path, vecnorm_stats = load_pretrained_pacman(args.pacman)
    
    # Create a temporary env to load the model
    temp_env = DummyVecEnv([lambda: ActionMasker(
        PacmanEnv(layout_name=args.layout, ghost_type='random', max_steps=500),
        lambda e: e.action_masks()
    )])
    
    if vecnorm_stats:
        temp_env = VecNormalize.load(vecnorm_stats, temp_env)
        temp_env.training = False
        temp_env.norm_reward = False
    
    pacman_model = MaskablePPO.load(pacman_path, env=temp_env)
    temp_env.close()
    
    # Baseline evaluation
    baseline = evaluate(pacman_model, args.layout, vecnorm_stats=vecnorm_stats)
    print(f"Baseline vs random: {baseline*100:.1f}%")
    
    # Save as v0 (copy pretrained model and vecnorm to output dir)
    pacman_model.save(os.path.join(dirs['models'], "pacman_v0"))
    if vecnorm_stats:
        import shutil
        vecnorm_dest = os.path.join(dirs['models'], 'vecnormalize.pkl')
        shutil.copy(vecnorm_stats, vecnorm_dest)
        vecnorm_stats = vecnorm_dest  # Use the copied version from now on
        print(f"Copied VecNormalize to: {vecnorm_dest}")
    
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
        ghost_order = list(range(1, num_ghosts + 1))
        np.random.shuffle(ghost_order)
        for ghost_idx in ghost_order:
            ghost_models[ghost_idx] = train_ghost(
                ghost_idx, pacman_model, ghost_models,
                args.layout, dirs, args.ghost_steps, ghost_version,
                vecnorm_stats=vecnorm_stats
            )
        
        # Train Pac-Man
        pacman_version = round_num
        print(f"\nPhase: Train Pac-Man (v{pacman_version})")
        pacman_model, pacman_path, vecnorm_stats = train_pacman(
            pacman_path, args.layout, dirs, ghost_models,
            args.pacman_steps, pacman_version, vecnorm_stats=vecnorm_stats
        )
        
        # Evaluate
        wr_random = evaluate(pacman_model, args.layout, vecnorm_stats=vecnorm_stats)
        wr_trained = evaluate(pacman_model, args.layout, ghost_models, n=30, vecnorm_stats=vecnorm_stats)
        print(f"\n  Results: {wr_random*100:.1f}% vs random, {wr_trained*100:.1f}% vs trained")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    
    final_random = evaluate(pacman_model, args.layout, n=100, vecnorm_stats=vecnorm_stats)
    final_trained = evaluate(pacman_model, args.layout, ghost_models, n=50, vecnorm_stats=vecnorm_stats)
    
    print(f"  vs Random:  {final_random*100:.1f}%")
    print(f"  vs Trained: {final_trained*100:.1f}%")
    
    pacman_model.save(os.path.join(dirs['models'], "pacman_best"))
    print(f"\nSaved: {dirs['models']}/pacman_best.zip")
    print(f"VecNormalize: {vecnorm_stats}")
    
    print(f"\n{'='*60}")
    print(f"VIEW TENSORBOARD")
    print(f"{'='*60}")
    print(f"Run this command:")
    print(f"  tensorboard --logdir={dirs['logs']}")
    print(f"\nThen open: http://localhost:6006")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()