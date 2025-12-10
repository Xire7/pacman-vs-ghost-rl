"""
Mixed Adversarial Training for Pac-Man vs Ghosts

IMPROVED VERSION with Ghost Pretraining Phase:
- Phase 0: Ghosts train against RANDOM Pac-Man (learn basic strategies)
- Phase 1+: Alternating training on random and trained opponents

This prevents the "smart teacher" problem where ghosts can't learn from a 
Pac-Man that's too good. Ghosts first develop basic strategies (chasing, 
trapping, coordinating) against random Pac-Man before facing the trained one.
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


def pretrain_ghost_against_random(ghost_idx, layout, dirs, timesteps, num_ghosts):
    """
    Pretrain a ghost against random Pac-Man to learn basic strategies.
    
    This is the KEY addition: ghosts learn fundamentals (chasing, trapping, 
    coordination) against a random opponent before facing the smart Pac-Man.
    
    Args:
        ghost_idx: Which ghost to train
        layout: Map layout
        dirs: Output directories
        timesteps: Training timesteps
        num_ghosts: Total number of ghosts
    
    Returns:
        Trained DQN model
    """
    print(f"\n  Pretraining Ghost {ghost_idx} vs Random Pac-Man...")
    
    # Create environment with random Pac-Man (pacman_policy=None)
    env = IndependentGhostEnv(
        ghost_index=ghost_idx,
        layout_name=layout,
        pacman_policy=None,  # Random Pac-Man - KEY: easier opponent
        other_ghost_policies=None,  # Random other ghosts
        max_steps=500
    )
    
    tb_log_dir = os.path.join(dirs['logs'], f"ghost_{ghost_idx}_pretrain_v0")  # FIX: Unique directory
    
    # Optimized DQN hyperparameters for ghost learning
    # Key improvements: lower LR for stability, larger buffer for diversity,
    # longer exploration for discovering trapping strategies
    policy_kwargs = dict(
        net_arch=[256, 128, 64],  # Deeper network for complex coordination
    )
    
    model = DQN(
        "MlpPolicy", env,
        learning_rate=5e-4,         # Lower for stable convergence
        buffer_size=100000,         # 2x larger for experience diversity
        learning_starts=2000,       # More initial exploration
        batch_size=128,             # Larger batches = more stable gradients
        gamma=0.99,                 # Standard discount factor
        target_update_interval=2000,# Less frequent updates = more stability
        exploration_fraction=0.5,   # Longer exploration phase (50% of training)
        exploration_final_eps=0.05, # Minimum 5% random exploration
        train_freq=4,               # Update every 4 steps
        gradient_steps=1,           # One gradient update per training step
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=tb_log_dir
    )
    
    print(f"    Training for {timesteps:,} timesteps vs random Pac-Man...")
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    # Save pretrained model as v0
    model.save(os.path.join(dirs['models'], f"ghost_{ghost_idx}_v0"))
    print(f"    ✓ Ghost {ghost_idx} pretrained (saved as v0)")
    
    env.close()
    return model


def train_ghost(ghost_idx, pacman_model, ghost_models, layout, dirs, timesteps, version, vecnorm_stats=None):
    """
    Train a ghost using DQN against trained Pac-Man.
    
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
        max_steps=500,
        vecnorm_path=vecnorm_stats  # Pass VecNormalize path to ghost env
    )
    
    prev_path = os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{version-1}.zip")
    
    # FIX: Version-specific TensorBoard directory (prevents data overlap)
    tb_log_dir = os.path.join(dirs['logs'], f"ghost_{ghost_idx}_v{version}")
    
    if os.path.exists(prev_path) and version > 0:
        print(f"    Loading Ghost {ghost_idx} v{version-1} for refinement...")
        model = DQN.load(prev_path, env=env)
        
        # FIX: Reset timestep counter for clean TensorBoard logging
        model.num_timesteps = 0
        model._num_timesteps_at_start = 0
        
        # FIX: Assign new TensorBoard directory
        model.tensorboard_log = tb_log_dir
        
        # More conservative updates when refining to avoid catastrophic forgetting
        model.learning_rate = 3e-4  # Even lower LR for fine-tuning
        model.exploration_rate = 0.10  # Keep some exploration during refinement
        print(f"    Refining for {timesteps:,} timesteps (lr=3e-4, ε=0.10)")
        print(f"    TensorBoard: {tb_log_dir}")
    else:
        print(f"    Creating new DQN for Ghost {ghost_idx}...")
        print(f"    TensorBoard: {tb_log_dir}")
        
        policy_kwargs = dict(
            net_arch=[256, 128, 64],  # Deeper network for complex strategies
        )
        
        model = DQN(
            "MlpPolicy", env,
            learning_rate=5e-4,         # Optimized for ghost learning
            buffer_size=100000,         # Large buffer for experience diversity
            learning_starts=2000,       # More initial random exploration
            batch_size=128,             # Stable gradient estimates
            gamma=0.99,                 # Standard discount
            target_update_interval=2000,# Stable target network
            exploration_fraction=0.5,   # Long exploration phase
            exploration_final_eps=0.05, # Minimum exploration
            train_freq=4,
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=tb_log_dir
        )
        print(f"    Training for {timesteps:,} timesteps")
    
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
    
    # FIX: Version-specific TensorBoard directory
    log_dir = os.path.join(dirs['logs'], f"pacman_v{version}")
    
    # Phase 1: Train on random ghosts
    print(f"    Phase 1: {random_steps:,} steps vs random ghosts ({random_ratio*100:.0f}%)")
    print(f"    TensorBoard: {log_dir}")
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
    model = MaskablePPO.load(pacman_path, env=env_rand)
    
    # FIX: Reset timestep counter for clean TensorBoard logging
    model.num_timesteps = 0
    model._num_timesteps_at_start = 0
    
    # FIX: Assign new TensorBoard directory
    model.tensorboard_log = log_dir
    
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
    model2 = MaskablePPO.load(pacman_path, env=env_trained)
    model2.policy.load_state_dict(model.policy.state_dict())
    
    # FIX: Reset timesteps but use reset_num_timesteps=False to continue from Phase 1
    # This makes Phase 1 and Phase 2 appear as one continuous training run in TensorBoard
    model2.num_timesteps = model.num_timesteps  # Continue from Phase 1's final timestep
    model2.tensorboard_log = log_dir  # Same directory as Phase 1
    
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
    parser = argparse.ArgumentParser(description='Mixed Adversarial Training with Ghost Pretraining')
    parser.add_argument('--rounds', type=int, default=10, help='Training rounds')
    parser.add_argument('--layout', type=str, default='mediumClassic')
    parser.add_argument('--pacman', type=str, required=True, help='Pretrained Pac-Man path (.zip file)')
    parser.add_argument('--ghost-steps', type=int, default=80000, 
                       help='Ghost training steps per round (recommended: 60k-100k)')
    parser.add_argument('--pacman-steps', type=int, default=80000, 
                       help='Pac-Man training steps per round (recommended: 60k-100k)')
    parser.add_argument('--ghost-pretrain-steps', type=int, default=150000, 
                       help='Ghost pretraining steps vs random Pac-Man (recommended: 150k-200k)')
    parser.add_argument('--skip-ghost-pretrain', action='store_true',
                       help='Skip ghost pretraining phase (not recommended)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"MIXED ADVERSARIAL TRAINING WITH GHOST PRETRAINING")
    print(f"  Layout: {args.layout}")
    print(f"  Rounds: {args.rounds}")
    if args.skip_ghost_pretrain:
        print(f"  Ghost Pretrain: DISABLED")
    else:
        print(f"  Ghost Pretrain: {args.ghost_pretrain_steps:,} steps vs random Pac-Man")
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
    print(f"Baseline Pac-Man vs random ghosts: {baseline*100:.1f}%")
    
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
    print(f"Number of ghosts: {num_ghosts}\n")
    
    ghost_models = {i: None for i in range(1, num_ghosts + 1)}
    
    # ========== GHOST PRETRAINING PHASE ==========
    if not args.skip_ghost_pretrain:
        print(f"\n{'='*60}")
        print(f"PHASE 0: GHOST PRETRAINING (NEW!)")
        print(f"{'='*60}")
        print(f"Goal: Ghosts learn basic strategies against RANDOM Pac-Man")
        print(f"Why: Prevents 'smart teacher' problem - ghosts need to learn")
        print(f"     fundamentals before facing the trained Pac-Man")
        print(f"\nThis is like how Pac-Man learned vs random ghosts first!\n")
        
        for ghost_idx in range(1, num_ghosts + 1):
            ghost_models[ghost_idx] = pretrain_ghost_against_random(
                ghost_idx=ghost_idx,
                layout=args.layout,
                dirs=dirs,
                timesteps=args.ghost_pretrain_steps,
                num_ghosts=num_ghosts
            )
        
        print(f"\n All ghosts pretrained against random Pac-Man (v0)")
        
        # Evaluate pretrained ghosts vs trained Pac-Man
        print(f"\n{'─'*60}")
        print(f"Checkpoint: Testing pretrained ghosts vs trained Pac-Man")
        print(f"{'─'*60}")
        wr_baseline = evaluate(pacman_model, args.layout, None, n=30, vecnorm_stats=vecnorm_stats)
        wr_after_pretrain = evaluate(pacman_model, args.layout, ghost_models, n=30, vecnorm_stats=vecnorm_stats)
        print(f"\nPac-Man win rates:")
        print(f"  vs random ghosts:     {wr_baseline*100:.1f}%")
        print(f"  vs pretrained ghosts: {wr_after_pretrain*100:.1f}%")
        
        if wr_after_pretrain < wr_baseline:
            improvement = ((wr_baseline - wr_after_pretrain) / wr_baseline) * 100
            print(f"\n✓ Ghosts are {improvement:.1f}% more challenging than random!")
            print(f"  Ghosts learned basic strategies (chasing, trapping, etc.)")
        else:
            print(f"\n⚠ Ghosts not yet better than random - may need more pretraining")
        
        print(f"\nReady to begin adversarial training!\n")
    else:
        print(f"\n⚠ SKIPPED ghost pretraining phase")
        print(f"Ghosts will start from scratch (this may slow convergence)\n")
    
    # ========== ADVERSARIAL TRAINING LOOP ==========
    print(f"\n{'='*60}")
    print(f"ADVERSARIAL TRAINING ROUNDS")
    print(f"{'='*60}\n")
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'─'*60}")
        print(f"ROUND {round_num}/{args.rounds}")
        print(f"{'─'*60}")
        
        # Train ghosts
        ghost_version = round_num
        print(f"\nPhase: Train Ghosts (→ v{ghost_version})")
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
        print(f"\nPhase: Train Pac-Man (→ v{pacman_version})")
        pacman_model, pacman_path, vecnorm_stats = train_pacman(
            pacman_path, args.layout, dirs, ghost_models,
            args.pacman_steps, pacman_version, vecnorm_stats=vecnorm_stats
        )
        
        # Evaluate
        print(f"\n{'─'*60}")
        print(f"Round {round_num} Evaluation")
        print(f"{'─'*60}")
        wr_random = evaluate(pacman_model, args.layout, vecnorm_stats=vecnorm_stats, n=30)
        wr_trained = evaluate(pacman_model, args.layout, ghost_models, n=30, vecnorm_stats=vecnorm_stats)
        
        print(f"\nPac-Man v{pacman_version} Performance:")
        print(f"  vs random ghosts:  {wr_random*100:.1f}%")
        print(f"  vs trained ghosts: {wr_trained*100:.1f}%")
        
        # Check for catastrophic forgetting
        if wr_random < 0.70:
            print(f"\n⚠ WARNING: Pac-Man struggling vs random ghosts!")
            print(f"  May need to increase random ghost training ratio")
        
        # Check ghost improvement
        if round_num > 1 and wr_trained > 0.90:
            print(f"\n⚠ Ghosts may need more training - Pac-Man dominance too high")
    
    # ========== FINAL EVALUATION ==========
    print(f"\n\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    
    final_random = evaluate(pacman_model, args.layout, n=100, vecnorm_stats=vecnorm_stats)
    final_trained = evaluate(pacman_model, args.layout, ghost_models, n=100, vecnorm_stats=vecnorm_stats)
    
    print(f"\nPac-Man Final Performance:")
    print(f"  vs Random Ghosts:  {final_random*100:.1f}% (100 episodes)")
    print(f"  vs Trained Ghosts: {final_trained*100:.1f}% (100 episodes)")
    
    print(f"\nImprovement Analysis:")
    print(f"  Baseline:  {baseline*100:.1f}%")
    print(f"  Final:     {final_trained*100:.1f}%")
    
    if final_trained < baseline:
        challenge_increase = ((baseline - final_trained) / baseline) * 100
        print(f"  Ghosts are {challenge_increase:.1f}% more challenging!")
    
    # Save best model
    pacman_model.save(os.path.join(dirs['models'], "pacman_best"))
    print(f"\n✓ Saved best Pac-Man: {dirs['models']}/pacman_best.zip")
    print(f"✓ VecNormalize stats: {vecnorm_stats}")
    
    # Training complete
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nView training progress:")
    print(f"  tensorboard --logdir={dirs['logs']}")
    print(f"  Then open: http://localhost:6006")
    print(f"\nAll models saved to: {dirs['models']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()