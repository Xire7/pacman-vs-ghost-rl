#!/usr/bin/env python3
"""
Iterative Freeze Training Script for Pac-Man vs Ghosts

This script implements adversarial RL with:
- PPO for Pac-Man (strategic, policy-based)
- Independent DQN for each Ghost (reactive, value-based)
- Sequential within-round training with rotation
- Iterative freezing between rounds

Training Structure:
- Odd rounds: Train ghosts sequentially (each sees latest teammate updates)
- Even rounds: Train Pac-Man against all updated ghosts
- Ghost training order rotates each round to reduce asymmetry

Usage:
    python train_adversarial.py --rounds 10 --layout mediumClassic
"""

import argparse
import os
import numpy as np
from datetime import datetime
from visualize_agents import record_round_video
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from ghost_agent import IndependentGhostEnv
from gym_env import PacmanEnv


def create_directories(base_dir="training_output"):
    """Create directory structure for saving models and logs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    dirs = {
        'models': os.path.join(run_dir, 'models'),
        'logs': os.path.join(run_dir, 'logs'),
        'checkpoints': os.path.join(run_dir, 'checkpoints')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def train_ghost_sequential(
    ghost_idx,
    round_num,
    pacman_model,
    ghost_models,
    layout_name,
    dirs,
    initial_timesteps=50000,
    refinement_timesteps=30000
):
    """
    Train a single ghost with DQN.
    
    Args:
        ghost_idx: Which ghost to train (1-4)
        round_num: Current round number
        pacman_model: Current Pac-Man policy (or None for random)
        ghost_models: Dict of current ghost policies
        layout_name: Map layout
        dirs: Dictionary of output directories
        initial_timesteps: Training steps for first round
        refinement_timesteps: Training steps for subsequent rounds
    
    Returns:
        Trained DQN model for this ghost
    """
    print(f"\n{'─'*60}")
    print(f"Training Ghost {ghost_idx}")
    print(f"{'─'*60}")
    
    # Get version number
    version = (round_num // 2) + 1
    
    # Create environment with latest versions of other ghosts
    num_ghosts = len(ghost_models)

    other_ghost_policies = {
        idx: ghost_models[idx] 
        for idx in range(1, num_ghosts + 1) if idx != ghost_idx and ghost_models[idx] is not None
    }
    
    env = IndependentGhostEnv(
        ghost_index=ghost_idx,
        layout_name=layout_name,
        pacman_policy=pacman_model,
        other_ghost_policies=other_ghost_policies,
        render_mode=None,
        max_steps=500
    )
    
    # Load previous version or create new model
    prev_version = version - 1
    prev_model_path = os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{prev_version}")
    
    if os.path.exists(prev_model_path + ".zip"):
        print(f"Loading Ghost {ghost_idx} v{prev_version} for continued training...")
        model = DQN.load(prev_model_path, env=env)
        timesteps = refinement_timesteps
        print(f"Refining for {timesteps:,} timesteps")
    else:
        print(f"Creating new DQN model for Ghost {ghost_idx}...")
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=os.path.join(dirs['logs'], f"ghost_{ghost_idx}")
        )
        timesteps = initial_timesteps
        print(f"Training from scratch for {timesteps:,} timesteps")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(dirs['checkpoints'], f"ghost_{ghost_idx}_v{version}"),
        name_prefix=f"ghost_{ghost_idx}_v{version}",
        save_replay_buffer=False
    )
    
    # Train
    print(f"Opponents: Pac-Man v{version-1}, Other Ghosts v{version-1} (latest)")
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model_path = os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{version}")
    model.save(model_path)
    print(f"✓ Ghost {ghost_idx} v{version} saved to {model_path}")
    
    env.close()
    
    return model


def train_pacman(
    round_num,
    pacman_model,
    ghost_models,
    layout_name,
    dirs,
    initial_timesteps=100000,
    refinement_timesteps=50000
):
    """
    Train Pac-Man with PPO against current ghost policies.
    
    Args:
        round_num: Current round number
        pacman_model: Previous Pac-Man policy (or None)
        ghost_models: Dict of current ghost policies
        layout_name: Map layout
        dirs: Dictionary of output directories
        initial_timesteps: Training steps for first round
        refinement_timesteps: Training steps for subsequent rounds
    
    Returns:
        Trained PPO model for Pac-Man
    """
    print(f"\n{'═'*60}")
    print(f"Training Pac-Man")
    print(f"{'═'*60}")
    
    version = round_num // 2
    
    # Create Pac-Man environment with trained ghosts
    env = PacmanEnv(
        layout_name=layout_name,
        ghost_policies=ghost_models,
        max_steps=500,
        render_mode=None,
        reward_shaping=False
    )
    
    # Load previous version or create new model
    prev_version = version - 1
    prev_model_path = os.path.join(dirs['models'], f"pacman_v{prev_version}")
    
    if os.path.exists(prev_model_path + ".zip"):
        print(f"Loading Pac-Man v{prev_version} for continued training...")
        model = PPO.load(prev_model_path, env=env)
        timesteps = refinement_timesteps
        print(f"Refining for {timesteps:,} timesteps")
    else:
        print(f"Creating new PPO model for Pac-Man...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=os.path.join(dirs['logs'], "pacman")
        )
        timesteps = initial_timesteps
        print(f"Training from scratch for {timesteps:,} timesteps")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(dirs['checkpoints'], f"pacman_v{version}"),
        name_prefix=f"pacman_v{version}",
        save_replay_buffer=False
    )
    
    # Train
    print(f"Opponents: Ghosts v{version}")
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model_path = os.path.join(dirs['models'], f"pacman_v{version}")
    model.save(model_path)
    print(f"✓ Pac-Man v{version} saved to {model_path}")
    
    env.close()
    
    return model


def evaluate_matchup(pacman_model, ghost_models, layout_name, n_episodes=20):
    """
    Evaluate current Pac-Man vs Ghosts matchup.
    
    Returns:
        dict: Statistics including win rates, scores, etc.
    """
    print(f"\n{'─'*60}")
    print(f"Evaluating Current Matchup ({n_episodes} episodes)")
    print(f"{'─'*60}")
    
    env = PacmanEnv(
        layout_name=layout_name,
        ghost_policies=ghost_models,
        max_steps=500,
        render_mode=None,
        reward_shaping=False
    )
    
    pacman_wins = 0
    ghost_wins = 0
    timeouts = 0
    scores = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = pacman_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        if info.get('win', False):
            pacman_wins += 1
            result = "🟡 Pac-Man"
        elif info.get('lose', False):
            ghost_wins += 1
            result = "👻 Ghosts"
        else:
            timeouts += 1
            result = "⏱️  Timeout"
        
        scores.append(info.get('raw_score', 0))
        episode_lengths.append(steps)
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode+1}/{n_episodes}: {result} | Score: {scores[-1]:.0f} | Steps: {steps}")
    
    env.close()
    
    stats = {
        'pacman_wins': pacman_wins,
        'ghost_wins': ghost_wins,
        'timeouts': timeouts,
        'pacman_win_rate': pacman_wins / n_episodes,
        'ghost_win_rate': ghost_wins / n_episodes,
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'avg_episode_length': np.mean(episode_lengths)
    }
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Pac-Man Wins:  {pacman_wins}/{n_episodes} ({100*stats['pacman_win_rate']:.1f}%)")
    print(f"  Ghost Wins:    {ghost_wins}/{n_episodes} ({100*stats['ghost_win_rate']:.1f}%)")
    print(f"  Timeouts:      {timeouts}/{n_episodes}")
    print(f"  Avg Score:     {stats['avg_score']:.1f} ± {stats['std_score']:.1f}")
    print(f"  Avg Length:    {stats['avg_episode_length']:.1f} steps")
    print(f"{'='*60}")
    
    return stats


def train_adversarial_rl(
    num_rounds=10,
    layout_name='mediumClassic',
    num_ghosts=4,
    ghost_initial_timesteps=50000,
    ghost_refinement_timesteps=30000,
    pacman_initial_timesteps=100000,
    pacman_refinement_timesteps=50000,
    eval_frequency=2,
    eval_episodes=20
):
    """
    Main adversarial RL training loop with sequential ghost updates and rotation.
    
    Args:
        num_rounds: Total training rounds (odd=ghosts, even=pacman)
        layout_name: Pac-Man map layout
        num_ghosts: Number of ghosts to train (1-4)
        ghost_initial_timesteps: Training steps for ghost initial training
        ghost_refinement_timesteps: Training steps for ghost refinement
        pacman_initial_timesteps: Training steps for Pac-Man initial training
        pacman_refinement_timesteps: Training steps for Pac-Man refinement
        eval_frequency: Evaluate every N rounds
        eval_episodes: Number of episodes for evaluation
    
    Returns:
        tuple: (final_pacman_model, final_ghost_models, training_history)
    """
    print(f"\n{'#'*60}")
    print(f"# ADVERSARIAL RL TRAINING")
    print(f"#")
    print(f"# Layout: {layout_name}")
    print(f"# Rounds: {num_rounds}")
    print(f"# Ghosts: {num_ghosts}")
    print(f"#")
    print(f"# Strategy:")
    print(f"#   - Pac-Man: PPO (policy-based, strategic)")
    print(f"#   - Ghosts:  DQN (value-based, aggressive)")
    print(f"#   - Sequential training with rotation")


    temp_env = PacmanEnv(layout_name=layout_name, render_mode=None)
    temp_state, _ = temp_env.reset()
    actual_num_ghosts = temp_env.game_state.getNumAgents() - 1
    temp_env.close()

    if num_ghosts > actual_num_ghosts:
        print(f"Warning: Layout '{layout_name}' has only {actual_num_ghosts} ghosts. "
              f"Adjusting num_ghosts to {actual_num_ghosts}.")
        num_ghosts = actual_num_ghosts
    
    print(f"{'#'*60}\n")

    # Create output directories
    dirs = create_directories()
    print(f"Output directory: {os.path.dirname(dirs['models'])}\n")
    
    # Initialize models (None = random policy)
    pacman_model = None
    ghost_models = {i: None for i in range(1, num_ghosts + 1)}
    
    # Training history
    history = {
        'rounds': [],
        'evaluations': []
    }
    
    # Main training loop
    for round_num in range(1, num_rounds + 1):
        print(f"\n\n{'#'*60}")
        print(f"# ROUND {round_num}/{num_rounds}")
        print(f"{'#'*60}")
        
        if round_num % 2 == 1:
            # ========== ODD ROUNDS: Train Ghosts Sequentially ==========
            version = (round_num // 2) + 1
            print(f"\nPhase: Ghost Training (→ v{version})")
            print(f"   Target: Train ghosts against Pac-Man v{version-1}")
            
            # Determine training order with rotation
            base_order = list(range(1, num_ghosts + 1))
            rotation = (round_num - 1) // 2  # Rotate based on ghost training round
            ghost_order = base_order[rotation % num_ghosts:] + base_order[:rotation % num_ghosts]
            
            print(f"   Training order: {ghost_order} (rotated by {rotation % num_ghosts})")
            
            # Train each ghost sequentially
            for ghost_idx in ghost_order:
                ghost_models[ghost_idx] = train_ghost_sequential(
                    ghost_idx=ghost_idx,
                    round_num=round_num,
                    pacman_model=pacman_model,
                    ghost_models=ghost_models,
                    layout_name=layout_name,
                    dirs=dirs,
                    initial_timesteps=ghost_initial_timesteps,
                    refinement_timesteps=ghost_refinement_timesteps
                )
                
                print(f"   ✓ Ghost {ghost_idx} updated to v{version}")
                print(f"   → Next ghost will train against this new policy\n")
            
            print(f"\n✅ All ghosts trained to v{version}")
        
        else:
            # ========== EVEN ROUNDS: Train Pac-Man ==========
            version = round_num // 2
            print(f"\nPhase: Pac-Man Training (→ v{version})")
            print(f"   Target: Train Pac-Man against Ghosts v{version}")
            
            pacman_model = train_pacman(
                round_num=round_num,
                pacman_model=pacman_model,
                ghost_models=ghost_models,
                layout_name=layout_name,
                dirs=dirs,
                initial_timesteps=pacman_initial_timesteps,
                refinement_timesteps=pacman_refinement_timesteps
            )
            
            print(f"\nPac-Man trained to v{version}")

            record_round_video(
                round_num=round_num,
                pacman_model=pacman_model,
                ghost_models=ghost_models,
                layout_name=layout_name,
                dirs=dirs
            ) # After pacman training, we get a demo video for the round on this pacman v the ghost
        
        # Evaluate periodically
        if round_num % eval_frequency == 0 and pacman_model is not None:
            stats = evaluate_matchup(
                pacman_model=pacman_model,
                ghost_models=ghost_models,
                layout_name=layout_name,
                n_episodes=eval_episodes
            )
            
            history['evaluations'].append({
                'round': round_num,
                'pacman_version': round_num // 2,
                'ghost_version': (round_num // 2) + (1 if round_num % 2 == 1 else 0),
                'stats': stats
            })
        
        history['rounds'].append(round_num)
    
    print(f"\n\n{'#'*60}")
    print(f"# TRAINING COMPLETE!")
    print(f"{'#'*60}")
    print(f"Final versions:")
    print(f"  Pac-Man: v{num_rounds // 2}")
    print(f"  Ghosts:  v{(num_rounds // 2) + (1 if num_rounds % 2 == 1 else 0)}")
    print(f"\nModels saved to: {dirs['models']}")
    print(f"Logs saved to:   {dirs['logs']}")
    
    return pacman_model, ghost_models, history


def main():
    parser = argparse.ArgumentParser(
        description='Train Pac-Man vs Ghosts with Adversarial RL'
    )
    
    # Training parameters
    parser.add_argument('--rounds', type=int, default=10,
                       help='Number of training rounds (default: 10)')
    parser.add_argument('--layout', type=str, default='mediumClassic',
                       choices=['smallGrid', 'mediumGrid', 'mediumClassic', 
                               'trickyClassic', 'testClassic'],
                       help='Map layout (default: mediumClassic)')
    parser.add_argument('--num-ghosts', type=int, default=4,
                       help='Number of ghosts to train (1-4, default: 4)')
    
    # Ghost training parameters
    parser.add_argument('--ghost-initial-steps', type=int, default=50000,
                       help='Ghost initial training timesteps (default: 50000)')
    parser.add_argument('--ghost-refine-steps', type=int, default=30000,
                       help='Ghost refinement timesteps (default: 30000)')
    
    # Pac-Man training parameters
    parser.add_argument('--pacman-initial-steps', type=int, default=100000,
                       help='Pac-Man initial training timesteps (default: 100000)')
    parser.add_argument('--pacman-refine-steps', type=int, default=50000,
                       help='Pac-Man refinement timesteps (default: 50000)')
    
    # Evaluation parameters
    parser.add_argument('--eval-freq', type=int, default=2,
                       help='Evaluate every N rounds (default: 2)')
    parser.add_argument('--eval-episodes', type=int, default=20,
                       help='Episodes per evaluation (default: 20)')
    
    args = parser.parse_args()
    
    # Run training
    pacman_model, ghost_models, history = train_adversarial_rl(
        num_rounds=args.rounds,
        layout_name=args.layout,
        num_ghosts=args.num_ghosts,
        ghost_initial_timesteps=args.ghost_initial_steps,
        ghost_refinement_timesteps=args.ghost_refine_steps,
        pacman_initial_timesteps=args.pacman_initial_steps,
        pacman_refinement_timesteps=args.pacman_refine_steps,
        eval_frequency=args.eval_freq,
        eval_episodes=args.eval_episodes
    )
    
    # Final evaluation
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION")
    print(f"{'='*60}")
    
    final_stats = evaluate_matchup(
        pacman_model=pacman_model,
        ghost_models=ghost_models,
        layout_name=args.layout,
        n_episodes=100  # More episodes for final eval
    )
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()