"""
Evaluate and visualize curriculum-trained Pac-Man agents.

Usage:
    # Evaluate against random ghosts
    python evaluate_curriculum.py --model path/to/model --stage random_ghosts
    
    # Evaluate against adversarial ghosts
    python evaluate_curriculum.py --model path/to/model --stage adversarial --ghost-dir path/to/ghosts
    
    # Show visual gameplay
    python evaluate_curriculum.py --model path/to/model --render --episodes 5
"""

import argparse
import os
import sys
import numpy as np
from stable_baselines3 import PPO, DQN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'game_environment'))
from gym_env import make_pacman_env, PacmanEnv
import time


def evaluate_agent(
    model_path,
    layout_name,
    num_ghosts,
    ghost_type=None,
    ghost_models=None,
    n_episodes=20,
    render=False,
    frame_time=0.1,
    deterministic=True
):
    """
    Evaluate a trained Pac-Man agent.
    
    Args:
        model_path: Path to PPO model
        layout_name: Map layout
        num_ghosts: Number of ghosts
        ghost_type: 'random', 'directional', or None
        ghost_models: Dict of trained ghost models (for adversarial)
        n_episodes: Number of test episodes
        render: Whether to show visualization
        frame_time: Time between frames (if rendering)
        deterministic: Use deterministic policy
    
    Returns:
        dict: Evaluation statistics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING PAC-MAN AGENT")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Layout: {layout_name}")
    print(f"Ghosts: {num_ghosts} ({ghost_type if ghost_type else 'adversarial' if ghost_models else 'none'})")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*60}\n")
    
    # Create environment
    render_mode = 'human' if render else None
    
    if ghost_models:
        # Adversarial ghosts
        env = PacmanEnv(
            layout_name=layout_name,
            ghost_policies=ghost_models,
            max_steps=500,
            render_mode=render_mode,
            reward_shaping=False
        )
    elif num_ghosts == 0:
        # No ghosts
        env = PacmanEnv(
            layout_name=layout_name,
            ghost_agents=[],
            max_steps=500,
            render_mode=render_mode,
            reward_shaping=True
        )
    else:
        # Scripted ghosts
        env = make_pacman_env(
            layout_name=layout_name,
            ghost_type=ghost_type,
            num_ghosts=num_ghosts,
            max_steps=500,
            render_mode=render_mode,
            reward_shaping=True
        )
    
    # Load model
    if model_path.endswith('.zip'):
        model_path = model_path[:-4]
    model = PPO.load(model_path)
    
    # Run episodes
    wins = 0
    losses = 0
    timeouts = 0
    rewards = []
    scores = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                env.render()
                time.sleep(frame_time)
        
        # Record results
        if info.get('win', False):
            wins += 1
            result = "WIN"
        elif info.get('lose', False):
            losses += 1
            result = "LOSS"
        else:
            timeouts += 1
            result = "TIMEOUT"
        
        rewards.append(episode_reward)
        scores.append(info.get('raw_score', 0))
        episode_lengths.append(steps)
        
        print(f"Episode {episode+1:2d}: {result:7s} | "
              f"Reward: {episode_reward:6.1f} | "
              f"Score: {scores[-1]:4.0f} | "
              f"Steps: {steps:3d}")
    
    env.close()
    
    # Calculate statistics
    stats = {
        'episodes': n_episodes,
        'wins': wins,
        'losses': losses,
        'timeouts': timeouts,
        'win_rate': wins / n_episodes,
        'loss_rate': losses / n_episodes,
        'timeout_rate': timeouts / n_episodes,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Episodes:      {stats['episodes']}")
    print(f"Wins:          {stats['wins']:2d} ({100*stats['win_rate']:5.1f}%)")
    print(f"Losses:        {stats['losses']:2d} ({100*stats['loss_rate']:5.1f}%)")
    print(f"Timeouts:      {stats['timeouts']:2d} ({100*stats['timeout_rate']:5.1f}%)")
    print(f"")
    print(f"Mean Reward:   {stats['mean_reward']:6.1f} ± {stats['std_reward']:.1f}")
    print(f"Mean Score:    {stats['mean_score']:6.1f} ± {stats['std_score']:.1f}")
    print(f"Mean Length:   {stats['mean_length']:6.1f} ± {stats['std_length']:.1f} steps")
    print(f"{'='*60}\n")
    
    return stats


def compare_stages(curriculum_dir, layout_name, n_episodes=20):
    """
    Compare performance across curriculum stages.
    
    Args:
        curriculum_dir: Directory containing curriculum models
        layout_name: Map layout
        n_episodes: Episodes per stage
    """
    print(f"\n{'#'*60}")
    print("# CURRICULUM PROGRESSION ANALYSIS")
    print(f"{'#'*60}\n")
    
    models_dir = os.path.join(curriculum_dir, 'models')
    
    stages = [
        ('Stage 1: No Ghosts', 'pacman_no_ghosts_final', 0, 'random'),
        ('Stage 2: Random Ghosts', 'pacman_random_ghosts_final', 4, 'random'),
    ]
    
    results = []
    
    for stage_name, model_name, num_ghosts, ghost_type in stages:
        model_path = os.path.join(models_dir, model_name)
        
        if not os.path.exists(model_path + ".zip"):
            print(f"Skipping {stage_name}: model not found at {model_path}")
            continue
        
        print(f"\n{stage_name}")
        print(f"{'-'*60}")
        
        stats = evaluate_agent(
            model_path=model_path,
            layout_name=layout_name,
            num_ghosts=num_ghosts,
            ghost_type=ghost_type if num_ghosts > 0 else None,
            n_episodes=n_episodes,
            render=False,
            deterministic=True
        )
        
        results.append({
            'stage': stage_name,
            'win_rate': stats['win_rate'],
            'mean_reward': stats['mean_reward'],
            'mean_score': stats['mean_score']
        })
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("PROGRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Stage':<25} {'Win Rate':>10} {'Reward':>12} {'Score':>10}")
    print(f"{'-'*60}")
    
    for r in results:
        print(f"{r['stage']:<25} {100*r['win_rate']:>9.1f}% "
              f"{r['mean_reward']:>11.1f} {r['mean_score']:>10.1f}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate curriculum-trained Pac-Man agents'
    )
    
    # Evaluation mode
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'compare'],
                       help='Evaluation mode: single model or compare stages')
    
    # Model specification
    parser.add_argument('--model', type=str,
                       help='Path to Pac-Man model (for single mode)')
    parser.add_argument('--curriculum-dir', type=str,
                       help='Curriculum output directory (for compare mode)')
    
    # Environment settings
    parser.add_argument('--layout', type=str, default='mediumClassic',
                       help='Map layout')
    parser.add_argument('--stage', type=str, default='random_ghosts',
                       choices=['no_ghosts', 'random_ghosts', 'adversarial'],
                       help='Which stage to test against')
    parser.add_argument('--ghost-type', type=str, default='random',
                       choices=['random', 'directional'],
                       help='Ghost type (for scripted ghosts)')
    parser.add_argument('--num-ghosts', type=int, default=4,
                       help='Number of ghosts')
    
    # Adversarial ghost settings
    parser.add_argument('--ghost-dir', type=str,
                       help='Directory with trained ghost models (for adversarial stage)')
    parser.add_argument('--ghost-version', type=int, default=1,
                       help='Ghost version number (for adversarial stage)')
    
    # Evaluation parameters
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of evaluation episodes')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (overrides --deterministic)')
    
    # Visualization
    parser.add_argument('--render', action='store_true',
                       help='Show visual gameplay')
    parser.add_argument('--frame-time', type=float, default=0.1,
                       help='Time between frames (seconds)')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        if not args.curriculum_dir:
            parser.error("--curriculum-dir required for compare mode")
        
        compare_stages(
            curriculum_dir=args.curriculum_dir,
            layout_name=args.layout,
            n_episodes=args.episodes
        )
    
    else:  # single mode
        if not args.model:
            parser.error("--model required for single mode")
        
        # Determine ghost configuration
        if args.stage == 'no_ghosts':
            num_ghosts = 0
            ghost_type = None
            ghost_models = None
        elif args.stage == 'adversarial':
            if not args.ghost_dir:
                parser.error("--ghost-dir required for adversarial stage")
            
            # Load ghost models
            ghost_models = {}
            for i in range(1, args.num_ghosts + 1):
                ghost_path = os.path.join(
                    args.ghost_dir,
                    f"ghost_{i}_v{args.ghost_version}"
                )
                if os.path.exists(ghost_path + ".zip"):
                    ghost_models[i] = DQN.load(ghost_path)
                    print(f"Loaded Ghost {i} v{args.ghost_version}")
                else:
                    print(f"Warning: Ghost {i} not found at {ghost_path}")
            
            num_ghosts = len(ghost_models)
            ghost_type = None
        else:  # random_ghosts
            num_ghosts = args.num_ghosts
            ghost_type = args.ghost_type
            ghost_models = None
        
        deterministic = not args.stochastic
        
        evaluate_agent(
            model_path=args.model,
            layout_name=args.layout,
            num_ghosts=num_ghosts,
            ghost_type=ghost_type,
            ghost_models=ghost_models,
            n_episodes=args.episodes,
            render=args.render,
            frame_time=args.frame_time,
            deterministic=deterministic
        )


if __name__ == '__main__':
    main()