"""
Live Visualization for Trained Pac-Man Agents

This script lets you watch your trained Pac-Man agents play in real-time
against scripted or adversarial ghosts.

Usage:
    # Watch curriculum Stage 2 agent (vs random ghosts)
    python visualize_agents.py \
        --pacman-model curriculum_output/run_*/models/pacman_random_ghosts_final \
        --ghost-type random

    # Watch adversarial agent (vs learned ghosts)
    python visualize_agents.py \
        --pacman-model curriculum_output/run_*/models/pacman_adversarial_v2 \
        --ghost-dir curriculum_output/run_*/models \
        --ghost-version 2

    # Watch multiple games in sequence
    python visualize_agents.py \
        --pacman-model path/to/model \
        --episodes 10 \
        --speed 0.05

    # Compare two Pac-Man models side-by-side (sequential)
    python visualize_agents.py --compare \
        --model1 curriculum_output/run_*/models/pacman_random_ghosts_final \
        --model2 curriculum_output/run_*/models/pacman_adversarial_v2 \
        --episodes 5
"""

import argparse
import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO, DQN

# Add game_environment to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'game_environment'))

from gym_env import make_pacman_env, PacmanEnv
import ghostAgents


class GameVisualizer:
    """Handles visualization of trained agents."""
    
    def __init__(self, frame_time=0.1):
        self.frame_time = frame_time
        self.stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'timeouts': 0,
            'total_score': 0,
            'total_steps': 0
        }
    
    def play_game(
        self,
        pacman_model,
        layout_name,
        ghost_policies=None,
        ghost_type='random',
        num_ghosts=4,
        max_steps=500,
        deterministic=True,
        show_stats=True
    ):
        """
        Play a single game and visualize it.
        
        Args:
            pacman_model: Trained PPO model
            layout_name: Map layout
            ghost_policies: Dict of {ghost_idx: DQN_model} for adversarial ghosts
            ghost_type: 'random' or 'directional' (if not using ghost_policies)
            num_ghosts: Number of ghosts
            max_steps: Maximum steps per game
            deterministic: Use deterministic policy
            show_stats: Print game statistics
        
        Returns:
            dict: Game statistics
        """
        # Create environment with rendering
        if ghost_policies:
            env = PacmanEnv(
                layout_name=layout_name,
                ghost_policies=ghost_policies,
                max_steps=max_steps,
                render_mode='human',
                reward_shaping=False
            )
            ghost_desc = f"adversarial (v{len(ghost_policies)})"
        else:
            env = make_pacman_env(
                layout_name=layout_name,
                ghost_type=ghost_type,
                num_ghosts=num_ghosts,
                max_steps=max_steps,
                render_mode='human',
                reward_shaping=True
            )
            ghost_desc = ghost_type
        
        if show_stats:
            print(f"\n{'='*60}")
            print(f"Game {self.stats['games_played'] + 1}")
            print(f"Layout: {layout_name}, Ghosts: {num_ghosts} ({ghost_desc})")
            print(f"{'='*60}")
        
        # Run episode
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Get action from policy
            action, _ = pacman_model.predict(obs, deterministic=deterministic)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Render
            env.render()
            time.sleep(self.frame_time)
        
        # Determine result
        if info.get('win', False):
            result = "WIN"
            self.stats['wins'] += 1
        elif info.get('lose', False):
            result = "LOSS"
            self.stats['losses'] += 1
        else:
            result = "TIMEOUT"
            self.stats['timeouts'] += 1
        
        self.stats['games_played'] += 1
        self.stats['total_score'] += info.get('raw_score', 0)
        self.stats['total_steps'] += steps
        
        # Print game result
        if show_stats:
            print(f"\n{'='*60}")
            print(f"Result: {result}")
            print(f"Score: {info.get('raw_score', 0):.0f}")
            print(f"Steps: {steps}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"{'='*60}")
        
        env.close()
        
        return {
            'result': result,
            'score': info.get('raw_score', 0),
            'steps': steps,
            'reward': episode_reward
        }
    
    def print_summary(self):
        """Print summary statistics."""
        if self.stats['games_played'] == 0:
            return
        
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Games Played: {self.stats['games_played']}")
        print(f"Wins:         {self.stats['wins']} ({100*self.stats['wins']/self.stats['games_played']:.1f}%)")
        print(f"Losses:       {self.stats['losses']} ({100*self.stats['losses']/self.stats['games_played']:.1f}%)")
        print(f"Timeouts:     {self.stats['timeouts']} ({100*self.stats['timeouts']/self.stats['games_played']:.1f}%)")
        print(f"")
        print(f"Avg Score:    {self.stats['total_score']/self.stats['games_played']:.1f}")
        print(f"Avg Steps:    {self.stats['total_steps']/self.stats['games_played']:.1f}")
        print(f"{'='*60}\n")


def load_ghost_policies(ghost_dir, ghost_version, num_ghosts):
    """
    Load trained ghost policies.
    
    Args:
        ghost_dir: Directory containing ghost models
        ghost_version: Version number
        num_ghosts: Number of ghosts to load
    
    Returns:
        dict: {ghost_idx: DQN_model}
    """
    ghost_policies = {}
    
    for i in range(1, num_ghosts + 1):
        ghost_path = os.path.join(ghost_dir, f"ghost_{i}_v{ghost_version}")
        
        if os.path.exists(ghost_path + ".zip"):
            try:
                ghost_policies[i] = DQN.load(ghost_path)
                print(f"✓ Loaded Ghost {i} v{ghost_version}")
            except Exception as e:
                print(f"✗ Failed to load Ghost {i}: {e}")
        else:
            print(f"⚠  Ghost {i} not found at {ghost_path}")
    
    if not ghost_policies:
        print("Warning: No ghost policies loaded")
        return None
    
    return ghost_policies


def visualize_single_model(args):
    """Visualize a single trained model."""
    print(f"\n{'#'*60}")
    print("# PAC-MAN AGENT VISUALIZATION")
    print(f"{'#'*60}")
    print(f"Model: {args.pacman_model}")
    print(f"Layout: {args.layout}")
    print(f"Episodes: {args.episodes}")
    print(f"Speed: {args.speed}s per frame")
    print(f"{'#'*60}\n")
    
    # Load Pac-Man model
    print("Loading Pac-Man model...")
    if args.pacman_model.endswith('.zip'):
        args.pacman_model = args.pacman_model[:-4]
    
    # Try direct path first
    model_path = args.pacman_model
    if not os.path.exists(model_path + ".zip"):
        # Try looking in best_model subdirectory
        best_path = os.path.join(model_path + "_best", "best_model")
        if os.path.exists(best_path + ".zip"):
            model_path = best_path
            print(f"Found best model at: {best_path}")
    
    try:
        pacman_model = PPO.load(model_path)
        print(f"✓ Loaded Pac-Man model from {model_path}")
    except Exception as e:
        print(f"✗ Failed to load Pac-Man model: {e}")
        return
    
    # Load ghost policies if specified
    ghost_policies = None
    if args.ghost_dir and args.ghost_version is not None:
        print("\nLoading adversarial ghost models...")
        ghost_policies = load_ghost_policies(
            args.ghost_dir,
            args.ghost_version,
            args.num_ghosts
        )
    
    # Determine ghost configuration
    if ghost_policies:
        ghost_type = None
        num_ghosts = len(ghost_policies)
        print(f"\nUsing {num_ghosts} adversarial ghost(s)")
    else:
        ghost_type = args.ghost_type
        num_ghosts = args.num_ghosts
        print(f"\nUsing {num_ghosts} {ghost_type} ghost(s)")
    
    # Create visualizer
    visualizer = GameVisualizer(frame_time=args.speed)
    
    # Play games
    print(f"\nStarting visualization ({args.episodes} episode(s))...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for episode in range(args.episodes):
            visualizer.play_game(
                pacman_model=pacman_model,
                layout_name=args.layout,
                ghost_policies=ghost_policies,
                ghost_type=ghost_type,
                num_ghosts=num_ghosts,
                max_steps=args.max_steps,
                deterministic=not args.stochastic,
                show_stats=True
            )
            
            # Pause between games
            if episode < args.episodes - 1:
                print(f"\nStarting next game in 2 seconds...")
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user.")
    
    # Print summary
    visualizer.print_summary()


def compare_models(args):
    """Compare two models side-by-side (sequentially)."""
    print(f"\n{'#'*60}")
    print("# COMPARING PAC-MAN MODELS")
    print(f"{'#'*60}")
    print(f"Model 1: {args.model1}")
    print(f"Model 2: {args.model2}")
    print(f"Episodes per model: {args.episodes}")
    print(f"{'#'*60}\n")
    
    # Load models
    print("Loading models...")
    
    if args.model1.endswith('.zip'):
        args.model1 = args.model1[:-4]
    if args.model2.endswith('.zip'):
        args.model2 = args.model2[:-4]
    
    try:
        model1 = PPO.load(args.model1)
        print(f"✓ Loaded Model 1")
    except Exception as e:
        print(f"✗ Failed to load Model 1: {e}")
        return
    
    try:
        model2 = PPO.load(args.model2)
        print(f"✓ Loaded Model 2")
    except Exception as e:
        print(f"✗ Failed to load Model 2: {e}")
        return
    
    # Play games with each model
    print("\n" + "="*60)
    print("Testing Model 1")
    print("="*60)
    
    visualizer1 = GameVisualizer(frame_time=args.speed)
    
    for episode in range(args.episodes):
        visualizer1.play_game(
            pacman_model=model1,
            layout_name=args.layout,
            ghost_type=args.ghost_type,
            num_ghosts=args.num_ghosts,
            max_steps=args.max_steps,
            deterministic=not args.stochastic,
            show_stats=True
        )
        if episode < args.episodes - 1:
            time.sleep(1)
    
    print("\n" + "="*60)
    print("Testing Model 2")
    print("="*60)
    
    visualizer2 = GameVisualizer(frame_time=args.speed)
    
    for episode in range(args.episodes):
        visualizer2.play_game(
            pacman_model=model2,
            layout_name=args.layout,
            ghost_type=args.ghost_type,
            num_ghosts=args.num_ghosts,
            max_steps=args.max_steps,
            deterministic=not args.stochastic,
            show_stats=True
        )
        if episode < args.episodes - 1:
            time.sleep(1)
    
    # Print comparison
    print("\n" + "#"*60)
    print("# COMPARISON RESULTS")
    print("#"*60)
    
    print(f"\nModel 1: {os.path.basename(args.model1)}")
    print(f"  Win Rate: {100*visualizer1.stats['wins']/visualizer1.stats['games_played']:.1f}%")
    print(f"  Avg Score: {visualizer1.stats['total_score']/visualizer1.stats['games_played']:.1f}")
    
    print(f"\nModel 2: {os.path.basename(args.model2)}")
    print(f"  Win Rate: {100*visualizer2.stats['wins']/visualizer2.stats['games_played']:.1f}%")
    print(f"  Avg Score: {visualizer2.stats['total_score']/visualizer2.stats['games_played']:.1f}")
    
    print("\n" + "#"*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize trained Pac-Man agents in real-time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch Stage 2 agent (vs random ghosts)
  python visualize_agents.py \\
      --pacman-model curriculum_output/run_*/models/pacman_random_ghosts_final \\
      --episodes 5

  # Watch adversarial agent (vs learned ghosts)
  python visualize_agents.py \\
      --pacman-model curriculum_output/run_*/models/pacman_adversarial_v2 \\
      --ghost-dir curriculum_output/run_*/models \\
      --ghost-version 2 \\
      --episodes 3

  # Slow motion (more time to see strategy)
  python visualize_agents.py \\
      --pacman-model path/to/model \\
      --speed 0.2 \\
      --episodes 1

  # Compare two models
  python visualize_agents.py --compare \\
      --model1 curriculum_output/run_*/models/pacman_random_ghosts_final \\
      --model2 curriculum_output/run_*/models/pacman_adversarial_v2 \\
      --episodes 5
        """
    )
    
    # Mode selection
    parser.add_argument('--compare', action='store_true',
                       help='Compare two models (sequential visualization)')
    
    # Single model mode
    parser.add_argument('--pacman-model', type=str,
                       help='Path to Pac-Man model (without .zip)')
    
    # Comparison mode
    parser.add_argument('--model1', type=str,
                       help='Path to first model for comparison')
    parser.add_argument('--model2', type=str,
                       help='Path to second model for comparison')
    
    # Ghost configuration
    parser.add_argument('--ghost-dir', type=str,
                       help='Directory with trained ghost models (for adversarial)')
    parser.add_argument('--ghost-version', type=int,
                       help='Ghost version number (for adversarial)')
    parser.add_argument('--ghost-type', type=str, default='random',
                       choices=['random', 'directional'],
                       help='Ghost type for scripted ghosts (default: random)')
    parser.add_argument('--num-ghosts', type=int, default=4,
                       help='Number of ghosts (default: 4)')
    
    # Environment settings
    parser.add_argument('--layout', type=str, default='mediumClassic',
                       help='Map layout (default: mediumClassic)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per game (default: 500)')
    
    # Visualization settings
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of games to play (default: 1)')
    parser.add_argument('--speed', type=float, default=0.1,
                       help='Time between frames in seconds (default: 0.1)')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (default: deterministic)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.compare:
        if not args.model1 or not args.model2:
            parser.error("--model1 and --model2 required for --compare mode")
        compare_models(args)
    else:
        if not args.pacman_model:
            parser.error("--pacman-model required (or use --compare mode)")
        visualize_single_model(args)


if __name__ == '__main__':
    main()