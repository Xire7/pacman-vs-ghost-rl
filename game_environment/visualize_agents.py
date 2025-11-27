#!/usr/bin/env python3
"""
Visualize and Record trained Pac-Man vs Ghosts agents

Features:
- Live visualization
- Video recording to MP4
- Episode statistics

Usage:
    # Live visualization only
    python visualize_agents.py --pacman-path models/pacman_v1 --ghost-dir models/ --ghost-version 1
    
    # Record video
    python visualize_agents.py --pacman-path models/pacman_v1 --ghost-dir models/ --ghost-version 1 --record --video-folder videos
"""

import argparse
import os
import time
import numpy as np
from stable_baselines3 import PPO, DQN
from gym_env import PacmanEnv
import json


def visualize_game(
    pacman_model_path,
    ghost_model_dir,
    ghost_version,
    layout_name='mediumClassic',
    num_ghosts=2,
    n_episodes=5,
    frame_delay=0.05,
    record_video=False,
    video_folder='videos',
    video_fps=10
):
    """
    Visualize trained agents playing with optional video recording.
    
    Args:
        pacman_model_path: Path to Pac-Man model (without .zip)
        ghost_model_dir: Directory containing ghost models
        ghost_version: Version number of ghosts to load
        layout_name: Map layout
        num_ghosts: Number of ghosts
        n_episodes: Number of episodes to play
        frame_delay: Delay between frames (seconds)
        record_video: Whether to record video
        video_folder: Directory to save videos
        video_fps: Frames per second for video
    
    Returns:
        dict: Statistics from all episodes
    """
    print(f"\n{'='*60}")
    print(f"Loading models...")
    print(f"{'='*60}")
    
    # Load Pac-Man
    pacman_model = PPO.load(pacman_model_path)
    print(f"✓ Loaded Pac-Man: {pacman_model_path}")
    
    # Load ghosts
    ghost_models = {}
    for i in range(1, num_ghosts + 1):
        ghost_path = os.path.join(ghost_model_dir, f"ghost_{i}_v{ghost_version}")
        if os.path.exists(ghost_path + ".zip"):
            ghost_models[i] = DQN.load(ghost_path)
            print(f"✓ Loaded Ghost {i}: {ghost_path}")
        else:
            print(f"⚠️  Ghost {i} not found: {ghost_path}")
    
    if len(ghost_models) == 0:
        print("Error: No ghost models loaded!")
        return None
    
    # Setup video recording if requested
    if record_video:
        try:
            import imageio
            print(f"\n📹 Video recording enabled")
            print(f"   Output folder: {video_folder}")
            print(f"   FPS: {video_fps}")
        except ImportError:
            print("\n⚠️  Video recording requires imageio[ffmpeg]")
            print("   Installing: pip install imageio[ffmpeg]")
            import subprocess
            subprocess.check_call(['pip', 'install', 'imageio[ffmpeg]'])
            import imageio
            print("✓ Installed!\n")
        
        os.makedirs(video_folder, exist_ok=True)
    
    # Create environment with rendering
    env = PacmanEnv(
        layout_name=layout_name,
        ghost_policies=ghost_models,
        max_steps=500,
        render_mode='human' if not record_video else 'text',  # Text mode for recording
        reward_shaping=False
    )
    
    print(f"\n{'='*60}")
    print(f"Starting visualization ({n_episodes} episodes)")
    print(f"{'='*60}\n")
    
    # Statistics tracking
    all_stats = {
        'episodes': [],
        'pacman_wins': 0,
        'ghost_wins': 0,
        'timeouts': 0,
        'scores': [],
        'steps': []
    }
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        
        print(f"\n{'─'*60}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'─'*60}")
        
        # Record frames if video recording enabled
        frames = [] if record_video else None
        
        while not done:
            # Capture frame for video (before action)
            if record_video:
                # Convert game state to ASCII art frame
                frame_text = str(env.game_state)
                frames.append(frame_text)
            
            # Pac-Man action
            action, _ = pacman_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Delay for visibility (only if showing GUI)
            if not record_video:
                time.sleep(frame_delay)
        
        # Save video if recording
        if record_video and frames:
            video_filename = f"episode_{episode+1}_v{ghost_version}.txt"
            video_path = os.path.join(video_folder, video_filename)
            
            with open(video_path, 'w') as f:
                f.write(f"Episode {episode + 1} - Pac-Man v{ghost_version} vs Ghosts v{ghost_version}\n")
                f.write("=" * 60 + "\n\n")
                for i, frame in enumerate(frames):
                    f.write(f"Step {i+1}:\n{frame}\n\n")
            
            print(f"📹 Saved ASCII replay: {video_path}")
        
        # Episode summary
        if info.get('win', False):
            result = "PAC-MAN WINS!"
            all_stats['pacman_wins'] += 1
        elif info.get('lose', False):
            result = "GHOSTS WIN!"
            all_stats['ghost_wins'] += 1
        else:
            result = "TIMEOUT"
            all_stats['timeouts'] += 1
        
        score = info.get('raw_score', 0)
        all_stats['scores'].append(score)
        all_stats['steps'].append(steps)
        
        episode_stats = {
            'episode': episode + 1,
            'result': result,
            'score': score,
            'steps': steps,
            'reward': episode_reward
        }
        all_stats['episodes'].append(episode_stats)
        
        print(f"  {result}")
        print(f"  Score:  {score:.0f}")
        print(f"  Steps:  {steps}")
        print(f"  Reward: {episode_reward:.2f}")
    
    env.close()
    
    # Print summary
    print(f"\n\n{'='*60}")
    print(f"SUMMARY ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Pac-Man Wins:  {all_stats['pacman_wins']}/{n_episodes} " +
          f"({100*all_stats['pacman_wins']/n_episodes:.1f}%)")
    print(f"  Ghost Wins:    {all_stats['ghost_wins']}/{n_episodes} " +
          f"({100*all_stats['ghost_wins']/n_episodes:.1f}%)")
    print(f"  Timeouts:      {all_stats['timeouts']}/{n_episodes}")
    print(f"  Avg Score:     {np.mean(all_stats['scores']):.1f} ± {np.std(all_stats['scores']):.1f}")
    print(f"  Avg Steps:     {np.mean(all_stats['steps']):.1f}")
    print(f"{'='*60}")
    
    # Save statistics to JSON
    if record_video:
        stats_path = os.path.join(video_folder, f"stats_v{ghost_version}.json")
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n✓ Statistics saved to: {stats_path}")
    
    print("\nVisualization complete! 🎉")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description='Visualize and record trained Pac-Man vs Ghosts'
    )
    
    # Model paths
    parser.add_argument('--pacman-path', type=str, required=True,
                       help='Path to Pac-Man model (without .zip)')
    parser.add_argument('--ghost-dir', type=str, required=True,
                       help='Directory containing ghost models')
    parser.add_argument('--ghost-version', type=int, required=True,
                       help='Ghost version number to load')
    
    # Environment settings
    parser.add_argument('--layout', type=str, default='mediumClassic',
                       help='Map layout (default: mediumClassic)')
    parser.add_argument('--num-ghosts', type=int, default=2,
                       help='Number of ghosts (default: 2)')
    
    # Visualization settings
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to visualize (default: 5)')
    parser.add_argument('--frame-delay', type=float, default=0.05,
                       help='Delay between frames in seconds (default: 0.05)')
    
    # Video recording
    parser.add_argument('--record', action='store_true',
                       help='Record episodes as text replay')
    parser.add_argument('--video-folder', type=str, default='videos',
                       help='Folder to save videos (default: videos)')
    parser.add_argument('--video-fps', type=int, default=10,
                       help='Video frames per second (default: 10)')
    
    args = parser.parse_args()
    
    visualize_game(
        pacman_model_path=args.pacman_path,
        ghost_model_dir=args.ghost_dir,
        ghost_version=args.ghost_version,
        layout_name=args.layout,
        num_ghosts=args.num_ghosts,
        n_episodes=args.episodes,
        frame_delay=args.frame_delay,
        record_video=args.record,
        video_folder=args.video_folder,
        video_fps=args.video_fps
    )


def create_gif_from_game(
    pacman_model_path,
    ghost_model_dir,
    ghost_version,
    layout_name='mediumClassic',
    num_ghosts=2,
    output_path='game.gif',
    max_steps=200,
    fps=5
):
    """
    Create a GIF animation of a single game.
    
    This uses matplotlib to render each frame as an image.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Installing matplotlib...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'matplotlib', 'pillow'])
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.animation import FuncAnimation, PillowWriter
    
    # Load models
    print(f"Loading models...")
    pacman_model = PPO.load(pacman_model_path)
    
    ghost_models = {}
    for i in range(1, num_ghosts + 1):
        ghost_path = os.path.join(ghost_model_dir, f"ghost_{i}_v{ghost_version}")
        if os.path.exists(ghost_path + ".zip"):
            ghost_models[i] = DQN.load(ghost_path)
    
    # Create environment
    env = PacmanEnv(
        layout_name=layout_name,
        ghost_policies=ghost_models,
        max_steps=max_steps,
        render_mode=None,
        reward_shaping=False
    )
    
    # Collect all frames
    print(f"Running game and collecting frames...")
    frames = []
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        # Capture current state
        state = env.game_state
        frames.append({
            'walls': state.getWalls(),
            'food': state.getFood(),
            'pacman_pos': state.getPacmanPosition(),
            'ghost_positions': state.getGhostPositions(),
            'score': state.getScore()
        })
        
        # Take action
        action, _ = pacman_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    env.close()
    
    print(f"Collected {len(frames)} frames. Creating GIF...")
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def draw_frame(frame_idx):
        ax.clear()
        frame = frames[frame_idx]
        
        # Get dimensions
        walls = frame['walls']
        width = walls.width
        height = walls.height
        
        # Set up plot
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Step {frame_idx + 1}/{len(frames)} | Score: {frame['score']:.0f}")
        ax.axis('off')
        
        # Draw walls
        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                             linewidth=0, facecolor='blue')
                    ax.add_patch(rect)
        
        # Draw food
        food = frame['food']
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    circle = patches.Circle((x, y), 0.1, color='white')
                    ax.add_patch(circle)
        
        # Draw Pac-Man
        pacman_x, pacman_y = frame['pacman_pos']
        pacman = patches.Circle((pacman_x, pacman_y), 0.4, color='yellow')
        ax.add_patch(pacman)
        
        # Draw ghosts
        colors = ['red', 'cyan', 'pink', 'orange']
        for i, (gx, gy) in enumerate(frame['ghost_positions']):
            ghost = patches.Circle((gx, gy), 0.4, color=colors[i % len(colors)])
            ax.add_patch(ghost)
    
    # Create animation
    anim = FuncAnimation(fig, draw_frame, frames=len(frames), 
                        interval=1000/fps, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    
    plt.close()
    
    print(f"✓ GIF saved to: {output_path}")
    
    return output_path

if __name__ == '__main__':
    main()