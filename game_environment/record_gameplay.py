#!/usr/bin/env python3
"""
Simple video recorder for Pac-Man visualization.

Captures the actual Tkinter rendering window and saves as MP4 or GIF.
"""

import argparse
import os
import time
import numpy as np
from stable_baselines3 import PPO, DQN
from gym_env import PacmanEnv
import json


def record_gameplay_video(
    pacman_model_path,
    ghost_model_dir,
    ghost_version,
    layout_name='mediumClassic',
    num_ghosts=2,
    output_path='gameplay.mp4',
    max_steps=500,
    fps=10,
    format='mp4'
):
    """
    Record gameplay by capturing the actual Tkinter window.
    
    Args:
        pacman_model_path: Path to Pac-Man model
        ghost_model_dir: Directory with ghost models
        ghost_version: Ghost version number
        layout_name: Map layout
        num_ghosts: Number of ghosts
        output_path: Where to save video
        max_steps: Max steps to record
        fps: Frames per second
        format: 'mp4' or 'gif'
    """
    # Install dependencies
    try:
        from PIL import ImageGrab
        import imageio
    except ImportError:
        print("📦 Installing dependencies...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pillow', 'imageio[ffmpeg]'])
        from PIL import ImageGrab
        import imageio
        print("✓ Dependencies installed!\n")
    
    print(f"\n{'='*60}")
    print(f"Recording Gameplay Video")
    print(f"{'='*60}")
    
    # Load models
    print(f"Loading models...")
    pacman_model = PPO.load(pacman_model_path)
    
    ghost_models = {}
    for i in range(1, num_ghosts + 1):
        ghost_path = os.path.join(ghost_model_dir, f"ghost_{i}_v{ghost_version}")
        if os.path.exists(ghost_path + ".zip"):
            ghost_models[i] = DQN.load(ghost_path)
            print(f"  ✓ Ghost {i}")
    
    # Create environment with rendering
    env = PacmanEnv(
        layout_name=layout_name,
        ghost_policies=ghost_models,
        max_steps=max_steps,
        render_mode='human',
        reward_shaping=False
    )
    
    print(f"\nStarting game (will capture window)...")
    print(f"Do not minimize or cover the game window during recording!\n")
    
    obs, _ = env.reset()
    
    # Wait longer for window to fully render and stabilize
    time.sleep(2.0)
    
    # Get window coordinates with retry
    bbox = None
    for attempt in range(3):
        try:
            import graphicsUtils
            canvas = graphicsUtils._canvas
            
            # Force update to get accurate dimensions
            canvas.update_idletasks()
            canvas.update()
            
            # Get canvas screen position
            x = canvas.winfo_rootx()
            y = canvas.winfo_rooty()
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            
            # Add small padding to ensure we get everything
            padding = 5
            bbox = (x - padding, y - padding, x + w + padding, y + h + padding)
            
            print(f"✓ Capturing region: {w}x{h} at ({x}, {y})")
            break
        except Exception as e:
            if attempt < 2:
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1.0)
            else:
                print(f"Could not get window coordinates: {e}")
                print("Using full screen capture (may need cropping)")
    
    # Record frames
    frames = []
    done = False
    step_count = 0
    final_info = {}
    
    print(f"Recording... (max {max_steps} steps)")
    
    while step_count < max_steps:
        # Capture frame
        try:
            if bbox:
                frame = ImageGrab.grab(bbox=bbox)
            else:
                frame = ImageGrab.grab()
            
            frames.append(np.array(frame))
        except Exception as e:
            print(f"Frame capture failed at step {step_count}: {e}")
        
        if done:
            # Capture a few extra frames after game ends to show final state
            if step_count >= max_steps or len(frames) > step_count + 10:
                break
        else:
            # Take action
            action, _ = pacman_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                final_info = info
        
        step_count += 1
        
        # Small delay for consistent frame rate
        time.sleep(1.0 / fps)
    
    env.close()
    
    print(f"\n✓ Captured {len(frames)} frames")
    
    # Save video
    if len(frames) == 0:
        print("No frames captured!")
        return None
    
    print(f"Saving video to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        if format == 'gif' or output_path.endswith('.gif'):
            # Save as GIF
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
        else:
            # Save as MP4
            imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
        
        print(f"✓ Video saved: {output_path}")
        print(f"  Duration: {len(frames) / fps:.1f} seconds")
        print(f"  Steps recorded: {step_count}")
        
        # Print game result
        if done and final_info:
            if final_info.get('win', False):
                print(f"\nPAC-MAN WINS! Score: {final_info.get('raw_score', 0):.0f}")
            elif final_info.get('lose', False):
                print(f"\n GHOSTS WIN! Score: {final_info.get('raw_score', 0):.0f}")
        else:
            print(f"\n⏱Recording stopped at {step_count} steps")
        
        return output_path
    
    except Exception as e:
        print(f"Failed to save video: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Record Pac-Man gameplay video')
    
    parser.add_argument('--pacman-path', type=str, required=True,
                       help='Path to Pac-Man model (without .zip)')
    parser.add_argument('--ghost-dir', type=str, required=True,
                       help='Directory containing ghost models')
    parser.add_argument('--ghost-version', type=int, required=True,
                       help='Ghost version number')
    
    parser.add_argument('--layout', type=str, default='mediumClassic',
                       help='Map layout (default: mediumClassic)')
    parser.add_argument('--num-ghosts', type=int, default=2,
                       help='Number of ghosts (default: 2)')
    
    parser.add_argument('--output', type=str, default='gameplay.mp4',
                       help='Output file path (default: gameplay.mp4)')
    parser.add_argument('--format', type=str, choices=['mp4', 'gif'], default='mp4',
                       help='Output format (default: mp4)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps to record (default: 500)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second (default: 10)')
    
    args = parser.parse_args()
    
    record_gameplay_video(
        pacman_model_path=args.pacman_path,
        ghost_model_dir=args.ghost_dir,
        ghost_version=args.ghost_version,
        layout_name=args.layout,
        num_ghosts=args.num_ghosts,
        output_path=args.output,
        max_steps=args.max_steps,
        fps=args.fps,
        format=args.format
    )


if __name__ == '__main__':
    main()