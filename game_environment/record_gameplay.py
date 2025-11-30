#!/usr/bin/env python3
"""
Simple video recorder for Pac-Man visualization.

Captures the actual Tkinter rendering window and saves as MP4 or GIF.
"""

import argparse
import os
import time
import numpy as np
import subprocess
import sys

# Try to import dependencies, install if missing
try:
    from PIL import ImageGrab
    import imageio
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call(['pip', 'install', 'pillow', 'imageio[ffmpeg]'])
    from PIL import ImageGrab
    import imageio

# Import models
from stable_baselines3 import PPO, DQN
from sb3_contrib import MaskablePPO
from gym_env import PacmanEnv


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
    Record gameplay by capturing the Tkinter rendering window.
    """
    print(f"\n{'='*60}")
    print("Loading models...")
    print(f"{'='*60}")

    # Load Pac-Man model (assumed MaskablePPO)
    pacman_model = MaskablePPO.load(pacman_model_path)

    # Load ghost models
    ghost_models = {}
    for i in range(1, num_ghosts + 1):
        ghost_path = os.path.join(ghost_model_dir, f"ghost_{i}_v{ghost_version}")
        if os.path.exists(ghost_path + ".zip"):
            ghost_models[i] = DQN.load(ghost_path)
            print(f"Loaded Ghost {i}: {ghost_path}")
        else:
            print(f"Warning: Ghost {i} model not found: {ghost_path}")

    # Create environment
    env = PacmanEnv(
        layout_name=layout_name,
        ghost_policies=ghost_models,
        max_steps=max_steps,
        render_mode='human',  # for Tkinter window
    )

    print("Starting game (will capture window)...")
    print("Do not minimize or cover the game window during recording!\n")
    obs, _ = env.reset()

    # Wait for the window to stabilize
    time.sleep(2.0)

    # Optional: get window coords (if needed for region capture)
    # Skip if you want full-screen capture; otherwise, you could implement region detection here

    frames = []
    step_count = 0
    done = False
    final_info = {}

    print(f"Recording... (max {max_steps} steps)")
    while step_count < max_steps:
        # Capture frame
        try:
            bbox = None
            # Optionally, specify bbox here if known.
            frame = ImageGrab.grab(bbox=bbox)
            frames.append(np.array(frame))
        except Exception as e:
            print(f"Frame capture error at step {step_count}: {e}")
            break

        if done:
            # Small buffer after game ends
            if step_count - len(frames) > 10:
                break
        else:
            # Take action
            action, _ = pacman_model.predict(obs, deterministic=True, action_masks=env.action_masks())
            if hasattr(action, 'item'):
                action = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                final_info = info

        step_count += 1
        time.sleep(1.0 / fps)

    env.close()

    print(f"Captured {len(frames)} frames.")
    if len(frames) == 0:
        print("No frames captured! Exiting.")
        return None

    print(f"Saving video to: {output_path}")
    # Save the frames as GIF or MP4
    try:
        if format == 'gif' or output_path.endswith('.gif'):
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
        else:
            # For MP4, use codec='libx264'
            imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
        print(f"Saved {format.upper()} to: {output_path}")
        print(f"Duration: {len(frames) / fps:.1f} seconds")
        if done and final_info:
            if final_info.get('win', False):
                print(f"PAC-MAN WINS! Score: {final_info.get('raw_score', 0):.0f}")
            elif final_info.get('lose', False):
                print(f"GHOSTS WIN! Score: {final_info.get('raw_score', 0):.0f}")
        else:
            print(f"Stopping recording at {step_count} steps.")
        return output_path
    except Exception as e:
        print(f"Error saving video: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Record Pac-Man gameplay video.')
    parser.add_argument('--pacman-path', type=str, required=True, help='Path to Pac-Man model (without .zip)')
    parser.add_argument('--ghost-dir', type=str, required=True, help='Directory with ghost models')
    parser.add_argument('--ghost-version', type=int, required=True, help='Ghost version number')
    parser.add_argument('--layout', type=str, default='mediumClassic', help='Layout name')
    parser.add_argument('--num-ghosts', type=int, default=2, help='Number of ghosts')
    parser.add_argument('--output', type=str, default='gameplay.mp4', help='Output filename')
    parser.add_argument('--format', type=str, choices=['mp4','gif'], default='mp4', help='Output format')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    args = parser.parse_args()

    # Run the recorder
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
