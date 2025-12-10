"""
Visualize and Record trained Pac-Man vs Ghosts agents
FIXED: Now supports VecNormalize!
"""

import argparse
import os
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_env import PacmanEnv
import json


def visualize_game(
    pacman_model_path,
    ghost_model_dir,
    ghost_version,
    layout_name='mediumClassic',
    num_ghosts=4,
    n_episodes=5,
    frame_delay=0.05,
    record_video=False,
    video_folder='videos',
    video_fps=10,
    vecnorm_path=None  # NEW: VecNormalize path
):
    # Install dependencies if recording
    if record_video:
        try:
            from PIL import ImageGrab
            import imageio
        except ImportError:
            print("Installing video recording dependencies...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'pillow', 'imageio[ffmpeg]'])
            from PIL import ImageGrab
            import imageio
            print("✓ Dependencies installed!\n")
        
        os.makedirs(video_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Loading models...")
    print(f"{'='*60}")
    
    # Auto-detect VecNormalize if not provided
    if vecnorm_path is None:
        model_dir = os.path.dirname(pacman_model_path)
        candidate = os.path.join(model_dir, 'vecnormalize.pkl')
        
        # Check parent directory too (for best/ subdirectory)
        if not os.path.exists(candidate):
            parent_dir = os.path.dirname(model_dir)
            parent_candidate = os.path.join(parent_dir, 'vecnormalize.pkl')
            if os.path.exists(parent_candidate):
                candidate = parent_candidate
        
        if os.path.exists(candidate):
            vecnorm_path = candidate
            print(f"✓ Auto-detected VecNormalize: {vecnorm_path}")
    
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"✓ Using VecNormalize: {vecnorm_path}")
        has_vecnorm = True
    else:
        print(f"⚠ No VecNormalize found (model may perform poorly!)")
        has_vecnorm = False
    
    # Load ghosts first
    ghost_models = {}
    for i in range(1, num_ghosts + 1):
        ghost_path = os.path.join(ghost_model_dir, f"ghost_{i}_v{ghost_version}")
        if os.path.exists(ghost_path + ".zip"):
            ghost_models[i] = DQN.load(ghost_path)
            print(f"✓ Loaded Ghost {i}: {ghost_path}")
        else:
            print(f"⚠ Ghost {i} not found: {ghost_path}")
    
    if len(ghost_models) == 0:
        print("Error: No ghost models loaded!")
        return None
    
    if record_video:
        print(f"\n✓ Video recording enabled")
        print(f"  Saving to: {video_folder}")
        print(f"  Format: MP4 @ {video_fps} FPS")
        print(f"  ⚠ Do NOT minimize or cover the game window during recording!")
    
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
        # Create environment with rendering
        base_env = PacmanEnv(
            layout_name=layout_name,
            ghost_policies=ghost_models,
            max_steps=500,
            render_mode='human',
        )
        # Wrap with ActionMasker for MaskablePPO
        env = ActionMasker(base_env, lambda e: e.action_masks())
        
        # Wrap in DummyVecEnv for VecNormalize compatibility
        env = DummyVecEnv([lambda: env])
        
        # Apply VecNormalize if available (CRITICAL FIX!)
        if has_vecnorm:
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False  # Don't update stats during visualization
            env.norm_reward = False  # Don't normalize rewards
        
        # Load Pac-Man model with the env
        pacman_model = MaskablePPO.load(pacman_model_path, env=env)
        
        obs = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        
        print(f"\n{'─'*60}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'─'*60}")
        
        # For video recording, get window coordinates
        frames = []
        bbox = None
        if record_video:
            time.sleep(1.5)  # Wait for window to render
            try:
                import graphicsUtils
                canvas = graphicsUtils._canvas
                canvas.update_idletasks()
                canvas.update()
                
                x = canvas.winfo_rootx()
                y = canvas.winfo_rooty()
                w = canvas.winfo_width()
                h = canvas.winfo_height()
                
                padding = 5
                bbox = (x - padding, y - padding, x + w + padding, y + h + padding)
                print(f"  Recording window: {w}x{h} at ({x}, {y})")
            except Exception as e:
                print(f"  ⚠ Could not get window coordinates: {e}")
                print(f"  Will use full screen capture")
        
        while not done:
            # Capture frame for video
            if record_video:
                try:
                    if bbox:
                        frame = ImageGrab.grab(bbox=bbox)
                    else:
                        frame = ImageGrab.grab()
                    frames.append(np.array(frame))
                except Exception as e:
                    print(f"  Frame capture error: {e}")
            
            # Get action masks from vectorized env
            action_masks = env.env_method('action_masks')[0]
            
            # Pac-Man action with action masking (now with normalized obs!)
            action, _ = pacman_model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks
            )
            
            # Convert to int
            if hasattr(action, 'item'):
                action = int(action.item())
            elif isinstance(action, np.ndarray):
                action = int(action[0])
            
            obs, reward, done_vec, info = env.step([action])
            
            episode_reward += reward[0]
            steps += 1
            done = done_vec[0]
            
            # Frame timing
            if not record_video:
                time.sleep(frame_delay)
            # When recording, go as fast as possible (no sleep)
        
        # Keep showing final state for a moment
        if record_video:
            for _ in range(int(video_fps * 0.5)):  # Show for 0.5 seconds
                try:
                    if bbox:
                        frame = ImageGrab.grab(bbox=bbox)
                    else:
                        frame = ImageGrab.grab()
                    frames.append(np.array(frame))
                except:
                    pass
        
        env.close()
        
        # Save video
        if record_video and len(frames) > 0:
            video_filename = f"episode_{episode+1}_v{ghost_version}.mp4"
            video_path = os.path.join(video_folder, video_filename)
            
            try:
                imageio.mimsave(video_path, frames, fps=video_fps, codec='libx264')
                print(f"  ✓ Saved video: {video_filename} ({len(frames)} frames)")
            except Exception as e:
                print(f"  ✗ Failed to save video: {e}")
        
        # Extract info from vectorized wrapper
        info = info[0]
        
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
        print(f"\nStatistics saved to: {stats_path}")
        print(f"Videos saved to: {video_folder}")
    
    print("\n✓ Visualization complete!")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(description='Visualize trained Pac-Man vs Ghosts')
    
    parser.add_argument('--pacman-path', type=str, required=True,
                       help='Path to Pac-Man model (.zip file)')
    parser.add_argument('--ghost-dir', type=str, required=True,
                       help='Directory containing ghost models')
    parser.add_argument('--ghost-version', type=int, required=True,
                       help='Ghost version to load (e.g., 10 for v10)')
    parser.add_argument('--vecnorm-path', type=str, default=None,
                       help='Path to vecnormalize.pkl (auto-detects if not provided)')
    parser.add_argument('--layout', type=str, default='mediumClassic')
    parser.add_argument('--num-ghosts', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--frame-delay', type=float, default=0.05)
    parser.add_argument('--record', action='store_true', help='Record videos of gameplay')
    parser.add_argument('--video-folder', type=str, default='videos')
    parser.add_argument('--video-fps', type=int, default=10)
    
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
        video_fps=args.video_fps,
        vecnorm_path=args.vecnorm_path  # NEW parameter
    )


if __name__ == '__main__':
    main()