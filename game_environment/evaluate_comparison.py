import os
import sys
import argparse
import time
import numpy as np
from typing import Dict, Any, Optional, Union

# --- Add game_environment to path to enable imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if "game_environment" not in script_dir:
    sys.path.insert(0, os.path.join(script_dir, 'game_environment'))

try:
    from stable_baselines3 import DQN
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gym_env import PacmanEnv
except ImportError:
    print("Error: Required libraries (stable-baselines3, sb3-contrib, gym_env) not found.")
    print("Please ensure your environment is set up and check requirements.txt.")
    sys.exit(1)


# Define type hint for convenience
VecEnvType = Union[VecNormalize, DummyVecEnv]
SingleEnvType = ActionMasker


def load_models(run_dir: str, num_ghosts: int, final_version: int) -> tuple[MaskablePPO, Dict[int, Any], Optional[str]]:
    """Loads all models and determines the VecNormalize path for Pac-Man."""
    
    model_dir_base = os.path.join(run_dir, 'models')
    pacman_model_base_path = os.path.join(model_dir_base, f"pacman_v{final_version}")
    pacman_path_zip = pacman_model_base_path + ".zip"

    # 1. Load Pac-Man Model
    print(f"Loading Pac-Man v{final_version} from: {pacman_path_zip}")
    if not os.path.exists(pacman_path_zip):
        raise FileNotFoundError(f"Pac-Man model file not found: {pacman_path_zip}")
    
    pacman_model = MaskablePPO.load(pacman_path_zip, custom_objects={})
    print("✓ Pac-Man model loaded successfully.")

    # 2. Check for VecNormalize stats
    vecnorm_path = os.path.join(model_dir_base, 'vecnormalize.pkl')
    if os.path.exists(vecnorm_path):
        print(f"✓ Found VecNormalize stats at: {vecnorm_path}")
    else:
        vecnorm_path = None
        print("✗ VecNormalize stats not found. Assuming model was trained without normalization.")

    # 3. Load Ghost Models
    ghost_models = {}
    print("Loading Ghost models...")
    ghost_version = final_version
    
    for i in range(1, num_ghosts + 1):
        ghost_path = os.path.join(model_dir_base, f"ghost_{i}_v{ghost_version}.zip")
        if os.path.exists(ghost_path):
            ghost_models[i] = DQN.load(ghost_path, custom_objects={})
            print(f"  ✓ Ghost {i} v{ghost_version} loaded.")
        else:
            print(f"  ✗ Warning: Ghost {i} model not found. Will use RandomGhost for this agent.")

    return pacman_model, ghost_models, vecnorm_path


def create_env_with_vecnorm(
    layout: str, 
    vecnorm_path: Optional[str], 
    ghost_models: Optional[Dict[int, Any]], 
    ghost_type: str,
    render_mode: Optional[str] = None
) -> Union[VecEnvType, SingleEnvType]:
    """Creates a single vectorized environment, optionally wrapping it with VecNormalize."""
    
    def make_env_fn():
        env_unwrapped = PacmanEnv(
            layout_name=layout, 
            ghost_policies=ghost_models,
            ghost_type=ghost_type, 
            max_steps=500,
            render_mode=render_mode  # Pass render mode (None or 'human')
        )
        return ActionMasker(env_unwrapped, lambda e: e.action_masks())

    # Create a DummyVecEnv from the factory function
    env = DummyVecEnv([make_env_fn])

    # Apply VecNormalize if path exists
    if vecnorm_path:
        env_wrapper = VecNormalize.load(vecnorm_path, env)
        env_wrapper.training = False  # Freeze running statistics for evaluation
        env_wrapper.norm_reward = False # Do not normalize rewards for final score
        return env_wrapper
    else:
        return env


def evaluate_pacman(
    model: MaskablePPO, 
    layout: str, 
    n_episodes: int, 
    vecnorm_path: Optional[str], 
    ghost_models: Optional[Dict[int, Any]] = None, 
    ghost_type: str = 'random',
    render: bool = False,
    frame_delay: float = 0.05
) -> Dict[str, float]:
    """Runs evaluation episodes with correct environment setup and optional rendering."""
    wins = 0
    scores = []
    
    render_mode = 'human' if render else None
    
    label = 'Trained Ghosts' if ghost_models else f'{ghost_type.capitalize()} Ghosts'
    print(f"\nEvaluating against {label} ({n_episodes} episodes)...")

    for i in range(1, n_episodes + 1):
        # Create environment
        env_wrapper = create_env_with_vecnorm(layout, vecnorm_path, ghost_models, ghost_type, render_mode)
        
        is_vec_env = isinstance(env_wrapper, (VecNormalize, DummyVecEnv))
        
        # Robust Reset
        reset_output = env_wrapper.reset()
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            obs, _ = reset_output
        else:
            obs = reset_output 

        done = False
        steps = 0
        
        while not done:
            # Retrieve action masks
            if is_vec_env:
                masks = env_wrapper.env_method('action_masks')[0]
            else:
                masks = env_wrapper.action_masks()
                
            # Predict action
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            
            # Robust Step
            if is_vec_env:
                obs_array, _, done_array, info_list = env_wrapper.step([action])
                obs = obs_array
                info = info_list[0]
                done = done_array[0]
            else:
                obs, _, terminated, truncated, info = env_wrapper.step(action)
                done = terminated or truncated
            
            # Render delay
            if render:
                time.sleep(frame_delay)
                
            steps += 1
        
        if info.get('win'):
            wins += 1
            result = "WIN"
        else:
            result = "LOSE"
            
        score = info.get('raw_score', 0)
        scores.append(score)
        
        env_wrapper.close()
        
        # Progress logging
        if render:
            # If rendering, print every episode so user knows what happened
            print(f"  Ep {i}: {result} | Score: {score} | Steps: {steps}")
        elif i % (n_episodes // 5 if n_episodes >= 5 else 1) == 0:
             print(f"  Episodes complete: {i}/{n_episodes} | Wins: {wins}")
             
    return {
        'win_rate': wins / n_episodes,
        'avg_score': np.mean(scores),
        'std_score': np.std(scores)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained Pac-Man PPO model against trained DQN ghosts and random ghosts.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--run-dir', type=str, required=True,
                       help="The base path to the training output directory.")
    parser.add_argument('--version', type=int, required=True,
                       help="The version number (round number) of the models to load.")
    parser.add_argument('--layout', type=str, default='mediumClassic',
                       help="The layout used for training/evaluation.")
    parser.add_argument('--num-ghosts', type=int, default=4,
                       help="Number of ghosts in the layout.")
    parser.add_argument('--episodes', type=int, default=50,
                       help="Number of evaluation episodes per scenario.")
    
    # Visualization args
    parser.add_argument('--render', action='store_true',
                       help="Enable real-time visualization of the games.")
    parser.add_argument('--frame-delay', type=float, default=0.05,
                       help="Delay between frames in seconds (default: 0.05).")

    if not sys.argv[1:]:
        print("Required arguments missing.")
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    # --- Load Models ---
    try:
        pacman_model, trained_ghost_models, vecnorm_path = load_models(args.run_dir, args.num_ghosts, args.version)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        return

    # --- Run Evaluations ---
    print(f"\n{'='*60}")
    print("STARTING EVALUATION RUNS")
    if args.render:
        print(f"Visualization Enabled (Delay: {args.frame_delay}s)")
    print(f"{'='*60}")
    
    # 1. Evaluation against Trained Ghosts
    stats_trained = evaluate_pacman(
        pacman_model, 
        args.layout, 
        args.episodes, 
        vecnorm_path,
        ghost_models=trained_ghost_models,
        ghost_type='directional',
        render=args.render,
        frame_delay=args.frame_delay
    )
    
    # 2. Evaluation against Random Ghosts
    stats_random = evaluate_pacman(
        pacman_model, 
        args.layout, 
        args.episodes, 
        vecnorm_path,
        ghost_models=None, 
        ghost_type='random',
        render=args.render,
        frame_delay=args.frame_delay
    )
    
    # --- Print Comparison ---
    print(f"\n\n{'#'*70}")
    print("PAC-MAN MODEL EVALUATION COMPARISON")
    print(f"Model: PPO v{args.version} | Layout: {args.layout} | Episodes: {args.episodes}")
    print(f"{'#'*70}")

    print(f"{'| Evaluation Scenario':<30} | {'Win Rate':<10} | {'Avg Score (±Std)':<25} |")
    print("-" * 70)
    
    print(f"| {'vs TRAINED GHOSTS':<30} | {stats_trained['win_rate']*100:<9.1f}% | {stats_trained['avg_score']:<7.1f} (±{stats_trained['std_score']:<5.1f}) |")
    print(f"| {'vs RANDOM GHOSTS (Baseline)':<30} | {stats_random['win_rate']*100:<9.1f}% | {stats_random['avg_score']:<7.1f} (±{stats_random['std_score']:<5.1f}) |")

    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()