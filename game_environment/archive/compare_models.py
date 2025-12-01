#!/usr/bin/env python3
"""
Compare Pac-Man models against random and trained ghosts.

Usage:
    python compare_models.py --pacman path/to/model.zip [--pacman path/to/model2.zip ...]
    python compare_models.py --pacman-dir training_output/mixed_*/models/
    python compare_models.py --all  # Compare all models in training_output/
    
Options:
    --pacman PATH       Path to a Pac-Man model (.zip file)
    --pacman-dir PATH   Directory containing pacman*.zip models
    --ghost1 PATH       Path to ghost 1 model (default: latest in training_output)
    --ghost2 PATH       Path to ghost 2 model (default: latest in training_output)
    --games N           Number of games per evaluation (default: 500)
    --layout NAME       Layout to use (default: mediumClassic)
    --all               Evaluate all pacman models found in training_output/
    --random-only       Only evaluate against random ghosts
    --trained-only      Only evaluate against trained ghosts
"""

import argparse
import glob
import os
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN
from gym_env import PacmanEnv
import numpy as np


def evaluate(model, layout='mediumClassic', ghost_models=None, n=500, verbose=True):
    """Evaluate a Pac-Man model over n games."""
    wins = 0
    total_score = 0
    for i in range(n):
        if ghost_models:
            env = PacmanEnv(layout_name=layout, ghost_policies=ghost_models, max_steps=500)
        else:
            env = PacmanEnv(layout_name=layout, ghost_type='random', max_steps=500)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            if isinstance(action, np.ndarray):
                action = int(action.item())
            obs, _, t, tr, info = env.step(action)
            done = t or tr
        if info.get('win'):
            wins += 1
        total_score += info.get('score', 0)
        env.close()
        if verbose and (i + 1) % 100 == 0:
            print(f'    {i+1}/{n} games... ({wins}/{i+1} wins = {wins/(i+1)*100:.1f}%)')
    return {
        'win_rate': wins / n,
        'wins': wins,
        'games': n,
        'avg_score': total_score / n,
    }


def find_latest_ghost_models(base_dir='training_output'):
    """Find the latest trained ghost models."""
    # Look for ghost models in mixed training outputs
    ghost1_files = sorted(glob.glob(f'{base_dir}/mixed_*/models/ghost_1_v*.zip'))
    ghost2_files = sorted(glob.glob(f'{base_dir}/mixed_*/models/ghost_2_v*.zip'))
    
    if ghost1_files and ghost2_files:
        return ghost1_files[-1], ghost2_files[-1]
    
    # Fallback: look for any ghost models
    ghost1_files = sorted(glob.glob(f'{base_dir}/**/ghost_1*.zip', recursive=True))
    ghost2_files = sorted(glob.glob(f'{base_dir}/**/ghost_2*.zip', recursive=True))
    
    if ghost1_files and ghost2_files:
        return ghost1_files[-1], ghost2_files[-1]
    
    return None, None


def find_pacman_models(base_dir='training_output'):
    """Find all Pac-Man models in the training output directory."""
    models = {}
    
    # Look for models in various locations
    patterns = [
        f'{base_dir}/mixed_*/models/pacman_best.zip',
        f'{base_dir}/mixed_*/models/pacman_v*.zip',
        f'{base_dir}/random_only_*/models/pacman*.zip',
        f'{base_dir}/ppo_*/models/pacman*.zip',
        'models/ppo_*/best/best_model.zip',
        'models/**/pacman*.zip',
    ]
    
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            # Create a short name from the path
            name = path.replace('training_output/', '').replace('models/', '').replace('.zip', '')
            name = name.replace('/pacman_best', '').replace('/best_model', '')
            name = name.strip('/')
            if name not in models:
                models[name] = path
    
    return models


def load_ghost_models(ghost1_path, ghost2_path):
    """Load ghost models and return as dict."""
    if ghost1_path is None or ghost2_path is None:
        return None
    
    try:
        ghost1 = DQN.load(ghost1_path)
        ghost2 = DQN.load(ghost2_path)
        return {1: ghost1, 2: ghost2}
    except Exception as e:
        print(f"Warning: Could not load ghost models: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Compare Pac-Man models')
    parser.add_argument('--pacman', action='append', help='Path to Pac-Man model(s)')
    parser.add_argument('--pacman-dir', help='Directory containing Pac-Man models')
    parser.add_argument('--ghost1', help='Path to ghost 1 model')
    parser.add_argument('--ghost2', help='Path to ghost 2 model')
    parser.add_argument('--games', type=int, default=500, help='Games per evaluation')
    parser.add_argument('--layout', default='mediumClassic', help='Layout name')
    parser.add_argument('--all', action='store_true', help='Evaluate all models')
    parser.add_argument('--random-only', action='store_true', help='Only eval vs random')
    parser.add_argument('--trained-only', action='store_true', help='Only eval vs trained')
    args = parser.parse_args()
    
    # Collect models to evaluate
    models = {}
    
    if args.all:
        models = find_pacman_models()
        if not models:
            print("No models found in training_output/")
            return
    elif args.pacman_dir:
        for path in glob.glob(os.path.join(args.pacman_dir, 'pacman*.zip')):
            name = os.path.basename(path).replace('.zip', '')
            models[name] = path
    elif args.pacman:
        for path in args.pacman:
            name = os.path.basename(path).replace('.zip', '')
            models[name] = path
    else:
        # Default: compare key models
        models = {
            'Base Model': 'models/ppo_mediumClassic_20251129_005155/best/best_model.zip',
        }
        # Add latest mixed and random-only if they exist
        mixed = sorted(glob.glob('training_output/mixed_*/models/pacman_best.zip'))
        if mixed:
            models['Mixed (latest)'] = mixed[-1]
        random_only = sorted(glob.glob('training_output/random_only_*/models/pacman*.zip'))
        if random_only:
            models['Random-Only (latest)'] = random_only[-1]
    
    if not models:
        print("No models specified. Use --pacman, --pacman-dir, or --all")
        return
    
    # Load ghost models
    ghost1_path = args.ghost1
    ghost2_path = args.ghost2
    
    if ghost1_path is None or ghost2_path is None:
        ghost1_path, ghost2_path = find_latest_ghost_models()
    
    ghost_models = None
    if not args.random_only:
        if ghost1_path and ghost2_path:
            print(f"Loading ghost models:")
            print(f"  Ghost 1: {ghost1_path}")
            print(f"  Ghost 2: {ghost2_path}")
            ghost_models = load_ghost_models(ghost1_path, ghost2_path)
        else:
            print("Warning: No trained ghost models found. Only evaluating vs random.")
            args.random_only = True
    
    # Evaluate each model
    results = {}
    
    for name, path in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"Path:  {path}")
        print(f"{'='*60}")
        
        try:
            model = MaskablePPO.load(path)
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue
        
        results[name] = {}
        
        if not args.trained_only:
            print(f"\n  vs RANDOM ghosts ({args.games} games):")
            res = evaluate(model, args.layout, None, args.games)
            results[name]['vs_random'] = res
            print(f"  -> Win rate: {res['win_rate']*100:.1f}%, Avg score: {res['avg_score']:.1f}")
        
        if not args.random_only and ghost_models:
            print(f"\n  vs TRAINED ghosts ({args.games} games):")
            res = evaluate(model, args.layout, ghost_models, args.games)
            results[name]['vs_trained'] = res
            print(f"  -> Win rate: {res['win_rate']*100:.1f}%, Avg score: {res['avg_score']:.1f}")
    
    # Print summary table
    print("\n")
    print("=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)
    
    headers = ['Model']
    if not args.trained_only:
        headers.append('vs Random')
    if not args.random_only and ghost_models:
        headers.append('vs Trained')
    
    print(f"{headers[0]:<35}", end='')
    for h in headers[1:]:
        print(f"{h:>15}", end='')
    print()
    print("-" * 70)
    
    for name, res in results.items():
        # Truncate long names
        display_name = name[:33] + '..' if len(name) > 35 else name
        print(f"{display_name:<35}", end='')
        
        if 'vs_random' in res:
            print(f"{res['vs_random']['win_rate']*100:>14.1f}%", end='')
        if 'vs_trained' in res:
            print(f"{res['vs_trained']['win_rate']*100:>14.1f}%", end='')
        print()
    
    print("=" * 70)


if __name__ == "__main__":
    main()
