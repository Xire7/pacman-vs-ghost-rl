#!/usr/bin/env python3
"""
Quick launcher for common curriculum training workflows.

Usage:
    python quick_train.py --preset quick-test
    python quick_train.py --preset full
    python quick_train.py --preset large-map
"""

import argparse
import subprocess
import sys

PRESETS = {
    'quick-test': {
        'description': 'Quick test run (~30 min on CPU)',
        'args': [
            '--full-curriculum',
            '--layout', 'mediumGrid',  # Has 1 ghost, but we'll train just 1
            '--num-ghosts', '1',  # Use 1 ghost for quick test
            '--stage1-timesteps', '50000',
            '--stage2-timesteps', '100000',
            '--adversarial-rounds', '4',
            '--ghost-initial-steps', '20000',
            '--ghost-refine-steps', '15000',
            '--pacman-refine-steps', '20000',
            '--num-envs', '4'
        ]
    },
    
    'full': {
        'description': 'Full training pipeline (recommended, ~6-8 hours)',
        'args': [
            '--full-curriculum',
            '--layout', 'mediumClassic',
            '--num-ghosts', '4',
            '--stage1-timesteps', '200000',
            '--stage2-timesteps', '500000',
            '--adversarial-rounds', '10',
            '--ghost-initial-steps', '50000',
            '--ghost-refine-steps', '30000',
            '--pacman-refine-steps', '50000',
            '--num-envs', '4'
        ]
    },
    
    'conservative': {
        'description': 'Conservative/safe training (slower but more stable, ~10-12 hours)',
        'args': [
            '--full-curriculum',
            '--layout', 'mediumClassic',
            '--num-ghosts', '4',
            '--stage1-timesteps', '300000',
            '--stage2-timesteps', '800000',
            '--adversarial-rounds', '16',
            '--ghost-initial-steps', '60000',
            '--ghost-refine-steps', '40000',
            '--pacman-refine-steps', '60000',
            '--num-envs', '4',
            '--learning-rate', '2e-4'
        ]
    },
    
    'large-map': {
        'description': 'Training on larger/trickier map (~8-10 hours)',
        'args': [
            '--full-curriculum',
            '--layout', 'trickyClassic',
            '--num-ghosts', '4',
            '--stage1-timesteps', '250000',
            '--stage2-timesteps', '600000',
            '--adversarial-rounds', '12',
            '--ghost-initial-steps', '60000',
            '--ghost-refine-steps', '35000',
            '--pacman-refine-steps', '55000',
            '--num-envs', '4'
        ]
    },
    
    'fast-experiment': {
        'description': 'Minimal training for experimentation (~1 hour)',
        'args': [
            '--full-curriculum',
            '--layout', 'smallGrid',
            '--num-ghosts', '1',
            '--stage1-timesteps', '30000',
            '--stage2-timesteps', '60000',
            '--adversarial-rounds', '4',
            '--ghost-initial-steps', '15000',
            '--ghost-refine-steps', '10000',
            '--pacman-refine-steps', '15000',
            '--num-envs', '2'
        ]
    },
    
    'resume-stage2': {
        'description': 'Resume from Stage 1 (provide --pacman-model)',
        'args': [
            '--skip-to', 'random_ghosts',
            '--layout', 'mediumClassic',
            '--num-ghosts', '4',
            '--stage2-timesteps', '500000',
            '--num-envs', '4'
        ],
        'requires_model': True
    },
    
    'resume-adversarial': {
        'description': 'Resume from Stage 2 (provide --pacman-model)',
        'args': [
            '--skip-to', 'adversarial',
            '--layout', 'mediumClassic',
            '--num-ghosts', '4',
            '--adversarial-rounds', '10',
            '--ghost-initial-steps', '50000',
            '--ghost-refine-steps', '30000',
            '--pacman-refine-steps', '50000',
            '--num-envs', '4'
        ],
        'requires_model': True
    }
}


def list_presets():
    """Display all available presets."""
    print("\n" + "="*70)
    print("AVAILABLE TRAINING PRESETS")
    print("="*70)
    
    for name, config in PRESETS.items():
        print(f"\n{name}:")
        print(f"  {config['description']}")
        if config.get('requires_model'):
            print(f"  ⚠️  Requires: --pacman-model <path>")
    
    print("\n" + "="*70)
    print("\nUsage:")
    print(f"  python {sys.argv[0]} --preset <preset_name>")
    print(f"  python {sys.argv[0]} --preset resume-stage2 --pacman-model path/to/model")
    print("\nFor custom settings:")
    print(f"  python train_curriculum.py --help")
    print("="*70 + "\n")


def run_preset(preset_name, extra_args=None):
    """Run training with a preset configuration."""
    if preset_name not in PRESETS:
        print(f"Error: Unknown preset '{preset_name}'")
        print(f"Available presets: {', '.join(PRESETS.keys())}")
        print(f"Use --list to see descriptions")
        sys.exit(1)
    
    config = PRESETS[preset_name]
    
    # Check if model required
    if config.get('requires_model') and (not extra_args or '--pacman-model' not in extra_args):
        print(f"Error: Preset '{preset_name}' requires --pacman-model argument")
        print(f"Usage: python {sys.argv[0]} --preset {preset_name} --pacman-model path/to/model")
        sys.exit(1)
    
    # Build command
    cmd = ['python', 'train_curriculum.py'] + config['args']
    
    if extra_args:
        cmd.extend(extra_args)
    
    # Display configuration
    print("\n" + "="*70)
    print(f"RUNNING PRESET: {preset_name}")
    print("="*70)
    print(f"Description: {config['description']}")
    print(f"\nCommand: {' '.join(cmd)}")
    print("="*70 + "\n")
    
    # Confirm
    response = input("Start training? (y/n): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n\nTraining failed with error code {e.returncode}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        description='Quick launcher for curriculum training presets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python quick_train.py --preset quick-test
  
  # Full recommended training
  python quick_train.py --preset full
  
  # Resume from checkpoint
  python quick_train.py --preset resume-stage2 --pacman-model path/to/model
  
  # See all presets
  python quick_train.py --list
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='List all available presets')
    parser.add_argument('--preset', type=str,
                       help='Preset configuration to use')
    parser.add_argument('--pacman-model', type=str,
                       help='Path to Pac-Man model (for resume presets)')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory')
    
    args, unknown_args = parser.parse_known_args()
    
    if args.list:
        list_presets()
        sys.exit(0)
    
    if not args.preset:
        print("Error: --preset required (or use --list to see options)")
        parser.print_help()
        sys.exit(1)
    
    # Build extra args
    extra_args = []
    if args.pacman_model:
        extra_args.extend(['--pacman-model', args.pacman_model])
    if args.output_dir:
        extra_args.extend(['--output-dir', args.output_dir])
    if unknown_args:
        extra_args.extend(unknown_args)
    
    run_preset(args.preset, extra_args if extra_args else None)


if __name__ == '__main__':
    main()