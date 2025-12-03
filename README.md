# Pac-Man vs Ghost RL

Adversarial reinforcement learning framework for training Pac-Man (MaskablePPO) against Ghost agents (DQN) using the Berkeley Pac-Man game engine.

## Features

- **Symmetric Warmup**: Both Pac-Man and Ghosts learn fundamentals before facing each other
- **Curriculum Learning**: Gradually increases difficulty during adversarial training
- **Smart Ghost Features**: BFS pathfinding, flanking detection, escape route counting
- **Action Masking**: Legal move enforcement for both agents
- **Cross-Platform**: Models trained on Windows work on Linux and vice versa

## Installation

```bash
# Clone the repository
git clone https://github.com/Xire7/pacman-vs-ghost-rl.git
cd pacman-vs-ghost-rl

# Install as a package (recommended)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### Using the Package (Recommended)

```python
from pacman_rl import Trainer, PacmanEnv

# Train adversarially
trainer = Trainer(layout_name='mediumClassic')
trainer.train_adversarial(iterations=5)

# Evaluate
results = trainer.evaluate(episodes=100)
print(f"Win rate: {results['vs_trained']['win_rate']:.2%}")

# Render games
trainer.render(games=3)
```

### Using the CLI

```bash
# Adversarial training (recommended)
python -m pacman_rl.scripts.train --mode adversarial --iterations 10

# Pac-Man only (vs random ghosts)
python -m pacman_rl.scripts.train --mode pacman --timesteps 500000

# Evaluate trained models
python -m pacman_rl.scripts.train --eval --training-dir training/mediumClassic_*

# Watch games
python -m pacman_rl.scripts.train --render --training-dir training/mediumClassic_* --games 5
```

### Legacy CLI (game_environment)

```bash
cd game_environment
python train.py --mode adversarial --iterations 10
```

## Training Modes

### Adversarial Training (Recommended)

```bash
python -m pacman_rl.scripts.train --mode adversarial \
    --iterations 10 \
    --pacman-warmup-timesteps 250000 \
    --ghost-warmup-timesteps 150000 \
    --pacman-timesteps 200000 \
    --ghost-timesteps 150000
```

Training phases:
1. **Pac-Man Warmup**: Learn navigation vs random ghosts
2. **Ghost Warmup**: Learn chasing vs random Pac-Man  
3. **Adversarial**: Alternating training with curriculum learning

### Resume Training

```bash
# Resume from iteration 3
python -m pacman_rl.scripts.train --mode adversarial \
    --resume training/mediumClassic_20251202_* \
    --iterations 10

# Auto-detects last completed iteration
```

## Project Structure

```
pacman-vs-ghost-rl/
├── pacman_rl/                 # Main package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Hyperparameters and constants
│   ├── envs/                 # Gymnasium environments
│   │   ├── pacman_env.py    # PacmanEnv with action masking
│   │   └── ghost_env.py     # IndependentGhostEnv for DQN
│   ├── agents/               # Agent utilities
│   │   └── state_extractor.py  # 33-dim observation extraction
│   ├── training/             # Training infrastructure
│   │   ├── trainer.py       # Unified Trainer class
│   │   ├── callbacks.py     # WinRateCallback, CatchRateCallback
│   │   └── utils.py         # Evaluation, rendering, model loading
│   ├── berkeley/             # UC Berkeley Pac-Man engine (vendored)
│   ├── layouts/              # Game layout files
│   └── scripts/              # CLI entry points
│       └── train.py         # Main training script
├── game_environment/         # Legacy standalone version
├── pyproject.toml           # Package configuration
└── requirements.txt         # Dependencies
```

## Algorithms

| Agent | Algorithm | Observation | Actions |
|-------|-----------|-------------|---------|
| **Pac-Man** | MaskablePPO | 33-dim (position, ghosts, food, danger) | 5 (N, S, E, W, Stop) |
| **Ghosts** | DQN | 33-dim (position, Pac-Man, walls, BFS, flanking) | 4 (N, S, E, W) |

## Ghost Intelligence Features

The ghost observation (33-dimensional) includes:

| Feature | Indices | Description |
|---------|---------|-------------|
| Position | 0-1 | Ghost's normalized (x, y) position |
| Scared State | 2-3 | Is scared, timer normalized |
| Pac-Man Position | 4-5 | Pac-Man's normalized position |
| Relative Position | 6-7 | Vector to Pac-Man |
| Distance | 8 | Manhattan distance to Pac-Man |
| Wall Sensors | 9-12 | Walls in N/S/E/W directions |
| Legal Actions | 13-16 | Valid moves |
| Other Ghosts | 17-20 | Distances to other ghosts |
| Direction to Pac-Man | 21-24 | One-hot encoded |
| **Pac-Man Direction** | 25-28 | Predict where Pac-Man is heading |
| **Escape Routes** | 29 | Count of Pac-Man's available moves |
| **Flanking Score** | 30 | Detects coordinated attacks |
| **BFS Ratio** | 31 | Actual path / Manhattan distance |
| **Best Direction** | 32 | Optimal move using BFS |

## Configuration

Hyperparameters are centralized in `pacman_rl/config.py`:

```python
from pacman_rl.config import (
    NetworkConfig,
    PacmanTrainingConfig,
    GhostTrainingConfig,
    GhostRewardConfig,
)

# Customize network architecture
network = NetworkConfig(
    pacman_hidden_sizes=[256, 256],
    ghost_hidden_sizes=[256, 256],
)

# Customize training
pacman_config = PacmanTrainingConfig(
    initial_lr=3e-4,
    gamma=0.995,
    n_steps=512,
)
```

## Command Reference

```bash
# Full adversarial training
python -m pacman_rl.scripts.train --mode adversarial --iterations 10

# Skip warmup (pure self-play)
python -m pacman_rl.scripts.train --mode adversarial --iterations 15 \
    --pacman-warmup-timesteps 0 --ghost-warmup-timesteps 0

# Evaluate vs random and trained ghosts
python -m pacman_rl.scripts.train --eval --training-dir <path> --episodes 100

# Render with trained ghosts
python -m pacman_rl.scripts.train --render --training-dir <path> --games 5

# Render vs random ghosts
python -m pacman_rl.scripts.train --render --training-dir <path> --games 5 --vs-random

# Use specific layout
python -m pacman_rl.scripts.train --mode adversarial --layout smallClassic

# Use CPU instead of GPU
python -m pacman_rl.scripts.train --mode adversarial --device cpu
```

## API Reference

### PacmanEnv

```python
from pacman_rl.envs import PacmanEnv

env = PacmanEnv(
    layout_name='mediumClassic',
    ghost_type='random',          # 'random', 'directional', 'mixed'
    max_steps=500,
    render_mode='human',          # or None
    ghost_policies={1: model1},   # Optional trained ghosts
)

obs, info = env.reset()
action_mask = env.action_masks()
obs, reward, terminated, truncated, info = env.step(action)
```

### IndependentGhostEnv

```python
from pacman_rl.envs import IndependentGhostEnv

env = IndependentGhostEnv(
    ghost_index=1,                # 1-based ghost index
    layout_name='mediumClassic',
    pacman_policy=pacman_model,   # Trained Pac-Man
    other_ghost_policies={2: g2}, # Other trained ghosts
)

obs, info = env.reset()
action_mask = env.action_masks()
obs, reward, terminated, truncated, info = env.step(action)
```

### Trainer

```python
from pacman_rl.training import Trainer

trainer = Trainer(
    layout_name='mediumClassic',
    output_dir='my_training',
    device='cuda',
)

# Train Pac-Man only
trainer.train_pacman(timesteps=500000)

# Train single ghost
trainer.train_ghost(ghost_index=1, timesteps=100000)

# Full adversarial training
trainer.train_adversarial(iterations=10)

# Save/Load
trainer.save()
trainer.load('training/mediumClassic_*')

# Evaluate
results = trainer.evaluate(episodes=100)

# Render
trainer.render(games=3)
```

## License

This project uses the Berkeley Pac-Man game engine for educational purposes.
Original source: http://ai.berkeley.edu/project_overview.html

