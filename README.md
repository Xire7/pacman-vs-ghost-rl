# Pac-Man vs Ghost RL

Adversarial reinforcement learning project training Pac-Man (MaskablePPO) against Ghost agents (DQN) using the Berkeley Pac-Man game engine.

## Features

- **Symmetric Warmup**: Both Pac-Man and Ghosts learn fundamentals before facing each other
- **Curriculum Learning**: Gradually increases difficulty during adversarial training
- **Smart Ghost Features**: BFS pathfinding, flanking detection, escape route counting
- **Action Masking**: Legal move enforcement for both agents

## Quick Start

```bash
cd game_environment

# Adversarial training with symmetric warmup (recommended)
python train.py --mode adversarial --iterations 10

# Pac-Man only (vs random ghosts)
python train.py --mode pacman --timesteps 500000

# Evaluate trained models
python train.py --eval --training-dir training/mediumClassic_YYYYMMDD_HHMMSS

# Watch games
python train.py --render --training-dir training/mediumClassic_YYYYMMDD_HHMMSS --games 5
```

## Training Modes

### Adversarial Training (Recommended)
```bash
python train.py --mode adversarial \
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

### Pac-Man Only
```bash
python train.py --mode pacman --timesteps 500000 --ghost-type random
```

## Project Structure

```
game_environment/
├── train.py              # Unified training script
├── gym_env.py            # Pac-Man Gymnasium environment
├── ghost_agent.py        # Ghost DQN environment
├── state_extractor.py    # Observation extraction (33-dim ghost, 33-dim Pac-Man)
├── training_utils.py     # Callbacks, evaluation, rendering
├── layouts/              # Game layouts
└── training/             # Saved models and logs
```

## Algorithms

| Agent | Algorithm | Observation | Actions |
|-------|-----------|-------------|---------|
| **Pac-Man** | MaskablePPO | 33-dim (position, ghosts, food, danger) | 5 (N, S, E, W, Stop) |
| **Ghosts** | DQN | 33-dim (position, Pac-Man, walls, BFS, flanking) | 4 (N, S, E, W) |

## Ghost Intelligence Features

The ghost observation includes:
- **BFS Distance**: Actual shortest path (not just Manhattan distance)
- **Best Direction**: Optimal move using pathfinding
- **Flanking Score**: Detects when ghosts approach from opposite sides
- **Escape Routes**: Count of Pac-Man's available moves
- **Pac-Man Direction**: Predict where Pac-Man is heading

## Command Reference

```bash
# Full adversarial training
python train.py --mode adversarial --iterations 10

# Skip warmup (pure self-play)
python train.py --mode adversarial --iterations 15 \
    --pacman-warmup-timesteps 0 --ghost-warmup-timesteps 0

# Evaluate vs random and trained ghosts
python train.py --eval --training-dir <path> --episodes 100

# Render with trained ghosts
python train.py --render --training-dir <path> --games 5

# Render vs random ghosts
python train.py --render --training-dir <path> --games 5 --vs-random
```

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended, ~3.5x faster)
- stable-baselines3
- sb3-contrib (for MaskablePPO)
- gymnasium

