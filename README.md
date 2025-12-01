# Pac-Man vs Ghost RL

Adversarial reinforcement learning project training MaskablePPO Pac-Man agents against DQN Ghost agents using the Berkeley Pac-Man game engine.

## Quick Start

### 1. Train Initial Pac-Man (vs Random Ghosts)
```bash
cd game_environment
python train_ppo.py --layout mediumClassic --timesteps 2000000
```

### 2. Adversarial Training (vs Learned Ghosts)
```bash
python train_mixed.py --model-path models/.../final_model.zip --rounds 10
```

### 3. Evaluate
```bash
# vs Random Ghosts
python train_mixed.py --eval --model-path training_output/.../pacman_best.zip --episodes 500

# vs Trained Ghosts  
python train_mixed.py --eval --model-path training_output/.../pacman_best.zip \
    --ghost1 training_output/.../ghost_1_v10.zip \
    --ghost2 training_output/.../ghost_2_v10.zip \
    --episodes 500

# With visualization
python train_mixed.py --eval --render --episodes 5 --model-path ...
```

## Project Structure

```
game_environment/
├── train_ppo.py          # Initial Pac-Man training (vs random ghosts)
├── train_mixed.py        # Adversarial training + evaluation
├── training_utils.py     # Shared utilities
├── gym_env.py            # Gymnasium environment for Pac-Man
├── ghost_agent.py        # Ghost training environment
├── state_extractor.py    # Observation extraction
├── layouts/              # Game layouts
├── training_output/      # Saved models and logs
└── archive/              # Old/experimental scripts
```

## Key Files

| File | Purpose |
|------|---------|
| `train_ppo.py` | Train Pac-Man with MaskablePPO against random ghosts |
| `train_mixed.py` | Adversarial self-play training + evaluation |
| `gym_env.py` | Pac-Man Gymnasium environment with action masking |
| `ghost_agent.py` | Ghost DQN training environment |

## Algorithms

- **Pac-Man**: MaskablePPO (sb3-contrib) with action masking for legal moves
- **Ghosts**: DQN (stable-baselines3) with chase/flee reward shaping

