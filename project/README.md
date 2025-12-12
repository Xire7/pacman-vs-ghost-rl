# Adversarial Multi-Agent Reinforcement Learning: Pac-Man vs. Ghosts

Note that the files in src are solely used to run this jupyter notebook to follow project formatting directions. To train the models, please cd into **game_environment** and run the appropriate commands on that directory's train_ppo.py & train_mixed.py (details are listed in README.md at the root directory). While it may be possible to train in this directory (project/src) as we've copied the files onto it, there can be uncaught bugs.

## File Descriptions

### Main Deliverables
- `project.ipynb`
- `project.html`

### Models (Sample Data)
| File | Description |
|------|-------------|
| `models/ppo_pacman_v0.zip` | Untrained Pac-Man baseline (random actions) |
| `models/ppo_pacman_best.zip` | Trained Pac-Man (2M steps, 94% win rate vs random) |
| `models/mixed_pacman_best.zip` | Final Pac-Man after 7 adversarial rounds |
| `models/ghost_1_v7.zip` | Trained ghost agent #1 (curriculum + adversarial) |
| `models/ghost_2_v7.zip` | Trained ghost agent #2 (curriculum + adversarial) |
| `models/ppo_vecnormalize/vecnormalize.pkl` | Normalization stats for baseline Pac-Man |
| `models/mixed_vecnormalize/vecnormalize.pkl` | Normalization stats for adversarial Pac-Man |

### Source Code (`src/`)

**Core Training Scripts:**
- `train_ppo.py` - Single-agent PPO training for Pac-Man baseline
- `train_mixed.py` - Curriculum learning + adversarial multi-agent training pipeline
- `ghost_agent.py` - Independent ghost training environment with enhanced rewards

**Evaluation & Visualization:**
- `evaluate_comparison.py` - Comprehensive evaluation toolkit with moving average plots
- `visualize_agents.py` - Real-time game visualization with optional video recording
- `state_extractor.py` - Feature extraction utilities for agents

**Custom Environment:**
- `gym_env.py` - OpenAI Gym wrapper for Pac-Man enabling RL training

**Game Engine (adapted from UC Berkeley CS188):**
- `game.py` - Core game state management and game logic
- `pacman.py` - Main game controller and simulation loop
- `pacmanAgents.py` - Pac-Man agent implementations
- `ghostAgents.py` - Ghost agent base classes and scripted policies
- `layout.py` - Maze layout parser and game board handler
- `graphicsDisplay.py` - GUI rendering system for visualization
- `graphicsUtils.py` - Graphics utility functions
- `textDisplay.py` - Text-based display for headless training
- `util.py` - General utility functions
- `layouts/` - Directory containing maze layout files
  - `mediumClassic.lay` - Primary maze used for training and evaluation
  - `smallClassic.lay` - Smaller test maze for quick experiments

## Key Contributions

1. **Curriculum Learning Solution:** Addressed "smart teacher problem" with 3-stage progressive training (random -> fleeing -> smart Pac-Man) teaching ghosts actual pursuit behavior before adversarial training

2. **Enhanced Reward Shaping:** Implemented proximity-based rewards (50x stronger), chase rewards (10x multiplier), and direction alignment rewards to encourage active pursuit and shadowing behavior

3. **VecNormalize Discovery:** Identified critical importance of matching observation normalization statistics to training distribution (90% vs 40% performance gap)

4. **Stable Multi-Agent Training:** Implemented balanced curriculum + adversarial schedule preventing agent collapse (150k curriculum + 120k steps per round x 7 rounds)

## Training Pipeline

```bash
# Step 0: CD into game_environment from the root directory
cd game_environment # if you are in project/ do cd ../game_environment

# Step 1: Train Pac-Man baseline (in game_environment)
python game_environment/train_ppo.py --timesteps 2000000 --num-envs 16 --normalize --lr-decay

# This will produce an output in game_environment/models/ppo_layoutname_date, use best/best_model.zip for step 2

# Step 2: Curriculum + adversarial training (in game_environment)
python game_environment/train_mixed.py --pacman game_environment/models/ppo_layoutName_datetime/best/best_model.zip --rounds 7 --ghost-pretrain-steps 200000

# Step 3: Evaluate results (in game_environment)
python game_environment/visualize_agents.py --pacman-path game_environment/training_output/[mixed_####_####]/models/pacman_best.zip --ghost-dir training_output/[mixed_####_####]/models --ghost-version 7 --episodes 5

```

## Curriculum Learning Stages (Ghost Pretraining)

| Stage | Steps | Opponent | Goal |
|-------|-------|----------|------|
| Stage 1 (30%) | 60k | Random Pac-Man | Learn basic movement |
| Stage 2 (50%) | 100k | Fleeing Pac-Man | Learn pursuit & chase strategies |
| Stage 3 (20%) | 40k | Smart Pac-Man | Adapt to trained opponent |

## Results

| Configuration | Win Rate |
|---------------|----------|
| Untrained | ~0-5% vs random ghosts |
| Baseline | ~94% vs random ghosts |
| Final (v7) | ~40-60% vs trained ghosts (competitive equilibrium) |
| Final (v7) | ~90%+ vs random ghosts (baseline maintained) |

**Ghost Behaviors Learned:**
- Active pursuit and chasing of Pac-Man
- Close shadowing and following behavior
- Pressure application (staying near Pac-Man)
- Multi-ghost convergence on target

## Dependencies

```
stable-baselines3
sb3-contrib
gymnasium
numpy
matplotlib
torch
tensorboard
```
