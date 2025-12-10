PROJECT: Adversarial Multi-Agent Reinforcement Learning - Pac-Man vs. Ghosts
===============================================================================

FILE DESCRIPTIONS:

Main Deliverables:
--------------
project.ipynb 
project.html

Models (Sample Data):
--------------------
models/pacman_v0.zip - untrained Pac-Man baseline (random actions)
models/pacman_baseline.zip - trained Pac-Man (2M steps, 85% win rate vs random)
models/pacman_v10.zip  - final Pac-Man after 10 adversarial rounds
models/ghost_1_v10.zip  - trained ghost agent #1
models/ghost_2_v10.zip  - trained ghost agent #2 
models/vecnormalize.pkl - observation normalization statistics

Source Code (src/):
------------------
Core Training Scripts:
  train_ppo.py     - single-agent PPO training for Pac-Man baseline
  train_mixed.py  - adversarial multi-agent training pipeline with ghost pretraining

Evaluation & Visualization:
  evaluate_comparison.py   - comprehensive evaluation toolkit comparing training versions
  visualize_agents.py     - real-time game visualization with optional video recording

Custom Environment:
  gym_env.py                  - openAI Gym wrapper for Pac-Man enabling RL training

Game Engine (adapted from UC Berkeley CS188):
  game.py               - core game state management and game logic
  pacman.py               - main game controller and simulation loop
  ghostAgents.py          - ghost agent base classes and random policy
  layout.py             - maze layout parser and game board handler
  graphicsDisplay.py      - GUI rendering system for visualization
  graphicsUtils.py 
  util.py  
  layouts/               - directory containing maze layout files
    mediumClassic.lay     - primary maze used for training and evaluation
    smallClassic.lay      - smaller test maze for quick experiments
    [others]            - additional maze layouts

Key contributions:
-----------------
1. Ghost Pretraining Solution: Addressed "smart teacher problem" by pretraining 
   ghosts against random Pac-Man before adversarial training
2. Stable Multi-Agent Training: Implemented balanced training schedule preventing
   agent collapse (150k pretrain + 80k steps per round × 10 rounds)

TRAINING PIPELINE:
-----------------
Step 1: python src/train_ppo.py --timesteps 2000000 --num-envs 16
Step 2: python src/train_mixed.py --pacman [baseline] --rounds 10
Step 3: python src/evaluate_comparison.py --version 10

RESULTS:
----------
Untrained:  ~5% win rate vs random ghosts
Baseline:   ~85% win rate vs random ghosts
Final (v10): ~65% win rate vs trained ghosts (competitive equilibrium achieved)

DEPENDENCIES: stable-baselines3, sb3-contrib, gymnasium, numpy, matplotlib, torch