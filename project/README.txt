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
models/pacman_baseline.zip - trained Pac-Man (2M steps, 94% win rate vs random)
models/pacman_v7.zip  - final Pac-Man after 7 adversarial rounds
models/ghost_1_v7.zip  - trained ghost agent #1 (curriculum + adversarial)
models/ghost_2_v7.zip  - trained ghost agent #2 (curriculum + adversarial)
models/ghost_3_v7.zip  - trained ghost agent #3 (curriculum + adversarial)
models/ghost_4_v7.zip  - trained ghost agent #4 (curriculum + adversarial)
models/ppo_vecnormalize/vecnormalize.pkl - normalization stats for baseline Pac-Man
models/mixed_vecnormalize/vecnormalize.pkl - normalization stats for adversarial Pac-Man

Source Code (src/):
------------------
Core Training Scripts:
  train_ppo.py     - single-agent PPO training for Pac-Man baseline
  train_mixed.py   - curriculum learning + adversarial multi-agent training pipeline
  ghost_agent.py   - independent ghost training environment with enhanced rewards

Evaluation & Visualization:
  evaluate_comparison.py   - comprehensive evaluation toolkit with moving average plots
  visualize_agents.py      - real-time game visualization with optional video recording
  state_extractor.py       - feature extraction utilities for agents

Custom Environment:
  gym_env.py                  - OpenAI Gym wrapper for Pac-Man enabling RL training

Game Engine (adapted from UC Berkeley CS188):
  game.py               - core game state management and game logic
  pacman.py             - main game controller and simulation loop
  pacmanAgents.py       - Pac-Man agent implementations
  ghostAgents.py        - ghost agent base classes and scripted policies
  layout.py             - maze layout parser and game board handler
  graphicsDisplay.py    - GUI rendering system for visualization
  graphicsUtils.py      - graphics utility functions
  textDisplay.py        - text-based display for headless training
  util.py               - general utility functions
  layouts/              - directory containing maze layout files
    mediumClassic.lay   - primary maze used for training and evaluation
    smallClassic.lay    - smaller test maze for quick experiments
    [others]            - additional maze layouts

Key Contributions:
-----------------
1. Curriculum Learning Solution: Addressed "smart teacher problem" with 3-stage 
   progressive training (random → fleeing → smart Pac-Man) teaching ghosts actual
   pursuit behavior before adversarial training
   
2. Enhanced Reward Shaping: Implemented proximity-based rewards (50x stronger), 
   chase rewards (10x multiplier), and direction alignment rewards to encourage
   active pursuit and shadowing behavior
   
3. VecNormalize Discovery: Identified critical importance of matching observation
   normalization statistics to training distribution (90% vs 40% performance gap)
   
4. Stable Multi-Agent Training: Implemented balanced curriculum + adversarial 
   schedule preventing agent collapse (150k curriculum + 120k steps per round × 7 rounds)

TRAINING PIPELINE:
-----------------
Step 1: python src/train_ppo.py --timesteps 2000000 --num-envs 16 --normalize --lr-decay
Step 2: python src/train_mixed.py --pacman [baseline] --rounds 7 --ghost-pretrain-steps 200000
Step 3: python src/evaluate_comparison.py --version 7 --episodes 100 --plot

CURRICULUM LEARNING STAGES (Ghost Pretraining):
-----------------------------------------------
Stage 1 (30% - 60k steps):  Random Pac-Man   → Learn basic movement
Stage 2 (50% - 100k steps):  Fleeing Pac-Man  → Learn pursuit & chase strategies
Stage 3 (20% - 40k steps):  Smart Pac-Man    → Adapt to trained opponent

RESULTS:
----------
Untrained:   ~0-5% win rate vs random ghosts
Baseline:    ~94% win rate vs random ghosts
Final (v7):  ~40-60% win rate vs trained ghosts (competitive equilibrium achieved)
             ~90%+ win rate vs random ghosts (baseline performance maintained)

Ghost Behaviors Learned:
- Active pursuit and chasing of Pac-Man
- Close shadowing and following behavior  
- Pressure application (staying near Pac-Man)
- Multi-ghost convergence on target

DEPENDENCIES: 
-------------
stable-baselines3, sb3-contrib, gymnasium, numpy, matplotlib, torch, tensorboard