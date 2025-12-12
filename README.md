# Pac-Man vs Ghost RL

Adversarial reinforcement learning project training PPO-based Pac-Man agents against DQN-based Ghost agents using the Berkeley Pac-Man game engine.

![Pacman vs Ghost Adversarial Training Thumbnail](thumbnail\pacmanvsghostRLThumbnail.png)

## Quick Start
```bash
pip install -r requirements.txt
```

## Option 1: View Pre-Trained Results
```bash
cd project/
# Open project.ipynb in Jupyter, or view project.html for pre-run results
# You could train the models here, but normally we trained it from Option 2 (game_environment folder)

```

## Option 2: Train Your Own Models
```bash
cd game_environment/

# 1. Train baseline Pac-Man (2M steps)
Step 1: python src/train_ppo.py --timesteps 2000000 --num-envs 16 --normalize --lr-decay

# 2. Train adversarial agents
python train_mixed.py --pacman models/best/[baseline_path_here].zip --rounds 7 --ghost-pretrain-steps 200000

# 3. Visualize trained agents
python visualize_agents.py --pacman-path training_output/[mixed_####_####]/models/pacman_best.zip --ghost-dir training_output/[mixed_####_####]/models --ghost-version 7 --episodes 5
```

## Documentation

See `project/README.txt` for detailed file descriptions and project structure.