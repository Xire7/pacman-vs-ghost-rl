# Curriculum Learning for Pac-Man RL

This curriculum learning pipeline trains a Pac-Man agent that can outperform both scripted and learned ghost opponents through progressive difficulty scaling.

## 🎯 Training Philosophy

The curriculum addresses the cold-start problem where Pac-Man struggles to learn against even random ghosts by breaking training into three stages:

### Stage 1: Foundation (No Ghosts)
- **Goal**: Learn basic navigation and pellet collection
- **Environment**: Empty maze with only food pellets
- **Duration**: ~200,000 timesteps
- **Key Skills**: Efficient pathfinding, systematic maze coverage

### Stage 2: Evasion (Random/Scripted Ghosts)
- **Goal**: Learn to avoid threats while collecting pellets
- **Environment**: Full maze with scripted ghost opponents
- **Duration**: ~500,000 timesteps
- **Key Skills**: Threat avoidance, risk assessment, tactical retreats

### Stage 3: Adaptation (Adversarial Ghosts)
- **Goal**: Learn to counter adaptive opponents
- **Environment**: Learned DQN ghost agents that improve over time
- **Duration**: 10+ rounds of alternating training
- **Key Skills**: Strategic planning, exploiting opponent weaknesses

## 🚀 Quick Start

### Full Curriculum Training

Train from scratch through all three stages:

```bash
python train_curriculum.py --full-curriculum \
    --layout mediumClassic \
    --stage1-timesteps 200000 \
    --stage2-timesteps 500000 \
    --adversarial-rounds 10
```

### Resume from Checkpoint

Skip completed stages and resume training:

```bash
# Skip to Stage 2 with pre-trained model
python train_curriculum.py \
    --skip-to random_ghosts \
    --pacman-model curriculum_output/run_*/models/pacman_no_ghosts_final \
    --stage2-timesteps 500000

# Skip to Stage 3 (adversarial training)
python train_curriculum.py \
    --skip-to adversarial \
    --pacman-model curriculum_output/run_*/models/pacman_random_ghosts_final \
    --adversarial-rounds 10
```

### Quick Test Run

For testing the pipeline with minimal compute:

```bash
python train_curriculum.py --full-curriculum \
    --layout smallGrid \
    --stage1-timesteps 50000 \
    --stage2-timesteps 100000 \
    --adversarial-rounds 4 \
    --num-ghosts 2
```

## 📊 Evaluation

### Evaluate Single Model

Test a specific trained model:

```bash
# Against random ghosts
python evaluate_curriculum.py \
    --model curriculum_output/run_*/models/pacman_random_ghosts_final \
    --stage random_ghosts \
    --episodes 50

# Against adversarial ghosts
python evaluate_curriculum.py \
    --model curriculum_output/run_*/models/pacman_adversarial_v5 \
    --stage adversarial \
    --ghost-dir curriculum_output/run_*/models \
    --ghost-version 5 \
    --episodes 50
```

### Compare Curriculum Stages

Compare performance across all stages:

```bash
python evaluate_curriculum.py \
    --mode compare \
    --curriculum-dir curriculum_output/run_20241127_120000 \
    --episodes 50
```

### Visual Evaluation

Watch your agent play:

```bash
python evaluate_curriculum.py \
    --model path/to/model \
    --stage random_ghosts \
    --render \
    --episodes 5 \
    --frame-time 0.1
```

## ⚙️ Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--layout` | mediumClassic | Map layout (smallGrid, mediumGrid, mediumClassic, etc.) |
| `--num-ghosts` | 4 | Number of ghost opponents |
| `--stage1-timesteps` | 200000 | Training steps for Stage 1 (no ghosts) |
| `--stage2-timesteps` | 500000 | Training steps for Stage 2 (scripted ghosts) |
| `--adversarial-rounds` | 10 | Number of adversarial training rounds |
| `--ghost-initial-steps` | 50000 | Initial training steps for each ghost |
| `--ghost-refine-steps` | 30000 | Refinement steps for ghosts in later rounds |
| `--pacman-refine-steps` | 50000 | Refinement steps for Pac-Man in adversarial phase |

### PPO Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning-rate` | 3e-4 | Learning rate for PPO |
| `--num-envs` | 4 | Number of parallel environments |
| `--max-steps` | 500 | Maximum steps per episode |

## 📁 Output Structure

```
curriculum_output/
└── run_YYYYMMDD_HHMMSS/
    ├── config.json              # Training configuration
    ├── models/
    │   ├── pacman_no_ghosts_final.zip
    │   ├── pacman_random_ghosts_final.zip
    │   ├── pacman_adversarial_v1.zip
    │   ├── ghost_1_v1.zip
    │   └── ...
    ├── logs/                    # TensorBoard logs
    │   ├── no_ghosts/
    │   ├── random_ghosts/
    │   └── ghost_1/
    ├── checkpoints/             # Training checkpoints
    └── eval_results/            # Evaluation statistics
        ├── no_ghosts_results.json
        ├── random_ghosts_results.json
        └── curriculum_summary.json
```

## 📈 Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir curriculum_output/run_*/logs
```

Key metrics to watch:
- **ep_rew_mean**: Average episode reward (should increase)
- **ep_len_mean**: Average episode length
- **custom/win_rate**: Win rate (Stage 2+)
- **fps**: Training speed (frames per second)

## 🎮 Expected Performance

Based on the Stanford CS229 baseline and our testing:

| Stage | Environment | Expected Win Rate | Notes |
|-------|-------------|-------------------|-------|
| 1 | No ghosts | 100% | Should achieve perfect pellet collection |
| 2 | Random ghosts | 60-80% | Significant improvement over untrained (0-20%) |
| 3 | Adversarial ghosts (early) | 40-60% | Ghosts are learning too |
| 3 | Adversarial ghosts (late) | 50-70% | Arms race stabilizes |

## 🐛 Troubleshooting

### Issue: Agent not improving in Stage 2

**Symptoms**: Win rate stays near 0% after significant training

**Solutions**:
1. Increase Stage 1 timesteps to ensure solid foundation
2. Try easier layout (smallGrid instead of mediumClassic)
3. Reduce num-ghosts temporarily (start with 1-2)
4. Increase Stage 2 timesteps

### Issue: Training too slow

**Symptoms**: Taking many hours per stage

**Solutions**:
1. Reduce `--num-envs` if memory constrained
2. Use smaller layouts (smallGrid, mediumGrid)
3. Reduce timesteps for initial testing
4. Consider GPU training for PPO (though CPU is usually fine for small networks)

### Issue: Adversarial training unstable

**Symptoms**: Win rates oscillate wildly, no convergence

**Solutions**:
1. Reduce learning rates in adversarial phase
2. Increase refinement timesteps
3. Add more evaluation episodes to smooth out variance
4. Ensure Stage 2 model is well-trained before starting Stage 3

## 🔬 Experimental Variations

### Conservative Curriculum (Slower but Safer)

```bash
python train_curriculum.py --full-curriculum \
    --stage1-timesteps 300000 \
    --stage2-timesteps 800000 \
    --adversarial-rounds 20 \
    --ghost-refine-steps 50000 \
    --pacman-refine-steps 80000
```

### Aggressive Curriculum (Faster but Riskier)

```bash
python train_curriculum.py --full-curriculum \
    --stage1-timesteps 100000 \
    --stage2-timesteps 300000 \
    --adversarial-rounds 6 \
    --ghost-refine-steps 20000 \
    --pacman-refine-steps 30000
```

### Multi-Ghost Incremental

If you want to try adding ghosts one at a time in Stage 2:

```bash
# Stage 2a: 1 ghost
python train_curriculum.py --skip-to random_ghosts \
    --pacman-model path/to/stage1_model \
    --num-ghosts 1 \
    --stage2-timesteps 200000

# Stage 2b: 2 ghosts
python train_curriculum.py --skip-to random_ghosts \
    --pacman-model path/to/stage2a_model \
    --num-ghosts 2 \
    --stage2-timesteps 200000

# Stage 2c: All ghosts
python train_curriculum.py --skip-to random_ghosts \
    --pacman-model path/to/stage2b_model \
    --num-ghosts 4 \
    --stage2-timesteps 300000
```

## 📝 Notes on Design Decisions

### Why this curriculum order?

1. **No Ghosts First**: Establishes reward signal and basic competence
2. **Scripted Ghosts Second**: Predictable opponents allow learning stable evasion patterns
3. **Adversarial Last**: Only tackle adaptive opponents once basics are solid

### Why not add ghosts incrementally?

While adding 1 ghost at a time might seem gentler, in practice:
- **Convergence time**: Each intermediate stage adds training overhead
- **Transfer**: Skills learned against 1-2 ghosts don't transfer perfectly to 4 ghosts
- **Simplicity**: Fewer stages = fewer hyperparameters to tune

The jump from 0→4 ghosts is manageable because Stage 1 provides a strong foundation.

### Why PPO for Pac-Man and DQN for Ghosts?

- **PPO**: Better for complex strategy spaces (Pac-Man needs to balance multiple objectives)
- **DQN**: More aggressive/greedy (perfect for ghosts whose goal is pure pursuit)

This asymmetric design creates more interesting dynamics than using the same algorithm for both.

## 🎓 Academic Context

This project implements adversarial RL with curriculum learning as an extension of:
- Stanford CS229 2017 Pac-Man project (DQN baseline)
- UC Berkeley CS188 AI course (game environment)
- Modern curriculum learning principles (Bengio et al.)

Key innovations:
1. Systematic curriculum to overcome cold-start problem
2. Asymmetric algorithms (PPO vs DQN) for adversarial training
3. Sequential ghost training with rotation to reduce asymmetry

## 📚 References

- Original UC Berkeley Pac-Man: http://ai.berkeley.edu/project_overview.html
- Stable-Baselines3 documentation: https://stable-baselines3.readthedocs.io/
- Curriculum Learning: Bengio et al. (2009)
- Adversarial RL: Pinto et al. (2017), Bansal et al. (2018)

## 🤝 Contributing

This is an academic project. Suggestions for improvement:
- Experiment with different curriculum schedules
- Try alternative reward shaping
- Test on additional layouts
- Implement ghost diversity mechanisms
- Add opponent modeling techniques

## 📄 License

Based on UC Berkeley CS188 materials - free for educational use with attribution.