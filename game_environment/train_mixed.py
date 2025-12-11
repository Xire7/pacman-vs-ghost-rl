"""
Mixed Adversarial Training for Pac-Man vs Ghosts

CURRICULUM LEARNING VERSION:
- Phase 0: Ghosts train with 3-stage curriculum:
  * Stage 1 (30%): Random Pac-Man - learn basic movement
  * Stage 2 (50%): Fleeing Pac-Man - learn pursuit and cut-offs
  * Stage 3 (20%): Smart Pac-Man - adapt to real opponent
- Phase 1+: Alternating adversarial training

This progressive difficulty prevents confusion and accelerates learning.
"""

import argparse
import os
import numpy as np
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from ghost_agent import IndependentGhostEnv
from gym_env import PacmanEnv
from game import Directions
from util import manhattanDistance


class FleeingPacmanPolicy:
    """
    Simple scripted Pac-Man that tries to run away from ghosts.
    This teaches ghosts actual pursuit behavior during pretraining.
    """
    
    def predict(self, obs, deterministic=True, action_masks=None):
        """
        Compatible with stable-baselines3 API.
        Returns action and dummy state.
        """
        return 0, None  # dummy return
    
    def get_action(self, game_state):
        """
        Get action based on game state.
        Strategy: Run away from nearest dangerous ghost.
        """
        legal = game_state.getLegalPacmanActions()
        pacman_pos = game_state.getPacmanPosition()
        
        # find nearest dangerous ghost
        min_dist = float('inf')
        nearest_ghost_pos = None
        for ghost_state in game_state.getGhostStates():
            if ghost_state.scaredTimer == 0:  # only flee from dangerous ghosts
                ghost_pos = ghost_state.getPosition()
                dist = manhattanDistance(pacman_pos, ghost_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_ghost_pos = ghost_pos
        
        if nearest_ghost_pos and min_dist < 10:  # only flee if ghost is close
            # calculate direction away from ghost
            dx = pacman_pos[0] - nearest_ghost_pos[0]
            dy = pacman_pos[1] - nearest_ghost_pos[1]
            
            # choose direction based on which axis has larger difference
            if abs(dy) > abs(dx):
                preferred = Directions.NORTH if dy > 0 else Directions.SOUTH
                secondary = Directions.EAST if dx > 0 else Directions.WEST
            else:
                preferred = Directions.EAST if dx > 0 else Directions.WEST
                secondary = Directions.NORTH if dy > 0 else Directions.SOUTH
            
            if preferred in legal and preferred != Directions.STOP:
                return preferred
            if secondary in legal and secondary != Directions.STOP:
                return secondary
        
        food_list = game_state.getFood().asList()
        if food_list:
            min_food_dist = float('inf')
            nearest_food = None
            for food_pos in food_list:
                dist = manhattanDistance(pacman_pos, food_pos)
                if dist < min_food_dist:
                    min_food_dist = dist
                    nearest_food = food_pos
            
            if nearest_food:
                dx = nearest_food[0] - pacman_pos[0]
                dy = nearest_food[1] - pacman_pos[1]
                
                if abs(dy) > abs(dx):
                    preferred = Directions.NORTH if dy > 0 else Directions.SOUTH
                else:
                    preferred = Directions.EAST if dx > 0 else Directions.WEST
                
                if preferred in legal and preferred != Directions.STOP:
                    return preferred
        
        non_stop_actions = [a for a in legal if a != Directions.STOP]
        if non_stop_actions:
            return np.random.choice(non_stop_actions)
        return np.random.choice(legal)


class FleeingPacmanWrapper:
    """
    Wrapper to make FleeingPacmanPolicy compatible with the training pipeline.
    This allows us to use it in IndependentGhostEnv without major refactoring.
    """
    
    def __init__(self):
        self.policy = FleeingPacmanPolicy()
    
    def predict(self, obs, deterministic=True, action_masks=None):
        """
        Placeholder for SB3 compatibility.
        The actual decision making happens through the game state in IndependentGhostEnv.
        """
        return 0, None


def create_dirs(base_dir="training_output"):
    """Create output directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"mixed_{timestamp}")
    dirs = {
        'models': os.path.join(run_dir, 'models'),
        'logs': os.path.join(run_dir, 'logs'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return run_dir, dirs


def load_pretrained_pacman(model_path):
    """
    Load pretrained Pac-Man model with VecNormalize if available.
    
    Returns:
        tuple: (model, vecnorm_path or None)
    """
    print(f"\nLoading pretrained Pac-Man from: {model_path}")
    
    # Check for VecNormalize file
    model_dir = os.path.dirname(model_path)
    vecnorm_path = os.path.join(model_dir, 'vecnormalize.pkl')

    # If not found and we're in a subdirectory (like "best/"), check parent
    if not os.path.exists(vecnorm_path):
        parent_dir = os.path.dirname(model_dir)
        parent_vecnorm = os.path.join(parent_dir, 'vecnormalize.pkl')
        if os.path.exists(parent_vecnorm):
            vecnorm_path = parent_vecnorm
    
    if os.path.exists(vecnorm_path):
        print(f"Found VecNormalize stats: {vecnorm_path}")
        return model_path, vecnorm_path
    else:
        print(f"No VecNormalize found (model trained without --normalize)")
        return model_path, None


def evaluate(model, layout, ghost_models=None, n=50, vecnorm_stats=None):
    """
    Evaluate Pac-Man win rate.
    
    Args:
        model: Trained MaskablePPO model
        layout: Map layout name
        ghost_models: Dict of ghost models (or None for random)
        n: Number of episodes
        vecnorm_stats: Path to vecnormalize.pkl (or None)
    """
    wins = 0
    
    for _ in range(n):
        if ghost_models:
            env = PacmanEnv(layout_name=layout, ghost_policies=ghost_models, max_steps=500)
        else:
            env = PacmanEnv(layout_name=layout, ghost_type='random', max_steps=500)
        
        env = ActionMasker(env, lambda e: e.action_masks())
        
        env = DummyVecEnv([lambda: env])
        
        if vecnorm_stats:
            env = VecNormalize.load(vecnorm_stats, env)
            env.training = False
            env.norm_reward = False
        
        obs = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True, 
                                     action_masks=env.env_method('action_masks')[0])
            if isinstance(action, np.ndarray):
                action = int(action.item())
            obs, _, done, info = env.step([action])
            done = done[0]
        
        if info[0].get('win'):
            wins += 1
        
        env.close()
    
    return wins / n


def pretrain_ghost_curriculum(ghost_idx, pacman_model, layout, dirs, total_timesteps, num_ghosts, vecnorm_stats=None):
    """
    THREE-STAGE CURRICULUM LEARNING for ghost pretraining.
    
    Stage 1 (30%): Random Pac-Man - Learn basic movement and coordination
    Stage 2 (50%): Fleeing Pac-Man - Learn pursuit, cut-offs, and hunting
    Stage 3 (20%): Smart Pac-Man - Adapt to actual opponent behavior
    
    This progressive difficulty teaches ghosts effectively without confusion.
    
    Args:
        ghost_idx: Which ghost to train
        pacman_model: Trained Pac-Man model (for stage 3)
        layout: Map layout
        dirs: Output directories
        total_timesteps: Total training timesteps across all stages
        num_ghosts: Total number of ghosts
        vecnorm_stats: Path to VecNormalize for Pac-Man (for stage 3)
    
    Returns:
        Trained DQN model
    """
    print(f"\n{'='*60}")
    print(f"CURRICULUM LEARNING: Ghost {ghost_idx}")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Stage 1 (30%): Random Pac-Man   - {int(total_timesteps * 0.3):,} steps")
    print(f"Stage 2 (50%): Fleeing Pac-Man  - {int(total_timesteps * 0.5):,} steps")
    print(f"Stage 3 (20%): Smart Pac-Man    - {int(total_timesteps * 0.2):,} steps")
    print(f"{'='*60}\n")
    
    # Calculate timesteps per stage
    stage1_steps = int(total_timesteps * 0.3)
    stage2_steps = int(total_timesteps * 0.5)
    stage3_steps = total_timesteps - stage1_steps - stage2_steps  # Remainder
    
    tb_log_dir = os.path.join(dirs['logs'], f"ghost_{ghost_idx}_pretrain_v0")
    
    # ========== STAGE 1: RANDOM PAC-MAN (30%) ==========
    print(f"Stage 1/3: Learning basics vs Random Pac-Man...")
    print(f"  Goal: Basic movement, wall avoidance, staying near center")
    
    env1 = IndependentGhostEnv(
        ghost_index=ghost_idx,
        layout_name=layout,
        pacman_policy=None,  # Random Pac-Man
        other_ghost_policies=None,
        max_steps=500
    )
    
    # initialize DQN with good hyperparameters for curriculum learning
    policy_kwargs = dict(
        net_arch=[256, 128, 64],
    )
    
    model = DQN(
        "MlpPolicy", env1,
        learning_rate=5e-4,
        buffer_size=200000,         # large buffer for diverse experiences
        learning_starts=10000,      # more initial exploration
        batch_size=256,             # larger batches for stability
        gamma=0.99,
        target_update_interval=2000,
        exploration_fraction=0.5,   # explore for 50% of stage 1
        exploration_final_eps=0.10,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=tb_log_dir
    )
    
    print(f"  Training for {stage1_steps:,} steps...")
    model.learn(total_timesteps=stage1_steps, progress_bar=True, reset_num_timesteps=True)
    print(f"  Stage 1 complete\n")
    
    env1.close()
    
    # ========== STAGE 2: FLEEING PAC-MAN (50%) ==========
    print(f"Stage 2/3: Learning pursuit vs Fleeing Pac-Man...")
    print(f"  Goal: Chase behavior, cut-offs, trapping strategies")
    
    env2 = IndependentGhostEnv(
        ghost_index=ghost_idx,
        layout_name=layout,
        pacman_policy=FleeingPacmanWrapper(),  # Fleeing Pac-Man
        other_ghost_policies=None,
        max_steps=500
    )
    
    # Transfer to new environment
    model.set_env(env2)
    
    # Reduce learning rate for fine-tuning
    model.learning_rate = 3e-4
    model.exploration_rate = 0.15  # Keep some exploration
    
    print(f"  Training for {stage2_steps:,} steps...")
    print(f"  (Learning rate reduced to 3e-4 for stability)")
    model.learn(total_timesteps=stage2_steps, progress_bar=True, reset_num_timesteps=False)
    print(f"  Stage 2 complete\n")
    
    env2.close()
    
    # ========== STAGE 3: SMART PAC-MAN (20%) ==========
    print(f"Stage 3/3: Adapting to Smart Pac-Man...")
    print(f"  Goal: Handle opponent's evasive strategies")
    
    # Load other pretrained ghosts if they exist (for coordination)
    other_ghosts = {}
    for i in range(1, num_ghosts + 1):
        if i != ghost_idx:
            ghost_path = os.path.join(dirs['models'], f"ghost_{i}_v0.zip")
            if os.path.exists(ghost_path):
                other_ghosts[i] = DQN.load(ghost_path)
                print(f"  Loaded Ghost {i} v0 for coordination practice")
    
    if not other_ghosts:
        print(f"  No other ghosts loaded (first ghost being trained)")
    
    env3 = IndependentGhostEnv(
        ghost_index=ghost_idx,
        layout_name=layout,
        pacman_policy=pacman_model,  # Smart trained Pac-Man
        other_ghost_policies=other_ghosts if other_ghosts else None,
        max_steps=500,
        vecnorm_path=vecnorm_stats
    )
    
    # transfer to new environment
    model.set_env(env3)
    
    # further reduce learning rate for adaptation
    model.learning_rate = 1e-4
    model.exploration_rate = 0.05  # minimal exploration, mostly exploitation
    
    print(f"  Training for {stage3_steps:,} steps...")
    print(f"  (Learning rate reduced to 1e-4 for fine-tuning)")
    model.learn(total_timesteps=stage3_steps, progress_bar=True, reset_num_timesteps=False)
    print(f"  Stage 3 complete\n")
    
    env3.close()
    
    # Save the curriculum-trained model
    model.save(os.path.join(dirs['models'], f"ghost_{ghost_idx}_v0"))
    print(f"✓ Ghost {ghost_idx} curriculum training complete (saved as v0)")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Final exploration rate: {model.exploration_rate:.3f}")
    print(f"{'='*60}\n")
    
    return model


def train_ghost(ghost_idx, pacman_model, ghost_models, layout, dirs, timesteps, version, vecnorm_stats=None):
    """
    Train a ghost using DQN against trained Pac-Man.
    
    Args:
        vecnorm_stats: Path to VecNormalize stats for Pac-Man policy
    """
    print(f"\n  Training Ghost {ghost_idx} (v{version})...")
    
    num_ghosts = len(ghost_models)
    other_ghosts = {i: ghost_models[i] for i in range(1, num_ghosts + 1) 
                    if i != ghost_idx and ghost_models[i] is not None}
    
    # Create ghost training environment
    env = IndependentGhostEnv(
        ghost_index=ghost_idx,
        layout_name=layout,
        pacman_policy=pacman_model,
        other_ghost_policies=other_ghosts,
        max_steps=500,
        vecnorm_path=vecnorm_stats
    )
    
    prev_path = os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{version-1}.zip")
    
    tb_log_dir = os.path.join(dirs['logs'], f"ghost_{ghost_idx}_v{version}")
    
    if os.path.exists(prev_path) and version > 0:
        print(f"    Loading Ghost {ghost_idx} v{version-1} for refinement...")
        model = DQN.load(prev_path, env=env)
        
        model.num_timesteps = 0
        model._num_timesteps_at_start = 0
        model.tensorboard_log = tb_log_dir
        
        model.learning_rate = 2e-4  # Lower LR for refinement
        model.exploration_rate = 0.08  # Keep some exploration
        print(f"    Refining for {timesteps:,} timesteps (lr=2e-4, ε=0.08)")
        print(f"    TensorBoard: {tb_log_dir}")
    else:
        print(f"    Creating new DQN for Ghost {ghost_idx}...")
        print(f"    TensorBoard: {tb_log_dir}")
        
        policy_kwargs = dict(
            net_arch=[256, 128, 64],
        )
        
        model = DQN(
            "MlpPolicy", env,
            learning_rate=5e-4,
            buffer_size=150000,
            learning_starts=5000,
            batch_size=256,
            gamma=0.99,
            target_update_interval=2000,
            exploration_fraction=0.5,
            exploration_final_eps=0.10,
            train_freq=4,
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=tb_log_dir
        )
        print(f"    Training for {timesteps:,} timesteps")
    
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(os.path.join(dirs['models'], f"ghost_{ghost_idx}_v{version}"))
    env.close()
    return model


def train_pacman(pacman_path, layout, dirs, ghost_models, timesteps, version, n_envs=8, vecnorm_stats=None):
    """
    Train Pac-Man using alternating random/trained ghost phases.
    
    Args:
        vecnorm_stats: Path to VecNormalize stats (will be updated during training)
    """
    print(f"\n  Training Pac-Man (v{version})...")
    
    def make_random_env():
        def _init():
            env = PacmanEnv(layout_name=layout, ghost_type='random', max_steps=500)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init
    
    def make_trained_env():
        def _init():
            env = PacmanEnv(layout_name=layout, ghost_policies=ghost_models, max_steps=500)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init

    # gradual shift: more random early, more trained later
    random_ratio = max(0.5, 0.9 - 0.05 * (version - 1))
    random_steps = int(timesteps * random_ratio)
    trained_steps = timesteps - random_steps
    
    log_dir = os.path.join(dirs['logs'], f"pacman_v{version}")
    
    # phase 1: Train on random ghosts
    print(f"    Phase 1: {random_steps:,} steps vs random ghosts ({random_ratio*100:.0f}%)")
    print(f"    TensorBoard: {log_dir}")
    env_rand = DummyVecEnv([make_random_env() for _ in range(n_envs)])
    env_rand = VecMonitor(env_rand)
    
    # apply VecNormalize
    if vecnorm_stats and os.path.exists(vecnorm_stats):
        print(f"    Loading VecNormalize stats from: {vecnorm_stats}")
        env_rand = VecNormalize.load(vecnorm_stats, env_rand)
        env_rand.training = True
        env_rand.norm_reward = True
    else:
        print(f"    Creating new VecNormalize")
        env_rand = VecNormalize(env_rand, norm_obs=True, norm_reward=True, 
                                clip_obs=10.0, clip_reward=10.0)
    
    # load model
    model = MaskablePPO.load(pacman_path, env=env_rand)
    
    model.num_timesteps = 0
    model._num_timesteps_at_start = 0
    model.tensorboard_log = log_dir
    
    model.learning_rate = 1e-4
    model.clip_range = lambda _: 0.1
    model.learn(total_timesteps=random_steps, progress_bar=True)
    
    # save updated VecNormalize stats
    vecnorm_save_path = os.path.join(dirs['models'], 'vecnormalize.pkl')
    env_rand.save(vecnorm_save_path)
    print(f"    Saved VecNormalize to: {vecnorm_save_path}")
    
    env_rand.close()
    
    # phase 2: Train on trained ghosts
    print(f"    Phase 2: {trained_steps:,} steps vs trained ghosts ({(1-random_ratio)*100:.0f}%)")
    env_trained = DummyVecEnv([make_trained_env() for _ in range(n_envs)])
    env_trained = VecMonitor(env_trained)
    
    env_trained = VecNormalize.load(vecnorm_save_path, env_trained)
    env_trained.training = True
    env_trained.norm_reward = True
    
    model2 = MaskablePPO.load(pacman_path, env=env_trained)
    model2.policy.load_state_dict(model.policy.state_dict())
    
    model2.num_timesteps = model.num_timesteps
    model2.tensorboard_log = log_dir
    
    model2.learning_rate = 1e-4
    model2.clip_range = lambda _: 0.1
    model2.learn(total_timesteps=trained_steps, progress_bar=True, reset_num_timesteps=False)
    
    env_trained.save(vecnorm_save_path)
    env_trained.close()
    
    save_path = os.path.join(dirs['models'], f"pacman_v{version}")
    model2.save(save_path)
    
    return model2, save_path + ".zip", vecnorm_save_path


def main():
    parser = argparse.ArgumentParser(
        description='Mixed Adversarial Training with Curriculum Learning',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--rounds', type=int, default=7, 
                       help='Training rounds (default: 7, reduced from 10 for quality over quantity)')
    parser.add_argument('--layout', type=str, default='mediumClassic')
    parser.add_argument('--pacman', type=str, required=True, help='Pretrained Pac-Man path (.zip file)')
    parser.add_argument('--ghost-steps', type=int, default=120000, 
                       help='Ghost training steps per round (default: 120k, up from 80k)')
    parser.add_argument('--pacman-steps', type=int, default=80000, 
                       help='Pac-Man training steps per round (default: 80k)')
    parser.add_argument('--ghost-pretrain-steps', type=int, default=150000, 
                       help='Ghost curriculum pretraining total timesteps (default: 150k)\n'
                            '  Split: 30%% random, 50%% fleeing, 20%% smart')
    parser.add_argument('--skip-ghost-pretrain', action='store_true',
                       help='Skip ghost curriculum pretraining (not recommended)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"MIXED ADVERSARIAL TRAINING WITH CURRICULUM LEARNING")
    print(f"  Layout: {args.layout}")
    print(f"  Rounds: {args.rounds}")
    if args.skip_ghost_pretrain:
        print(f"  Ghost Pretrain: DISABLED")
    else:
        print(f"  Ghost Curriculum Pretrain: {args.ghost_pretrain_steps:,} total steps")
        print(f"    Stage 1 (30%): Random Pac-Man   - {int(args.ghost_pretrain_steps * 0.3):,} steps")
        print(f"    Stage 2 (50%): Fleeing Pac-Man  - {int(args.ghost_pretrain_steps * 0.5):,} steps")
        print(f"    Stage 3 (20%): Smart Pac-Man    - {int(args.ghost_pretrain_steps * 0.2):,} steps")
    print(f"  Ghost Steps/Round: {args.ghost_steps:,}")
    print(f"  Pac-Man Steps/Round: {args.pacman_steps:,}")
    print(f"{'='*60}")
    
    run_dir, dirs = create_dirs()
    print(f"Output: {run_dir}\n")
    
    # load pretrained Pac-Man with VecNormalize
    pacman_path, vecnorm_stats = load_pretrained_pacman(args.pacman)
    
    # create a temporary env to load the model
    temp_env = DummyVecEnv([lambda: ActionMasker(
        PacmanEnv(layout_name=args.layout, ghost_type='random', max_steps=500),
        lambda e: e.action_masks()
    )])
    
    if vecnorm_stats:
        temp_env = VecNormalize.load(vecnorm_stats, temp_env)
        temp_env.training = False
        temp_env.norm_reward = False
    
    pacman_model = MaskablePPO.load(pacman_path, env=temp_env)
    temp_env.close()
    
    # baseline evaluation
    baseline = evaluate(pacman_model, args.layout, vecnorm_stats=vecnorm_stats)
    print(f"Baseline Pac-Man vs random ghosts: {baseline*100:.1f}%")
    
    # save as v0
    pacman_model.save(os.path.join(dirs['models'], "pacman_v0"))
    if vecnorm_stats:
        import shutil
        vecnorm_dest = os.path.join(dirs['models'], 'vecnormalize.pkl')
        shutil.copy(vecnorm_stats, vecnorm_dest)
        vecnorm_stats = vecnorm_dest
        print(f"Copied VecNormalize to: {vecnorm_dest}")
    
    # get ghost count
    temp_env = PacmanEnv(layout_name=args.layout)
    num_ghosts = temp_env.num_ghosts
    temp_env.close()
    print(f"Number of ghosts: {num_ghosts}\n")
    
    ghost_models = {i: None for i in range(1, num_ghosts + 1)}
    
    # ========== GHOST CURRICULUM PRETRAINING PHASE ==========
    if not args.skip_ghost_pretrain:
        print(f"\n{'='*60}")
        print(f"PHASE 0: GHOST CURRICULUM PRETRAINING")
        print(f"{'='*60}")
        print(f"Strategy: Progressive difficulty curriculum")
        print(f"  1. Random Pac-Man (30%) - Basic movement")
        print(f"  2. Fleeing Pac-Man (50%) - Chase & pursuit")
        print(f"  3. Smart Pac-Man (20%) - Real opponent adaptation\n")
        
        for ghost_idx in range(1, num_ghosts + 1):
            ghost_models[ghost_idx] = pretrain_ghost_curriculum(
                ghost_idx=ghost_idx,
                pacman_model=pacman_model,
                layout=args.layout,
                dirs=dirs,
                total_timesteps=args.ghost_pretrain_steps,
                num_ghosts=num_ghosts,
                vecnorm_stats=vecnorm_stats
            )
        
        print(f"\n{'='*60}")
        print(f"ALL GHOSTS CURRICULUM TRAINED (v0)")
        print(f"{'='*60}")
        
        # evaluate curriculum-trained ghosts
        print(f"\nCheckpoint: Testing curriculum-trained ghosts vs trained Pac-Man")
        print(f"{'─'*60}")
        wr_baseline = evaluate(pacman_model, args.layout, None, n=30, vecnorm_stats=vecnorm_stats)
        wr_after_pretrain = evaluate(pacman_model, args.layout, ghost_models, n=30, vecnorm_stats=vecnorm_stats)
        print(f"\nPac-Man win rates:")
        print(f"  vs random ghosts:              {wr_baseline*100:.1f}%")
        print(f"  vs curriculum-trained ghosts: {wr_after_pretrain*100:.1f}%")
        
        if wr_after_pretrain < wr_baseline:
            improvement = ((wr_baseline - wr_after_pretrain) / wr_baseline) * 100
            print(f"\n Curriculum training successful!")
            print(f"  Ghosts are {improvement:.1f}% more challenging than random")
            print(f"  Ghosts learned: movement → pursuit → adaptation")
        else:
            print(f"\n Ghosts not yet better than random")
            print(f"  Consider increasing --ghost-pretrain-steps")
        
        print(f"\nReady for adversarial training!\n")
    else:
        print(f"\n SKIPPED ghost curriculum pretraining")
        print(f"Ghosts will start from scratch (not recommended)\n")
    
    # ========== ADVERSARIAL TRAINING LOOP ==========
    print(f"\n{'='*60}")
    print(f"ADVERSARIAL TRAINING ROUNDS")
    print(f"{'='*60}\n")
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'─'*60}")
        print(f"ROUND {round_num}/{args.rounds}")
        print(f"{'─'*60}")
        
        # Train ghosts
        ghost_version = round_num
        print(f"\nPhase: Train Ghosts (→ v{ghost_version})")
        ghost_order = list(range(1, num_ghosts + 1))
        np.random.shuffle(ghost_order)
        
        for ghost_idx in ghost_order:
            ghost_models[ghost_idx] = train_ghost(
                ghost_idx, pacman_model, ghost_models,
                args.layout, dirs, args.ghost_steps, ghost_version,
                vecnorm_stats=vecnorm_stats
            )
        
        # Train Pac-Man
        pacman_version = round_num
        print(f"\nPhase: Train Pac-Man (→ v{pacman_version})")
        pacman_model, pacman_path, vecnorm_stats = train_pacman(
            pacman_path, args.layout, dirs, ghost_models,
            args.pacman_steps, pacman_version, vecnorm_stats=vecnorm_stats
        )
        
        # Evaluate
        print(f"\n{'─'*60}")
        print(f"Round {round_num} Evaluation")
        print(f"{'─'*60}")
        wr_random = evaluate(pacman_model, args.layout, vecnorm_stats=vecnorm_stats, n=30)
        wr_trained = evaluate(pacman_model, args.layout, ghost_models, n=30, vecnorm_stats=vecnorm_stats)
        
        print(f"\nPac-Man v{pacman_version} Performance:")
        print(f"  vs random ghosts:  {wr_random*100:.1f}%")
        print(f"  vs trained ghosts: {wr_trained*100:.1f}%")
        
        if wr_random < 0.70:
            print(f"\n WARNING: Pac-Man struggling vs random ghosts!")
            print(f"  May need to increase random ghost training ratio")
        
        if round_num > 1 and wr_trained > 0.85:
            print(f"\n Pac-Man dominating - ghosts may need more training")
    
    # ========== FINAL EVALUATION ==========
    print(f"\n\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    
    final_random = evaluate(pacman_model, args.layout, n=100, vecnorm_stats=vecnorm_stats)
    final_trained = evaluate(pacman_model, args.layout, ghost_models, n=100, vecnorm_stats=vecnorm_stats)
    
    print(f"\nPac-Man Final Performance:")
    print(f"  vs Random Ghosts:  {final_random*100:.1f}% (100 episodes)")
    print(f"  vs Trained Ghosts: {final_trained*100:.1f}% (100 episodes)")
    
    print(f"\nImprovement Analysis:")
    print(f"  Baseline:  {baseline*100:.1f}%")
    print(f"  Final:     {final_trained*100:.1f}%")
    
    if final_trained < baseline:
        challenge_increase = ((baseline - final_trained) / baseline) * 100
        print(f"  Ghosts are {challenge_increase:.1f}% more challenging!")
    
    # Save best model
    pacman_model.save(os.path.join(dirs['models'], "pacman_best"))
    print(f"\n Saved best Pac-Man: {dirs['models']}/pacman_best.zip")
    print(f" VecNormalize stats: {vecnorm_stats}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nView training progress:")
    print(f"  tensorboard --logdir={dirs['logs']}")
    print(f"\nAll models saved to: {dirs['models']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()