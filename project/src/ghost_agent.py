from util import manhattanDistance
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import Directions
from state_extractor import extract_ghost_observation, extract_pacman_observation, get_best_legal_action
from gym_env import PacmanEnv

class IndependentGhostEnv(gym.Env):
    """
    Environment that trains each ghost with its own DQN.
    
    Key insight: Instead of one network controlling all ghosts,
    we have 4 separate networks, each seeing only their own ghost's perspective.
    """
    
    def __init__(self, ghost_index, layout_name='mediumClassic', 
                 pacman_policy=None, other_ghost_policies=None, 
                 render_mode=None, max_steps=500, vecnorm_path=None):
        """
        Args:
            ghost_index: Which ghost this environment controls (1, 2, 3, or 4)
            pacman_policy: Trained Pac-Man policy (MaskablePPO model)
            other_ghost_policies: Dict of {ghost_idx: trained_model} for other ghosts
            vecnorm_path: Path to vecnormalize.pkl for Pac-Man policy (if trained with --normalize)
        """
        super().__init__()
        
        self.ghost_index = ghost_index
        self.pacman_policy = pacman_policy
        self.other_ghost_policies = other_ghost_policies or {}
        self.vecnorm_path = vecnorm_path
        
        # Lazy-loaded VecNormalize wrapper for Pac-Man observations
        self._vecnorm_wrapper = None
        
        self.pacman_env = PacmanEnv(
            layout_name=layout_name,
            max_steps=max_steps,
            render_mode=render_mode
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(25,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(4)
        
        self.game_state = None
        self.steps = 0
        self.prev_ghost_distances = {}
        self.prev_dist_to_pacman = float('inf')
    
    def _get_vecnorm_wrapper(self):
        """
        Lazy-load VecNormalize wrapper for normalizing Pac-Man observations.
        Only created once and reused for efficiency.
        """
        if self._vecnorm_wrapper is None and self.vecnorm_path:
            try:
                from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
                from sb3_contrib.common.wrappers import ActionMasker
                
                # Create a dummy environment for VecNormalize
                def make_dummy_env():
                    env = PacmanEnv(
                        layout_name=self.pacman_env.layout_name,
                        max_steps=self.pacman_env.max_steps,
                        render_mode=None
                    )
                    return ActionMasker(env, lambda e: e.action_masks())
                
                dummy_vec_env = DummyVecEnv([make_dummy_env])
                
                # Load VecNormalize statistics
                self._vecnorm_wrapper = VecNormalize.load(self.vecnorm_path, dummy_vec_env)
                self._vecnorm_wrapper.training = False  # Don't update stats during ghost training
                self._vecnorm_wrapper.norm_reward = False  # Don't normalize rewards
                
                print(f"  Ghost {self.ghost_index}: Loaded VecNormalize for Pac-Man policy")
            except Exception as e:
                print(f"  Warning: Failed to load VecNormalize: {e}")
                self._vecnorm_wrapper = None
        
        return self._vecnorm_wrapper
    
    def _normalize_pacman_obs(self, obs):
        """
        Normalize Pac-Man observation using VecNormalize if available.
        
        Args:
            obs: Raw observation from Pac-Man environment (shape: (33,))
        
        Returns:
            Normalized observation (same shape)
        """
        if self.vecnorm_path is None:
            # No normalization needed
            return obs
        
        vecnorm = self._get_vecnorm_wrapper()
        
        if vecnorm is None:
            # Failed to load VecNormalize, return raw obs
            return obs
        
        try:
            # VecNormalize expects batched input: (n_envs, obs_dim)
            obs_batch = obs.reshape(1, -1)
            normalized_batch = vecnorm.normalize_obs(obs_batch)
            return normalized_batch[0]  # Return unbatched
        except Exception as e:
            print(f"  ⚠ Warning: VecNormalize normalization failed: {e}")
            return obs
    
    def reset(self, *, seed=None, options=None):
        """Reset and return this ghost's observation."""
        super().reset(seed=seed)
        
        obs, info = self.pacman_env.reset(seed=seed)
        self.game_state = self.pacman_env.game_state
        self.game_state.data._agentMoved = 0
        self.steps = 0
        
        # Return only THIS ghost's observation
        ghost_obs = extract_ghost_observation(self.game_state, self.ghost_index)
        
        return ghost_obs, {}
    
    def step(self, action):
        """
        Execute full turn:
        1. Pac-Man moves (using policy)
        2. All ghosts move (this ghost uses action, others use their policies)
        """
        action_map = {
            0: Directions.NORTH,
            1: Directions.SOUTH,
            2: Directions.EAST,
            3: Directions.WEST
        }
        
        if isinstance(action, np.ndarray):
            action = int(action.item())
        
        # Store this ghost's action
        my_action = action_map[action]
        
        # ========== EXECUTE PAC-MAN MOVE ==========
        legal_pacman = self.game_state.getLegalActions(0)
        
        if self.pacman_policy is not None:
            # Sync pacman_env state and get observation
            self.pacman_env.game_state = self.game_state
            pacman_obs = self.pacman_env._get_observation()
            
            # NORMALIZE OBSERVATION if VecNormalize is available
            pacman_obs = self._normalize_pacman_obs(pacman_obs)
            
            # Get action masks
            action_masks = self.pacman_env.action_masks()
            
            # Get action with masking (observation is now normalized)
            pacman_action, _ = self.pacman_policy.predict(
                pacman_obs, deterministic=False, action_masks=action_masks
            )
            action_to_dir = {0: Directions.NORTH, 1: Directions.SOUTH, 
                             2: Directions.EAST, 3: Directions.WEST, 4: Directions.STOP}
            pacman_direction = action_to_dir[int(pacman_action)]
            
            if pacman_direction not in legal_pacman:
                pacman_direction = np.random.choice(legal_pacman)
        else:
            # Random Pac-Man
            pacman_direction = np.random.choice(legal_pacman)
        
        self.game_state = self.game_state.generateSuccessor(0, pacman_direction)
        
        # ========== EXECUTE ALL GHOST MOVES ==========
        for ghost_idx in range(1, self.game_state.getNumAgents()):
            if self.game_state.isWin() or self.game_state.isLose():
                break
            
            legal_ghost = self.game_state.getLegalActions(ghost_idx)
            
            if ghost_idx == self.ghost_index:
                # This is OUR ghost - use the action we're training
                ghost_direction = my_action
                if ghost_direction not in legal_ghost:
                    ghost_direction = np.random.choice(legal_ghost)
            
            elif ghost_idx in self.other_ghost_policies:
                # Other ghost has a trained policy
                other_obs = extract_ghost_observation(self.game_state, ghost_idx)
                other_action, _ = self.other_ghost_policies[ghost_idx].predict(
                    other_obs, deterministic=False
                )
                other_action_map = {0: Directions.NORTH, 1: Directions.SOUTH,
                                   2: Directions.EAST, 3: Directions.WEST}
                ghost_direction = other_action_map[int(other_action)]
                if ghost_direction not in legal_ghost:
                    ghost_direction = np.random.choice(legal_ghost)
            
            else:
                # Random ghost
                ghost_direction = np.random.choice(legal_ghost)
            
            self.game_state = self.game_state.generateSuccessor(ghost_idx, ghost_direction)
        
        # ========== CALCULATE REWARD ==========
        reward = self._calculate_individual_ghost_reward()
        
        terminated = self.game_state.isWin() or self.game_state.isLose()
        truncated = self.steps >= self.pacman_env.max_steps and not terminated
        
        self.steps += 1
        
        # ========== GET NEXT OBSERVATION ==========
        if not terminated and not truncated:
            ghost_obs = extract_ghost_observation(self.game_state, self.ghost_index)
        else:
            ghost_obs = np.zeros(25, dtype=np.float32)
        
        info = {
            'ghost_won': self.game_state.isLose(),
            'pacman_won': self.game_state.isWin(),
            'score': self.game_state.getScore()
        }
        
        return ghost_obs, reward, terminated, truncated, info
    
    def _calculate_individual_ghost_reward(self):
        """
        AGGRESSIVE chase-focused reward.
        Goal: Make ghosts HUNT Pac-Man relentlessly.
        """
        reward = 0.0
        
        # terminal rewards (keep strong)
        if self.game_state.isLose():
            reward += 200.0  # 2x - catching Pac-Man is the goal!
            return reward
        elif self.game_state.isWin():
            reward -= 100.0  # 2x penalty - failed to catch
            return reward
        
        pacman_pos = self.game_state.getPacmanPosition()
        ghost_pos = self.game_state.getGhostPosition(self.ghost_index)
        ghost_state = self.game_state.getGhostState(self.ghost_index)
        dist = manhattanDistance(pacman_pos, ghost_pos)
        
        if ghost_state.scaredTimer == 0:
            # Exponential reward for being close - makes proximity CRITICAL
            if dist <= 1:
                reward += 50.0  # right next to Pac-Man - HUGE reward!
            elif dist <= 2:
                reward += 25.0  # very close
            elif dist <= 3:
                reward += 12.0  # close
            elif dist <= 5:
                reward += 5.0   # nearby
            else:
                reward += 10.0 / (dist + 1)  # diminishing returns for being far
            
            # reward for getting closer - this is the KEY behavior we want
            if hasattr(self, 'prev_dist_to_pacman'):
                dist_improvement = self.prev_dist_to_pacman - dist
                if dist_improvement > 0:
                    #  reward for closing distance
                    reward += dist_improvement * 10.0  # was 2.0, now 10.0
                elif dist_improvement < 0:
                    # penalty for moving away
                    reward -= abs(dist_improvement) * 5.0  # Punish retreat!
            
            # reward for moving in Pac-Man's direction
            if hasattr(self, 'prev_ghost_pos'):
                # Vector from old position to new position
                move_dx = ghost_pos[0] - self.prev_ghost_pos[0]
                move_dy = ghost_pos[1] - self.prev_ghost_pos[1]
                
                #vVector from ghost to Pac-Man
                target_dx = pacman_pos[0] - ghost_pos[0]
                target_dy = pacman_pos[1] - ghost_pos[1]
                
                # dot product (measures alignment)
                if abs(target_dx) + abs(target_dy) > 0:
                    alignment = (move_dx * target_dx + move_dy * target_dy) / (abs(target_dx) + abs(target_dy))
                    reward += alignment * 3.0  # Reward moving toward Pac-Man
            
            self.prev_ghost_pos = ghost_pos
            
        else:
            # scared: flee!
            reward += dist * 1.0  # stronger flee reward
            
            # penalty for being caught while scared
            if dist <= 2:
                reward -= 20.0
        
        self.prev_dist_to_pacman = dist
        
        # reduce step penalty - don't discourage long chases
        reward -= 0.001  # was 0.01, much smaller now
        
        return reward
    
    def close(self):
        """Clean up resources."""
        if self.pacman_env:
            self.pacman_env.close()
        
        # Close VecNormalize wrapper if it was created
        if self._vecnorm_wrapper is not None:
            try:
                self._vecnorm_wrapper.close()
            except:
                pass
            self._vecnorm_wrapper = None