from util import manhattanDistance
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import Directions
from state_extractor import extract_ghost_observation, extract_pacman_observation, get_legal_action_from_policy
from gym_env import PacmanEnv

class IndependentGhostEnv(gym.Env):
    """Environment for training individual ghost agents with DQN."""
    
    def __init__(self, ghost_index, layout_name='mediumClassic', 
                 pacman_policy=None, other_ghost_policies=None, 
                 render_mode=None, max_steps=500,
                 pacman_obs_rms=None):
        """
        Args:
            ghost_index: Which ghost this env controls (1-based)
            layout_name: Layout to use
            pacman_policy: Trained Pac-Man model (MaskablePPO)
            other_ghost_policies: Dict of other ghost policies
            render_mode: Rendering mode
            max_steps: Max steps per episode
            pacman_obs_rms: VecNormalize obs_rms for normalizing Pac-Man observations
        """
        super().__init__()
        
        self.ghost_index = ghost_index
        self.pacman_policy = pacman_policy
        self.other_ghost_policies = other_ghost_policies or {}
        self.pacman_obs_rms = pacman_obs_rms  # For normalizing Pac-Man obs
        
        self.pacman_env = PacmanEnv(
            layout_name=layout_name,
            max_steps=max_steps,
            render_mode=render_mode
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(33,), dtype=np.float32  # Expanded for smarter features
        )
        
        self.action_space = spaces.Discrete(4)
        
        self.game_state = None
        self.steps = 0
        self.prev_ghost_distances = {}
        self.prev_dist_to_pacman = None  # Will be set on first step
    
    def action_masks(self) -> np.ndarray:
        """Return boolean mask of legal actions for current state."""
        if self.game_state is None:
            return np.ones(4, dtype=bool)
        
        legal_actions = self.game_state.getLegalActions(self.ghost_index)
        action_map = {Directions.NORTH: 0, Directions.SOUTH: 1,
                      Directions.EAST: 2, Directions.WEST: 3}
        
        mask = np.zeros(4, dtype=bool)
        for action in legal_actions:
            if action in action_map:
                mask[action_map[action]] = True
        
        # Ensure at least one action is valid
        if not mask.any():
            mask[:] = True
        
        return mask
    
    def reset(self, *, seed=None, options=None):
        """Reset and return this ghost's observation."""
        super().reset(seed=seed)
        
        obs, info = self.pacman_env.reset(seed=seed)
        self.game_state = self.pacman_env.game_state
        self.game_state.data._agentMoved = 0
        self.steps = 0
        self.prev_dist_to_pacman = None  # Reset distance tracking
        
        ghost_obs = extract_ghost_observation(self.game_state, self.ghost_index)
        return ghost_obs, {}
    
    def step(self, action):
        """Execute full turn: Pac-Man moves, then all ghosts move."""
        action_map = {
            0: Directions.NORTH,
            1: Directions.SOUTH,
            2: Directions.EAST,
            3: Directions.WEST
        }
        
        if isinstance(action, np.ndarray):
            action = int(action.item())
        
        my_action = action_map[action]
        
        # Check if action is legal BEFORE executing
        legal_ghost_actions = self.game_state.getLegalActions(self.ghost_index)
        action_was_illegal = my_action not in legal_ghost_actions
        
        # Pac-Man's turn
        legal_pacman = self.game_state.getLegalActions(0)
        if self.pacman_policy is not None:
            # Sync pacman_env state and get observation
            self.pacman_env.game_state = self.game_state
            pacman_obs = self.pacman_env._get_observation()
            
            # Normalize observation if we have the normalization stats
            if self.pacman_obs_rms is not None:
                pacman_obs = (pacman_obs - self.pacman_obs_rms.mean) / np.sqrt(self.pacman_obs_rms.var + 1e-8)
                pacman_obs = np.clip(pacman_obs, -10.0, 10.0)
            
            # Get action masks
            action_masks = self.pacman_env.action_masks()
            
            # Get action with masking
            pacman_action, _ = self.pacman_policy.predict(
                pacman_obs, deterministic=False, action_masks=action_masks
            )
            action_to_dir = {0: Directions.NORTH, 1: Directions.SOUTH, 
                             2: Directions.EAST, 3: Directions.WEST, 4: Directions.STOP}
            pacman_direction = action_to_dir[int(pacman_action)]
            
            if pacman_direction not in legal_pacman:
                pacman_direction = np.random.choice(legal_pacman)
        else:
            pacman_direction = np.random.choice(legal_pacman)
        
        self.game_state = self.game_state.generateSuccessor(0, pacman_direction)
        
        # Ghosts' turn
        for ghost_idx in range(1, self.game_state.getNumAgents()):
            if self.game_state.isWin() or self.game_state.isLose():
                break
            
            legal_ghost = self.game_state.getLegalActions(ghost_idx)
            
            if ghost_idx == self.ghost_index:
                # This ghost uses the action being trained
                ghost_direction = my_action
                if ghost_direction not in legal_ghost:
                    ghost_direction = np.random.choice(legal_ghost)
            
            elif ghost_idx in self.other_ghost_policies:
                # Other trained ghost - apply action masking
                other_obs = extract_ghost_observation(self.game_state, ghost_idx)
                policy = self.other_ghost_policies[ghost_idx]
                
                # Get Q-values and mask illegal actions (like gym_env.py)
                import torch
                obs_tensor = torch.tensor(other_obs, dtype=torch.float32).unsqueeze(0)
                obs_tensor = obs_tensor.to(policy.device)
                with torch.no_grad():
                    q_values = policy.q_net(obs_tensor).cpu().numpy()[0]
                
                # Create mask for legal actions
                other_action_map = {0: Directions.NORTH, 1: Directions.SOUTH,
                                   2: Directions.EAST, 3: Directions.WEST}
                dir_to_action = {Directions.NORTH: 0, Directions.SOUTH: 1,
                                Directions.EAST: 2, Directions.WEST: 3}
                
                legal_mask = np.full(4, -np.inf)
                for legal_dir in legal_ghost:
                    if legal_dir in dir_to_action:
                        legal_mask[dir_to_action[legal_dir]] = 0.0
                
                # Apply mask and select best legal action
                masked_q = q_values + legal_mask
                other_action = np.argmax(masked_q)
                ghost_direction = other_action_map[int(other_action)]
                
                # Fallback just in case
                if ghost_direction not in legal_ghost:
                    ghost_direction = np.random.choice(legal_ghost)
            
            else:
                ghost_direction = np.random.choice(legal_ghost)
            
            self.game_state = self.game_state.generateSuccessor(ghost_idx, ghost_direction)
        
        reward = self._calculate_individual_ghost_reward()
        
        # Note: illegal action penalty removed - ActionMaskingWrapper handles this
        # by replacing illegal actions with legal ones before they reach here
        
        terminated = self.game_state.isWin() or self.game_state.isLose()
        truncated = self.steps >= self.pacman_env.max_steps and not terminated
        
        self.steps += 1
        
        if not terminated and not truncated:
            ghost_obs = extract_ghost_observation(self.game_state, self.ghost_index)
        else:
            ghost_obs = np.zeros(33, dtype=np.float32)  # Match new observation size
        
        info = {
            'ghost_won': self.game_state.isLose(),
            'pacman_won': self.game_state.isWin(),
            'score': self.game_state.getScore(),
            'illegal_action': action_was_illegal
        }
        
        return ghost_obs, reward, terminated, truncated, info
    
    def _calculate_individual_ghost_reward(self):
        """Calculate reward for this ghost based on proximity and outcomes.
        
        Reward design principles:
        1. Strong, clear terminal rewards (+100 catch, -100 lose)
        2. Dense distance-based shaping that ALWAYS prefers being closer
        3. Bonus for actively closing distance
        4. In scared mode: reward for running away
        5. NEW: Flanking bonus for coordinated attacks
        6. NEW: Reward for cornering (reducing Pac-Man's escape routes)
        """
        # Terminal rewards (dominant signal)
        if self.game_state.isLose():  # Caught Pac-Man
            return 100.0
        elif self.game_state.isWin():  # Pac-Man won
            return -100.0
        
        # Get positions and distance
        pacman_pos = self.game_state.getPacmanPosition()
        ghost_pos = self.game_state.getGhostPosition(self.ghost_index)
        ghost_state = self.game_state.getGhostState(self.ghost_index)
        
        dist = manhattanDistance(pacman_pos, ghost_pos)
        
        # Scared mode: run away from Pac-Man
        if ghost_state.scaredTimer > 0:
            # Reward for being far from Pac-Man (survival)
            if dist <= 2:
                reward = -10.0  # Too close while scared = very dangerous!
            elif dist <= 5:
                reward = -2.0  # Still a bit close
            else:
                reward = 1.0  # Safe distance
            
            # Bonus for increasing distance while scared
            if self.prev_dist_to_pacman is not None:
                dist_delta = dist - self.prev_dist_to_pacman  # Note: reversed - want to increase distance
                if dist_delta > 0:
                    reward += 2.0 * dist_delta  # Reward for running away
            
            self.prev_dist_to_pacman = dist
            return float(np.clip(reward, -100.0, 100.0))
        
        # Chase mode: distance-based reward
        # Scale: about to catch = +15, far away = -3
        max_dist = 20.0
        if dist <= 1:
            reward = 15.0  # About to catch!
        elif dist <= 3:
            reward = 10.0  # Very close
        else:
            # Linear: +5 at dist=3, -3 at dist=20
            reward = 5.0 - (dist - 3) * (8.0 / (max_dist - 3))
            reward = max(reward, -3.0)  # Cap the penalty
        
        # Bonus for closing distance (the key learning signal!)
        if self.prev_dist_to_pacman is not None:
            dist_delta = self.prev_dist_to_pacman - dist
            if dist_delta > 0:
                # Closed distance: strong positive signal
                reward += 5.0 * dist_delta
            elif dist_delta < 0:
                # Increased distance: penalty
                reward += 2.0 * dist_delta  # negative since dist_delta < 0
        
        self.prev_dist_to_pacman = dist
        
        # === NEW REWARDS FOR SMARTER BEHAVIOR ===
        
        # Flanking bonus: reward if ghosts are on opposite sides of Pac-Man
        flanking_bonus = 0.0
        num_agents = self.game_state.getNumAgents()
        for other_idx in range(1, num_agents):
            if other_idx != self.ghost_index:
                other_pos = self.game_state.getGhostPosition(other_idx)
                other_state = self.game_state.getGhostState(other_idx)
                
                # Only count non-scared ghosts for flanking
                if other_state.scaredTimer > 0:
                    continue
                
                my_dx = ghost_pos[0] - pacman_pos[0]
                my_dy = ghost_pos[1] - pacman_pos[1]
                other_dx = other_pos[0] - pacman_pos[0]
                other_dy = other_pos[1] - pacman_pos[1]
                
                # Flanking if on opposite sides (x or y)
                x_flanking = (my_dx > 0 and other_dx < 0) or (my_dx < 0 and other_dx > 0)
                y_flanking = (my_dy > 0 and other_dy < 0) or (my_dy < 0 and other_dy > 0)
                
                if x_flanking or y_flanking:
                    # Bonus scales with how close both ghosts are
                    other_dist = manhattanDistance(pacman_pos, other_pos)
                    if dist <= 5 and other_dist <= 5:
                        flanking_bonus = 3.0  # Both close and flanking = great!
                    elif dist <= 8 or other_dist <= 8:
                        flanking_bonus = 1.5  # At least one close and flanking
                    else:
                        flanking_bonus = 0.5  # Flanking but far
                    break
        
        reward += flanking_bonus
        
        # Cornering bonus: reward for reducing Pac-Man's escape routes
        pacman_legal = self.game_state.getLegalActions(0)
        escape_count = len([a for a in pacman_legal if a != 'Stop'])
        
        # Fewer escapes = better (4 is worst for ghost, 1 is best)
        if escape_count == 1:
            reward += 2.0  # Pac-Man nearly cornered!
        elif escape_count == 2:
            reward += 1.0  # Limited options
        # No bonus/penalty for 3-4 escapes
        
        # Small step penalty to encourage efficiency
        reward -= 0.1
        
        return float(np.clip(reward, -100.0, 100.0))


class MaskedDQN:
    """
    A wrapper around SB3 DQN that implements action masking for single environments.
    
    For vectorized environments, use ActionMaskingWrapper instead.
    """
    
    def __init__(self, env, **dqn_kwargs):
        """
        Args:
            env: The IndependentGhostEnv (must have action_masks() method)
            **dqn_kwargs: Arguments passed to DQN
        """
        from stable_baselines3 import DQN
        import torch
        
        self.env = env
        self.dqn = DQN("MlpPolicy", env, **dqn_kwargs)
        self.device = self.dqn.device
        
    def learn(self, total_timesteps, callback=None, **kwargs):
        """Train with masked action selection."""
        return self.dqn.learn(total_timesteps, callback=callback, **kwargs)
    
    def predict(self, observation, state=None, episode_start=None, deterministic=True, action_mask=None):
        """Predict with optional action masking."""
        import torch
        
        if action_mask is None:
            if hasattr(self.env, 'action_masks'):
                action_mask = self.env.action_masks()
            else:
                action_mask = np.ones(self.dqn.action_space.n, dtype=bool)
        
        # Get Q-values and mask
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs_tensor = obs_tensor.to(self.device)
        
        with torch.no_grad():
            q_values = self.dqn.q_net(obs_tensor).cpu().numpy()
        
        if q_values.ndim == 1:
            q_values = q_values.reshape(1, -1)
        
        # Mask illegal actions
        masked_q = np.where(action_mask, q_values[0], -np.inf)
        action = np.argmax(masked_q)
        
        return np.array([action]), state
    
    def save(self, path):
        """Save the underlying DQN model."""
        self.dqn.save(path)
    
    @classmethod
    def load(cls, path, env=None, device='auto'):
        """Load a saved model."""
        from stable_baselines3 import DQN
        dqn = DQN.load(path, device=device)
        
        # Create wrapper without reinitializing DQN
        instance = cls.__new__(cls)
        instance.env = env
        instance.dqn = dqn
        instance.device = dqn.device
        return instance
    
    @property
    def q_net(self):
        """Access to underlying Q-network for inference."""
        return self.dqn.q_net
    
    @property 
    def exploration_rate(self):
        return self.dqn.exploration_rate


class ActionMaskingWrapper(gym.Wrapper):
    """
    A Gymnasium wrapper that ensures DQN only experiences legal actions.
    
    When DQN picks an illegal action (during epsilon-greedy exploration),
    this wrapper replaces it with a random legal action BEFORE executing.
    
    This ensures the replay buffer only contains valid (state, legal_action, reward, next_state)
    tuples, giving DQN clean training signal.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._current_mask = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_mask = self.env.action_masks() if hasattr(self.env, 'action_masks') else np.ones(4, dtype=bool)
        return obs, info
    
    def step(self, action):
        # Get current action mask
        self._current_mask = self.env.action_masks() if hasattr(self.env, 'action_masks') else np.ones(4, dtype=bool)
        
        # If action is illegal, replace with random legal action
        if isinstance(action, np.ndarray):
            action = int(action.item())
        
        if not self._current_mask[action]:
            # Pick random legal action
            legal_actions = np.where(self._current_mask)[0]
            action = np.random.choice(legal_actions)
        
        # Execute the (now guaranteed legal) action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update mask for next step
        if not (terminated or truncated):
            self._current_mask = self.env.action_masks() if hasattr(self.env, 'action_masks') else np.ones(4, dtype=bool)
        
        return obs, reward, terminated, truncated, info
    
    def action_masks(self):
        """Return current action mask."""
        if self._current_mask is None:
            return self.env.action_masks() if hasattr(self.env, 'action_masks') else np.ones(4, dtype=bool)
        return self._current_mask
