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
                 render_mode=None, max_steps=500):
        super().__init__()
        
        self.ghost_index = ghost_index
        self.pacman_policy = pacman_policy
        self.other_ghost_policies = other_ghost_policies or {}
        
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
    
    def reset(self, *, seed=None, options=None):
        """Reset and return this ghost's observation."""
        super().reset(seed=seed)
        
        obs, info = self.pacman_env.reset(seed=seed)
        self.game_state = self.pacman_env.game_state
        self.game_state.data._agentMoved = 0
        self.steps = 0
        
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
        
        # Pac-Man's turn
        legal_pacman = self.game_state.getLegalActions(0)
        if self.pacman_policy is not None:
            # Sync pacman_env state and get 53-dim observation
            self.pacman_env.game_state = self.game_state
            pacman_obs = self.pacman_env._get_observation()
            
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
                # Other trained ghost
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
                ghost_direction = np.random.choice(legal_ghost)
            
            self.game_state = self.game_state.generateSuccessor(ghost_idx, ghost_direction)
        
        reward = self._calculate_individual_ghost_reward()
        
        terminated = self.game_state.isWin() or self.game_state.isLose()
        truncated = self.steps >= self.pacman_env.max_steps and not terminated
        
        self.steps += 1
        
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
        """Calculate reward for this ghost based on proximity, coordination, and outcomes."""
        reward = 0.0
        
        # Terminal rewards
        if self.game_state.isLose():  # Caught Pac-Man
            return 150.0
        elif self.game_state.isWin():  # Pac-Man won
            return -50.0
        
        # Distance-based rewards
        pacman_pos = self.game_state.getPacmanPosition()
        ghost_pos = self.game_state.getGhostPosition(self.ghost_index)
        ghost_state = self.game_state.getGhostState(self.ghost_index)
        
        dist = manhattanDistance(pacman_pos, ghost_pos)
        
        if ghost_state.scaredTimer == 0:
            # Chase mode: reward proximity to Pac-Man
            if dist <= 1:
                reward += 8.0  # Very close - about to catch!
            elif dist <= 2:
                reward += 4.0
            elif dist <= 4:
                reward += 2.0
            elif dist <= 6:
                reward += 1.0
            else:
                # Penalty for being too far from Pac-Man
                reward -= 0.1 * (dist - 6)
            
            # Reward for closing distance (stronger incentive)
            if hasattr(self, 'prev_dist_to_pacman'):
                dist_delta = self.prev_dist_to_pacman - dist
                if dist_delta > 0:
                    reward += dist_delta * 2.0  # Stronger reward for approaching
                elif dist_delta < 0:
                    reward -= 1.0  # Stronger penalty for retreating
            
            # Coordination bonus when multiple ghosts are close
            other_ghost_dists = []
            for gidx in range(1, self.game_state.getNumAgents()):
                if gidx != self.ghost_index:
                    other_pos = self.game_state.getGhostPosition(gidx)
                    other_dist = manhattanDistance(pacman_pos, other_pos)
                    other_ghost_dists.append(other_dist)
            
            if other_ghost_dists:
                min_other_dist = min(other_ghost_dists)
                if dist <= 3 and min_other_dist <= 3:
                    reward += 1.5  # Potential trap
                    
        else:
            # Scared mode: stay away from Pac-Man
            if dist <= 2:
                reward -= 3.0
            else:
                reward += min(dist * 0.2, 2.0)
        
        self.prev_dist_to_pacman = dist
        reward -= 0.01  # Step penalty
        
        return reward
