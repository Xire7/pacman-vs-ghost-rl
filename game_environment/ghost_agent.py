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
                 render_mode=None, max_steps=500):
        """
        Args:
            ghost_index: Which ghost this environment controls (1, 2, 3, or 4)
            other_ghost_policies: Dict of {ghost_idx: trained_model} for other ghosts
        """
        super().__init__()
        
        self.ghost_index = ghost_index
        self.pacman_policy = pacman_policy
        self.other_ghost_policies = other_ghost_policies or {}
        
        self.pacman_env = PacmanEnv(
            layout_name=layout_name,
            ghost_agents=None,
            max_steps=max_steps,
            render_mode=render_mode,
            reward_shaping=False
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(4)
        
        self.game_state = None
        self.steps = 0
        self.prev_ghost_distances = {}
    
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
        
        # Execute Pac-Man move
        legal_pacman = self.game_state.getLegalActions(0)
        if self.pacman_policy is not None:
            pacman_obs = extract_pacman_observation(
                self.game_state, self.pacman_env.original_food
            )
            pacman_direction = get_best_legal_action(
                self.pacman_policy, pacman_obs, legal_pacman
            )
        else:
            pacman_direction = np.random.choice(legal_pacman)
        
        self.game_state = self.game_state.generateSuccessor(0, pacman_direction)
        
        # Execute all ghost moves
        for ghost_idx in range(1, self.game_state.getNumAgents()):
            if self.game_state.isWin() or self.game_state.isLose():
                break
            
            legal_ghost = self.game_state.getLegalActions(ghost_idx)
            
            if ghost_idx == self.ghost_index:
                # ✅ This is OUR ghost - use the action we're training
                ghost_direction = my_action
                if ghost_direction not in legal_ghost:
                    ghost_direction = np.random.choice(legal_ghost)
            
            elif ghost_idx in self.other_ghost_policies:
                # ✅ Other ghost has a trained policy
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
            ghost_obs = np.zeros(9, dtype=np.float32)
        
        info = {
            'ghost_won': self.game_state.isLose(),
            'pacman_won': self.game_state.isWin(),
            'score': self.game_state.getScore()
        }
        
        return ghost_obs, reward, terminated, truncated, info
    
    def _calculate_individual_ghost_reward(self):
        """
        Reward for THIS ghost specifically.
        
        Encourages individual contribution while maintaining team goal.
        """
        reward = 0.0
        
        # Team rewards (all ghosts get these)
        if self.game_state.isLose():
            reward += 100.0  # Caught Pac-Man!
        elif self.game_state.isWin():
            reward -= 50.0   # Pac-Man won
        
        # Individual reward: proximity to Pac-Man
        pacman_pos = self.game_state.getPacmanPosition()
        ghost_pos = self.game_state.getGhostPosition(self.ghost_index)
        ghost_state = self.game_state.getGhostState(self.ghost_index)
        
        dist = manhattanDistance(pacman_pos, ghost_pos)
        
        # Reward for being close (if not scared)
        if ghost_state.scaredTimer == 0:
            if dist <= 1:
                reward += 10.0  # Very close!
            elif dist <= 3:
                reward += 2.0   # Approaching
            else:
                reward -= 0.1 * dist  # Too far
        else:
            # If scared, reward for staying away
            reward += 0.05 * dist
        
        reward -= 0.01  # Small time penalty
        
        return reward