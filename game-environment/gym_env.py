import gymnasium as gym
import numpy as np
import random
from pacman import GameState, runGames
from layout import getLayout
from game import Directions
import ghostAgents  # Or your custom ghost

class PacmanEnv(gym.Env):
    """
    Gymnasium wrapper for Berkeley Pac-Man
    """
    def __init__(self, layout_name='mediumGrid', ghost_agents=None, max_steps=None):
        super().__init__()
        
        self.layout_name = layout_name
        self.layout = getLayout(layout_name)
        if self.layout is None:
            raise ValueError(f"Layout '{layout_name}' not found")
        
        self.ghost_agents = ghost_agents
        if ghost_agents is None:
            # use all ghosts defined in the layout
            self.num_ghosts = self.layout.getNumGhosts()
        else:
            # use the number of ghost agents provided
            self.num_ghosts = len(ghost_agents)
        
        # Define observation space 
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(30,),
            dtype=np.float32
        )
        
        # Define action space: 0=North, 1=South, 2=East, 3=West, 4=Stop
        self.action_space = gym.spaces.Discrete(5)
        
        # Episode state variables
        self.game_state = None  # Current GameState object
        self.current_score = 0  # Track score for computing reward deltas
        self.steps = 0  # Count actions taken this episode
        self.max_steps = max_steps  # Maximum steps before truncation (None = no limit)
        self.original_food = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment to start a new episode
        
        Returns:
            observation (np.ndarray): Initial state observation
            info (dict): Empty info dictionary
        """
        super().reset(seed=seed)
        
        # Create new game state
        self.game_state = GameState()
        self.game_state.initialize(self.layout, numGhostAgents=self.num_ghosts)
        
        # Reset episode tracking variables
        self.current_score = 0.0
        self.steps = 0
        self.original_food = self.game_state.getNumFood()  # Store initial food count
        
        # Extract initial observation
        obs = self._extract_state()
        return obs, {}
    
    def step(self, action):
        """Execute action"""
        # Convert action index to Directions
        action_map = {
            0: Directions.NORTH,
            1: Directions.SOUTH,
            2: Directions.EAST,
            3: Directions.WEST,
            4: Directions.STOP
        }
        direction = action_map[action]
        
        # Pac-Man moves
        self.game_state = self.game_state.generateSuccessor(0, direction)
        
        # Ghost moves (indices 1..N)
        for ghost_index in range(1, self.game_state.getNumAgents()):
            if self.game_state.isWin() or self.game_state.isLose():
                break
            
            # Use provided ghost agent or random actions
            if self.ghost_agents is not None and ghost_index - 1 < len(self.ghost_agents):
                ghost_agent = self.ghost_agents[ghost_index - 1]
                ghost_action = ghost_agent.getAction(self.game_state)
            else:
                # random ghost action
                legal_actions = self.game_state.getLegalActions(ghost_index)
                ghost_action = random.choice(legal_actions) if legal_actions else Directions.STOP
            
            self.game_state = self.game_state.generateSuccessor(ghost_index, ghost_action)
        
        # Calculate reward
        new_score = self.game_state.getScore()
        reward = new_score - self.current_score
        self.current_score = new_score
        
        # step counter
        self.steps += 1
        
        # natural termination (win or lose)
        terminated = self.game_state.isWin() or self.game_state.isLose()
        
        # timeout (only if not naturally terminated)
        truncated = False
        if self.max_steps is not None and self.steps >= self.max_steps:
            if not terminated:
                truncated = True
        
        # Extract next state
        next_state = self._extract_state()
        
        # Build info dictionary
        info = {
            'raw_score': new_score,
            'win': self.game_state.isWin(),
            'lose': self.game_state.isLose(),
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _extract_state(self):
        """
        Extract structured state representation
        Returns: np.array of shape (state_dim,)
        """
        state = []
        
        # Pac-Man position (normalized)
        pacman_pos = self.game_state.getPacmanPosition()
        width = self.layout.width
        height = self.layout.height
        state.extend([pacman_pos[0] / width, pacman_pos[1] / height])
        
        # Ghost positions and states
        ghost_states = self.game_state.getGhostStates()
        for ghost in ghost_states:
            ghost_pos = ghost.getPosition()
            state.extend([
                ghost_pos[0] / width,
                ghost_pos[1] / height,
                1.0 if ghost.scaredTimer > 0 else 0.0,
                ghost.scaredTimer / 40.0
            ])
        
        # Pad if fewer ghosts (assume max 4 ghosts)
        while len(state) < 2 + 4 * 4:  # 2 pacman + 4 ghosts * 4 features
            state.extend([0.0, 0.0, 0.0, 0.0])
        
        # Food remaining
        num_food = self.game_state.getNumFood()
        state.append(num_food / 100.0)  # Normalize
        
        # Capsules available
        capsules = self.game_state.getCapsules()
        state.append(len(capsules) / 4.0)  # Max 4 capsules
        
        return np.array(state[:30], dtype=np.float32)  # Trim to exact size