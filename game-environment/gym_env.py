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
        
        # validate action placeholder (can change this to apply negative reward if we're into that)
        legal_actions = self.game_state.getLegalActions(0)
        if direction not in legal_actions:
            direction = random.choice(legal_actions) if legal_actions else Directions.STOP
        
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
        """Extract structured state representation for RL agent
        
        Feature vector layout:
            [0-1]   Pac-Man position (x, y) normalized
            [2-17]  Up to 4 ghosts: (x, y, scared_flag, scared_timer) x 4
            [18]    Remaining food count normalized
            [19]    Remaining capsules normalized
            [20]    Nearest food distance normalized
            [21]    Current score normalized
            [22]    Original food count normalized
            [23]    Time fraction (steps/max_steps)
            [24-29] Reserved/padding
        
        Returns:
            np.ndarray: 30-dimensional feature vector, all values in [0, 1]
        """
        state = []
        
        # Get layout dimensions for normalization
        width = self.layout.width
        height = self.layout.height
        
        # Pac-Man position (normalized to [0,1])
        pacman_pos = self.game_state.getPacmanPosition()
        state.extend([pacman_pos[0] / width, pacman_pos[1] / height])
        
        # Ghost positions and states (up to 4 ghosts, 4 features each)
        ghost_states = self.game_state.getGhostStates()
        for ghost in ghost_states[:4]:  # first 4 ghosts only
            ghost_pos = ghost.getPosition()
            state.extend([
                ghost_pos[0] / width,
                ghost_pos[1] / height,
                1.0 if ghost.scaredTimer > 0 else 0.0,  # Binary scared flag
                ghost.scaredTimer / 40.0  # Timer normalized by SCARED_TIME
            ])
        
        # Pad with zeros if fewer than 4 ghosts
        for _ in range(4 - len(ghost_states)):
            state.extend([0.0, 0.0, 0.0, 0.0])
        
        # Food remaining
        num_food = self.game_state.getNumFood()
        state.append(num_food / 100.0)
        
        # Capsules available
        capsules = self.game_state.getCapsules()
        state.append(len(capsules) / 4.0) # max 4 capsules
        
        # Nearest food distance (Manhattan distance)
        food_positions = self.game_state.getFood().asList()
        if food_positions:
            distances = [abs(pacman_pos[0] - fx) + abs(pacman_pos[1] - fy) 
                        for fx, fy in food_positions]
            nearest_food_dist = min(distances) / (width + height)
        else:
            nearest_food_dist = 0.0
        state.append(nearest_food_dist)
        
        # Current score
        state.append(self.game_state.getScore() / 1000.0)
        
        # Original food count (context for progress tracking)
        state.append(self.original_food / 100.0 if self.original_food else 0.0)
        
        # Time fraction (how much of max_steps used)
        if self.max_steps is not None and self.max_steps > 0:
            time_frac = min(self.steps / self.max_steps, 1.0)
        else:
            time_frac = 0.0
        state.append(time_frac)
        
        # Padding
        while len(state) < 30:
            state.append(0.0)
        
        return np.array(state[:30], dtype=np.float32)
    
def render(self, mode='human'):
    if mode == 'human':
        if hasattr(self, 'display') and self.display:
            # graphic
            self.display.update(self.game_state.data)
        else:
            # text
            print(str(self.game_state))
    return None