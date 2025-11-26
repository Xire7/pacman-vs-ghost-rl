"""
Gymnasium wrapper for Berkeley Pac-Man environment.
Compatible with stable-baselines3 for PPO training.
"""

import gymnasium as gym
import numpy as np
import random
from pacman import GameState
from layout import getLayout
from game import Directions, Actions
import ghostAgents
import graphicsDisplay
import textDisplay
import graphicsUtils


class PacmanEnv(gym.Env):
    """
    Gymnasium wrapper for Berkeley Pac-Man.
    
    Observation Space: 30-dimensional Box[-1, 1]
        Features include Pac-Man position, ghost positions/states,
        food information, score, and directional features.
    
    Action Space: Discrete(5)
        0=NORTH, 1=SOUTH, 2=EAST, 3=WEST, 4=STOP
    """
    
    metadata = {"render_modes": ["human", "text", "rgb_array"], "render_fps": 10}
    
    # Action mapping
    ACTION_MAP = {
        0: Directions.NORTH,
        1: Directions.SOUTH,
        2: Directions.EAST,
        3: Directions.WEST,
        4: Directions.STOP
    }
    
    DIRECTION_TO_ACTION = {v: k for k, v in ACTION_MAP.items()}
    
    def __init__(self, layout_name='mediumGrid', ghost_agents=None, max_steps=500, 
                 render_mode=None, reward_shaping=True):
        """
        Initialize the Pac-Man environment.
        
        Args:
            layout_name: Name of layout file in layouts/ directory
            ghost_agents: List of ghost agent objects, or None for random ghosts
            max_steps: Maximum steps before truncation
            render_mode: 'human', 'text', 'rgb_array', or None
            reward_shaping: If True, add shaped rewards beyond score delta
        """
        super().__init__()
        
        self.layout_name = layout_name
        self.layout = getLayout(layout_name)
        if self.layout is None:
            raise ValueError(f"Layout '{layout_name}' not found. Check layouts/ directory.")
        
        # Store ghost agent configuration
        self._ghost_agent_config = ghost_agents
        self.num_ghosts = len(ghost_agents) if ghost_agents else self.layout.getNumGhosts()
        
        # Observation space: 30 features, normalized to [-1, 1]
        # Using [-1, 1] range to handle negative values like relative positions
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(30,),
            dtype=np.float32
        )
        
        # Action space: 5 discrete actions
        self.action_space = gym.spaces.Discrete(5)
        
        # Episode tracking
        self.game_state = None
        self.current_score = 0.0
        self.steps = 0
        self.max_steps = max_steps
        self.original_food = 0
        self.prev_food_count = 0
        self.prev_capsule_count = 0
        self.prev_pacman_pos = None
        self.reward_shaping = reward_shaping
        
        # Rendering
        self.render_mode = render_mode
        self.display = None
        self._display_initialized = False
        
    def _create_ghost_agents(self):
        """Create ghost agents for the episode."""
        if self._ghost_agent_config is not None:
            # Use provided ghost agents (recreate to reset internal state)
            return [type(g)(g.index) for g in self._ghost_agent_config]
        else:
            # Create random ghosts
            return [ghostAgents.RandomGhost(i + 1) for i in range(self.num_ghosts)]
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to start a new episode.
        
        Returns:
            observation: Initial state observation (np.ndarray)
            info: Additional information dict
        """
        super().reset(seed=seed)
        
        # Seed the random number generators
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create new game state
        self.game_state = GameState()
        self.game_state.initialize(self.layout, numGhostAgents=self.num_ghosts)
        
        # Create ghost agents for this episode
        self.ghost_agents = self._create_ghost_agents()
        
        # Reset tracking variables
        self.current_score = 0.0
        self.steps = 0
        self.original_food = self.game_state.getNumFood()
        self.prev_food_count = self.original_food
        self.prev_capsule_count = len(self.game_state.getCapsules())
        self.prev_pacman_pos = self.game_state.getPacmanPosition()
        
        # Initialize display for rendering
        if self.render_mode == 'human' and not self._display_initialized:
            self.display = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.05)
            self.display.initialize(self.game_state.data)
            self._display_initialized = True
            graphicsUtils.refresh()
        elif self.render_mode == 'human' and self._display_initialized:
            # Reinitialize for new episode
            self.display.initialize(self.game_state.data)
            graphicsUtils.refresh()
        
        # Extract observation
        obs = self._extract_observation()
        
        info = {
            'raw_score': 0,
            'food_remaining': self.original_food,
            'capsules_remaining': self.prev_capsule_count,
        }
        
        return obs, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action (0-4), or numpy array containing the action
            
        Returns:
            observation: Next state observation
            reward: Reward for this step
            terminated: Whether episode ended naturally (win/lose)
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # Handle numpy array input (from vectorized envs or model.predict)
        if isinstance(action, np.ndarray):
            action = action.item() if action.ndim == 0 else int(action.flat[0])
        else:
            action = int(action)
        
        # Convert action to direction
        direction = self.ACTION_MAP[action]
        
        # Get legal actions for Pac-Man
        legal_actions = self.game_state.getLegalActions(0)
        
        # Handle illegal action: use a random legal action instead
        # This is better for exploration than STOP
        if direction not in legal_actions:
            if legal_actions:
                # Use seeded random if available
                direction = random.choice(legal_actions)
            else:
                direction = Directions.STOP
        
        # Store previous state for reward shaping
        prev_score = self.game_state.getScore()
        prev_ghost_dists = self._get_ghost_distances()
        
        # Apply Pac-Man's action
        self.game_state = self.game_state.generatePacmanSuccessor(direction)
        
        # Update display after Pac-Man moves (before ghosts)
        if self.render_mode == 'human' and self.display is not None:
            self.display.update(self.game_state.data)
            graphicsUtils.refresh()
        
        # Execute ghost actions (if game not over)
        for ghost_idx in range(1, self.game_state.getNumAgents()):
            if self.game_state.isWin() or self.game_state.isLose():
                break
            
            # Get ghost action
            if ghost_idx - 1 < len(self.ghost_agents):
                ghost_agent = self.ghost_agents[ghost_idx - 1]
                ghost_action = ghost_agent.getAction(self.game_state)
            else:
                legal = self.game_state.getLegalActions(ghost_idx)
                ghost_action = random.choice(legal) if legal else Directions.STOP
            
            self.game_state = self.game_state.generateSuccessor(ghost_idx, ghost_action)
            
            # Update display after each ghost moves
            if self.render_mode == 'human' and self.display is not None:
                self.display.update(self.game_state.data)
                graphicsUtils.refresh()
        
        # Calculate reward
        new_score = self.game_state.getScore()
        reward = new_score - prev_score  # Base reward is score delta
        
        # Apply reward shaping if enabled
        if self.reward_shaping:
            reward = self._shape_reward(reward, prev_ghost_dists)
        
        self.current_score = new_score
        self.steps += 1
        
        # Check termination conditions
        terminated = self.game_state.isWin() or self.game_state.isLose()
        truncated = (self.max_steps is not None and 
                    self.steps >= self.max_steps and 
                    not terminated)
        
        # Update tracking for next step
        self.prev_food_count = self.game_state.getNumFood()
        self.prev_capsule_count = len(self.game_state.getCapsules())
        self.prev_pacman_pos = self.game_state.getPacmanPosition()
        
        # Extract observation
        obs = self._extract_observation()
        
        # Build info dict
        info = {
            'raw_score': new_score,
            'win': self.game_state.isWin(),
            'lose': self.game_state.isLose(),
            'food_remaining': self.prev_food_count,
            'capsules_remaining': self.prev_capsule_count,
            'steps': self.steps,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_ghost_distances(self):
        """Get Manhattan distances to each ghost."""
        pacman_pos = self.game_state.getPacmanPosition()
        distances = []
        for ghost_state in self.game_state.getGhostStates():
            ghost_pos = ghost_state.getPosition()
            dist = abs(pacman_pos[0] - ghost_pos[0]) + abs(pacman_pos[1] - ghost_pos[1])
            distances.append((dist, ghost_state.scaredTimer > 0))
        return distances
    
    def _shape_reward(self, base_reward, prev_ghost_dists):
        """
        Add reward shaping to encourage good behavior.
        
        Shaping includes:
        - Bonus for eating food (to make food more attractive)
        - Bonus for getting closer to scared ghosts (ghost hunting)
        - Small penalty for getting closer to dangerous ghosts
        - Win/lose bonuses (in addition to score)
        
        The base game score changes are:
        - -1 per time step
        - +10 for eating food
        - +200 for eating scared ghost
        - +500 for winning (all food eaten)
        - -500 for dying (ghost collision)
        """
        shaped_reward = 0.0
        
        # Detect what happened this step
        ate_food = base_reward >= 9  # +10 food - 1 time = +9
        ate_ghost = base_reward >= 199  # +200 ghost - 1 time = +199
        won = self.game_state.isWin()
        died = self.game_state.isLose()
        
        # === Reward components (all pre-scaled for PPO) ===
        
        # Time penalty (encourage efficiency)
        shaped_reward -= 0.01
        
        # Food reward (make eating food clearly positive)
        if ate_food and not won:  # Don't double-count win
            shaped_reward += 0.5  # Significant positive reward for food
        
        # Ghost eating reward
        if ate_ghost:
            shaped_reward += 2.0  # Big reward for eating ghost
        
        # Win bonus
        if won:
            shaped_reward += 10.0  # Large win bonus
        
        # Death penalty
        if died:
            shaped_reward -= 5.0  # Death penalty (but not overwhelming)
        
        # === Distance-based shaping ===
        current_ghost_dists = self._get_ghost_distances()
        for i, (curr_dist, is_scared) in enumerate(current_ghost_dists):
            if i < len(prev_ghost_dists):
                prev_dist, was_scared = prev_ghost_dists[i]
                if is_scared and was_scared:
                    # Reward for getting closer to scared ghosts
                    dist_delta = prev_dist - curr_dist
                    if dist_delta > 0:
                        shaped_reward += 0.1 * dist_delta  # Bonus for approaching scared ghost
                elif not is_scared and not was_scared:
                    # Small penalty for getting too close to dangerous ghosts
                    if curr_dist < 2:
                        shaped_reward -= 0.05  # Discourage being very close to danger
        
        return shaped_reward
    
    def _extract_observation(self):
        """
        Extract a 30-dimensional observation vector.
        
        Feature layout:
            [0-1]   Pac-Man position (normalized)
            [2-3]   Pac-Man velocity direction (one-hot x, y components)
            [4-19]  Ghost info: 4 ghosts × (x, y, scared, scared_timer)
            [20]    Remaining food ratio
            [21]    Remaining capsules (normalized)
            [22]    Nearest food distance (normalized)
            [23]    Nearest dangerous ghost distance (normalized)
            [24]    Nearest scared ghost distance (normalized)  
            [25]    Score (normalized and clipped)
            [26-29] Direction to nearest food (dx, dy normalized, and valid flags)
        
        Returns:
            np.ndarray: 30-dimensional feature vector in [-1, 1]
        """
        width = self.layout.width
        height = self.layout.height
        max_dist = width + height  # Maximum Manhattan distance
        
        features = []
        
        # === Pac-Man position (2 features) ===
        pacman_pos = self.game_state.getPacmanPosition()
        # Normalize to [0, 1] then shift to [-1, 1] range
        features.append(2.0 * pacman_pos[0] / (width - 1) - 1.0)
        features.append(2.0 * pacman_pos[1] / (height - 1) - 1.0)
        
        # === Pac-Man direction (2 features) ===
        pacman_state = self.game_state.data.agentStates[0]
        direction = pacman_state.getDirection()
        dx, dy = Actions.directionToVector(direction)
        features.append(float(dx))  # -1, 0, or 1
        features.append(float(dy))  # -1, 0, or 1
        
        # === Ghost information (16 features: 4 ghosts × 4 features) ===
        ghost_states = self.game_state.getGhostStates()
        for i in range(4):  # Always 4 ghost slots
            if i < len(ghost_states):
                ghost = ghost_states[i]
                ghost_pos = ghost.getPosition()
                # Position normalized to [-1, 1]
                features.append(2.0 * ghost_pos[0] / (width - 1) - 1.0)
                features.append(2.0 * ghost_pos[1] / (height - 1) - 1.0)
                # Scared state: -1 if scared, +1 if dangerous
                features.append(-1.0 if ghost.scaredTimer > 0 else 1.0)
                # Scared timer normalized
                features.append(ghost.scaredTimer / 40.0)
            else:
                # Padding for missing ghosts (mark as "not present")
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        # === Food information (3 features) ===
        food_positions = self.game_state.getFood().asList()
        num_food = len(food_positions)
        
        # Food ratio (how much food remains)
        if self.original_food > 0:
            features.append(2.0 * num_food / self.original_food - 1.0)
        else:
            features.append(-1.0)
        
        # Capsules remaining (normalized)
        capsules = self.game_state.getCapsules()
        features.append(len(capsules) / 4.0 * 2.0 - 1.0)  # Assume max 4 capsules
        
        # Nearest food distance
        if food_positions:
            distances = [abs(pacman_pos[0] - fx) + abs(pacman_pos[1] - fy) 
                        for fx, fy in food_positions]
            nearest_food_dist = min(distances)
            features.append(1.0 - 2.0 * nearest_food_dist / max_dist)  # Closer = higher value
        else:
            features.append(1.0)  # No food = we're done
        
        # === Ghost distances (2 features) ===
        dangerous_ghost_dists = []
        scared_ghost_dists = []
        for ghost in ghost_states:
            ghost_pos = ghost.getPosition()
            dist = abs(pacman_pos[0] - ghost_pos[0]) + abs(pacman_pos[1] - ghost_pos[1])
            if ghost.scaredTimer > 0:
                scared_ghost_dists.append(dist)
            else:
                dangerous_ghost_dists.append(dist)
        
        # Nearest dangerous ghost (closer = more negative for danger signal)
        if dangerous_ghost_dists:
            nearest_danger = min(dangerous_ghost_dists)
            features.append(2.0 * nearest_danger / max_dist - 1.0)  # Closer = lower value = danger
        else:
            features.append(1.0)  # No dangerous ghosts
        
        # Nearest scared ghost (closer = higher value = opportunity)
        if scared_ghost_dists:
            nearest_scared = min(scared_ghost_dists)
            features.append(1.0 - 2.0 * nearest_scared / max_dist)  # Closer = higher value
        else:
            features.append(-1.0)  # No scared ghosts
        
        # === Score (1 feature) ===
        # Normalize score: typical range is -500 to +1000
        score = self.game_state.getScore()
        normalized_score = np.clip(score / 500.0, -1.0, 1.0)
        features.append(normalized_score)
        
        # === Direction to nearest food (4 features) ===
        if food_positions:
            # Find nearest food
            distances_with_pos = [(abs(pacman_pos[0] - fx) + abs(pacman_pos[1] - fy), (fx, fy))
                                  for fx, fy in food_positions]
            _, nearest_food_pos = min(distances_with_pos, key=lambda x: x[0])
            
            # Direction vector (normalized)
            dx = nearest_food_pos[0] - pacman_pos[0]
            dy = nearest_food_pos[1] - pacman_pos[1]
            
            # Normalize direction
            mag = abs(dx) + abs(dy)
            if mag > 0:
                features.append(dx / mag)
                features.append(dy / mag)
            else:
                features.append(0.0)
                features.append(0.0)
            
            # Food exists flags
            features.append(1.0)  # Food in X direction exists
            features.append(1.0)  # Food in Y direction exists
        else:
            features.extend([0.0, 0.0, -1.0, -1.0])
        
        # === Ensure exactly 30 features ===
        while len(features) < 30:
            features.append(0.0)
        
        obs = np.array(features[:30], dtype=np.float32)
        
        # Clip to observation space bounds
        obs = np.clip(obs, -1.0, 1.0)
        
        return obs
    
    def render(self):
        """Render the current state."""
        if self.render_mode == 'human':
            if self.display is not None:
                self.display.update(self.game_state.data)
        elif self.render_mode == 'text':
            print(str(self.game_state))
        elif self.render_mode == 'rgb_array':
            # Return a simple representation (not implemented for graphics)
            return None
        return None
    
    def close(self):
        """Clean up resources."""
        if self.display is not None:
            try:
                self.display.finish()
            except:
                pass
            self._display_initialized = False
            self.display = None
    
    def get_legal_action_mask(self):
        """
        Get a mask of legal actions (useful for action masking in RL).
        
        Returns:
            np.ndarray: Boolean mask of shape (5,) where True = legal action
        """
        legal_actions = self.game_state.getLegalActions(0)
        mask = np.zeros(5, dtype=np.bool_)
        for action in legal_actions:
            if action in self.DIRECTION_TO_ACTION:
                mask[self.DIRECTION_TO_ACTION[action]] = True
        return mask


def make_pacman_env(layout_name='smallGrid', ghost_type='random', num_ghosts=None,
                    max_steps=500, render_mode=None, reward_shaping=True):
    """
    Factory function to create a PacmanEnv with specified configuration.
    
    Args:
        layout_name: Name of the layout (e.g., 'smallGrid', 'mediumClassic')
        ghost_type: 'random' or 'directional'
        num_ghosts: Number of ghosts (None = use layout default)
        max_steps: Maximum steps per episode
        render_mode: 'human', 'text', or None
        reward_shaping: Whether to apply reward shaping
        
    Returns:
        PacmanEnv: Configured environment instance
    """
    # Get layout to determine number of ghosts
    layout = getLayout(layout_name)
    if layout is None:
        raise ValueError(f"Layout '{layout_name}' not found")
    
    if num_ghosts is None:
        num_ghosts = layout.getNumGhosts()
    else:
        num_ghosts = min(num_ghosts, layout.getNumGhosts())
    
    # Create ghost agents
    if ghost_type == 'random':
        ghosts = [ghostAgents.RandomGhost(i + 1) for i in range(num_ghosts)]
    elif ghost_type == 'directional':
        ghosts = [ghostAgents.DirectionalGhost(i + 1) for i in range(num_ghosts)]
    else:
        ghosts = None  # Will use random by default
    
    return PacmanEnv(
        layout_name=layout_name,
        ghost_agents=ghosts,
        max_steps=max_steps,
        render_mode=render_mode,
        reward_shaping=reward_shaping
    )
