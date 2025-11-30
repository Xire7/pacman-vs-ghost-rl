"""
Gymnasium-compatible Pacman environment with action masking support.
Uses MaskablePPO from sb3-contrib for valid action enforcement.

OPTIMIZED VERSION: Improved reward shaping and observation space for faster convergence.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pacman import ClassicGameRules, SCARED_TIME
from game import Directions, Actions, Configuration, AgentState, GameStateData, Game
from layout import getLayout
from ghostAgents import RandomGhost, DirectionalGhost
from graphicsDisplay import PacmanGraphics
from textDisplay import NullGraphics
from state_extractor import extract_ghost_observation


# Observation space dimension (reduced from 53 for cleaner signal)
OBS_DIM = 33


class PacmanEnv(gym.Env):
    """
    Gymnasium environment for Pacman with action masking.
    
    Observation: 33-dimensional vector with carefully selected features:
        [0-1]   Pacman position (x, y) normalized to [-1, 1]
        [2-9]   Ghost relative positions (4 ghosts × 2 coords)
        [10-13] Ghost scared timers (4 ghosts, normalized)
        [14-17] Danger level per direction (N, S, E, W) - inverse distance to nearest dangerous ghost
        [18-21] Food signal per direction (N, S, E, W) - inverse distance to nearest food
        [22-25] Wall immediately adjacent (N, S, E, W) - binary
        [26-27] Nearest food direction (dx, dy normalized)
        [28]    Nearest food distance (normalized)
        [29]    Food remaining ratio
        [30]    Nearest capsule distance (normalized, 1.0 if none)
        [31]    Any ghost scared? (binary)
        [32]    Progress (steps / max_steps)
    
    Actions: 0=North, 1=South, 2=East, 3=West, 4=Stop
    """
    
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 15}
    
    # Action mapping
    ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
    ACTION_TO_INDEX = {d: i for i, d in enumerate(ACTIONS)}
    
    def __init__(
        self,
        layout_name: str = "mediumClassic",
        ghost_type: str = "random",
        num_ghosts: Optional[int] = None,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        frame_time: float = 0.05,
        ghost_policies: Optional[Dict[int, Any]] = None
    ):
        super().__init__()
        
        self.layout_name = layout_name
        self.ghost_type = ghost_type
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.frame_time = frame_time
        self.ghost_policies = ghost_policies or {}
        
        # Load layout
        self.layout = getLayout(layout_name)
        if self.layout is None:
            raise ValueError(f"Layout '{layout_name}' not found")
        
        self.width = self.layout.width
        self.height = self.layout.height
        self.max_dist = self.width + self.height  # Max Manhattan distance
        
        # Ghost setup
        self.num_ghosts = num_ghosts if num_ghosts is not None else self.layout.getNumGhosts()
        
        # Observation and action spaces (reduced dimension for cleaner learning)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        
        # State tracking
        self.game_state = None
        self.step_count = 0
        self.prev_score = 0
        self.prev_food_count = 0
        self.prev_capsule_count = 0
        self.prev_distance_to_food = float('inf')
        self.prev_num_ghosts_eaten = 0
        self.prev_min_ghost_dist = float('inf')
        
        # Display
        self.display = None
        self._display_initialized = False
        
    def _create_ghosts(self) -> List:
        """Create ghost agents based on ghost_type."""
        if self.ghost_type == "random":
            return [RandomGhost(i + 1) for i in range(self.num_ghosts)]
        elif self.ghost_type == "directional":
            return [DirectionalGhost(i + 1) for i in range(self.num_ghosts)]
        else:
            return [RandomGhost(i + 1) for i in range(self.num_ghosts)]
    
    def _get_ghost_action(self, ghost_idx: int) -> str:
        """Get action for a ghost, using RL policy if available."""
        legal_actions = self.game_state.getLegalActions(ghost_idx)
        
        if ghost_idx in self.ghost_policies and self.ghost_policies[ghost_idx] is not None:
            ghost_obs = extract_ghost_observation(self.game_state, ghost_idx)
            action, _ = self.ghost_policies[ghost_idx].predict(ghost_obs, deterministic=False)
            
            action_map = {0: Directions.NORTH, 1: Directions.SOUTH, 
                          2: Directions.EAST, 3: Directions.WEST}
            direction = action_map.get(int(action), Directions.STOP)
            
            if direction in legal_actions:
                return direction
            return np.random.choice(legal_actions)
        else:
            ghost_agent = self._create_ghosts()[ghost_idx - 1]
            return ghost_agent.getAction(self.game_state)
    
    def _init_display(self):
        """Initialize or reinitialize display for rendering."""
        if self.render_mode == "human":
            if self._display_initialized and self.display:
                self.display.initialize(self.game_state.data)
            else:
                self.display = PacmanGraphics(zoom=1.0, frameTime=self.frame_time)
                self.display.initialize(self.game_state.data)
                self._display_initialized = True
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Create initial game state
        rules = ClassicGameRules()
        ghosts = self._create_ghosts()
        
        from pacmanAgents import GreedyAgent
        pacman = GreedyAgent()
        
        game = rules.newGame(self.layout, self.max_steps, pacman, ghosts, NullGraphics(), quiet=True)
        self.game_state = game.state

        for _ in range(5):  # Execute 5 dummy steps to unstuck ghosts
            for ghost_idx in range(1, self.game_state.getNumAgents()):
                legal_actions = self.game_state.getLegalActions(ghost_idx)
                if legal_actions:
                    action = np.random.choice(legal_actions)
                    self.game_state = self.game_state.generateSuccessor(ghost_idx, action)
        
        # Reset tracking
        self.step_count = 0
        self.prev_score = 0
        self.prev_food_count = self.game_state.getNumFood()
        self.original_food = self.prev_food_count
        self.prev_capsule_count = len(self.game_state.getCapsules())
        self.prev_distance_to_food = self._get_min_food_distance()
        self.prev_num_ghosts_eaten = 0
        self.prev_min_ghost_dist = self._get_min_dangerous_ghost_distance()
        
        if self.render_mode == "human":
            self._init_display()
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.step_count += 1
        
        # Convert action to direction
        direction = self.ACTIONS[action]
        legal_actions = self.game_state.getLegalPacmanActions()
        
        # If action not legal, use STOP
        if direction not in legal_actions:
            direction = Directions.STOP
        
        # Execute Pacman's action
        self.game_state = self.game_state.generatePacmanSuccessor(direction)
        
        if self.render_mode == "human" and self.display:
            self.display.update(self.game_state.data)

        # Execute ghost actions
        for ghost_idx in range(1, self.num_ghosts + 1):
            if self.game_state.isWin() or self.game_state.isLose():
                break
            
            ghost_action = self._get_ghost_action(ghost_idx)
            self.game_state = self.game_state.generateSuccessor(ghost_idx, ghost_action)
            
            if self.render_mode == "human" and self.display:
                self.display.update(self.game_state.data)
        
        reward = self._calculate_reward()
        
        # Ensure reward is finite (safety check)
        if not np.isfinite(reward):
            reward = 0.0
        
        terminated = self.game_state.isWin() or self.game_state.isLose()
        truncated = self.step_count >= self.max_steps
        
        info = {
            "score": self.game_state.getScore(),
            "raw_score": self.game_state.getScore(),
            "win": self.game_state.isWin(),
            "lose": self.game_state.isLose(),  
            "steps": self.step_count
        }
        
        # Update tracking for next step
        self.prev_score = self.game_state.getScore()
        self.prev_food_count = self.game_state.getNumFood()
        self.prev_capsule_count = len(self.game_state.getCapsules())
        self.prev_distance_to_food = self._get_min_food_distance()
        self.prev_num_ghosts_eaten = self._count_ghosts_eaten()
        self.prev_min_ghost_dist = self._get_min_dangerous_ghost_distance()
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _count_ghosts_eaten(self) -> int:
        """Count how many ghosts have been eaten (score-based heuristic)."""
        # Each ghost eaten gives 200 points in standard Pac-Man
        return self.game_state.getScore() // 200 if self.game_state.getScore() > 0 else 0
    
    def _get_min_dangerous_ghost_distance(self) -> float:
        """Get Manhattan distance to nearest non-scared ghost."""
        pacman_pos = self.game_state.getPacmanPosition()
        min_dist = float('inf')
        
        for ghost_state in self.game_state.getGhostStates():
            if ghost_state.scaredTimer == 0:  # Not scared = dangerous
                ghost_pos = ghost_state.getPosition()
                dist = abs(pacman_pos[0] - ghost_pos[0]) + abs(pacman_pos[1] - ghost_pos[1])
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _calculate_reward(self) -> float:
        """
        Calculate shaped reward with strong, clear signals.
        
        Key principles:
        1. Terminal rewards dominate: win/lose is most important
        2. Ghost avoidance uses distance-based penalty
        3. Food collection provides steady progress signal
        4. Eating scared ghosts is bonus
        5. Step penalty encourages efficiency
        """
        reward = 0.0
        
        # ============== TERMINAL REWARDS (Dominant!) ==============
        if self.game_state.isWin():
            # Strong bonus for winning - more than sum of all food rewards
            efficiency_bonus = max(0, 1.0 - self.step_count / self.max_steps) * 10.0
            reward += 50.0 + efficiency_bonus  # Total: 50-60
            return reward
        
        if self.game_state.isLose():
            # Strong penalty for losing
            reward -= 100.0
            return reward
        
        # ============== FOOD REWARDS (smaller scale) ==============
        food_eaten = self.prev_food_count - self.game_state.getNumFood()
        if food_eaten > 0:
            # Modest reward per food (total ~100-150 for all food vs 50-60 for winning)
            # This ensures winning bonus > food rewards, but food still matters
            reward += food_eaten * 1.0
            
            # Progressive bonus: reward increases as more food is eaten
            food_progress = 1.0 - (self.game_state.getNumFood() / max(self.original_food, 1))
            reward += food_eaten * 0.5 * food_progress  # Up to +1.5 per food near end
        
        # Reward for moving toward food
        curr_food_dist = self._get_min_food_distance()
        if self.prev_distance_to_food < float('inf') and curr_food_dist < float('inf'):
            dist_improvement = self.prev_distance_to_food - curr_food_dist
            reward += dist_improvement * 0.1
        
        # ============== GHOST INTERACTION REWARDS ==============
        pacman_pos = self.game_state.getPacmanPosition()
        ghost_states = self.game_state.getGhostStates()
        
        # Check for eating scared ghosts
        ghosts_eaten_now = self._count_ghosts_eaten()
        new_ghosts_eaten = ghosts_eaten_now - self.prev_num_ghosts_eaten
        if new_ghosts_eaten > 0:
            reward += new_ghosts_eaten * 8.0  # Nice bonus but less than win
        
        # Danger avoidance - CRITICAL for survival
        min_danger_dist = float('inf')
        for ghost_state in ghost_states:
            if ghost_state.scaredTimer == 0:  # Dangerous ghost
                ghost_pos = ghost_state.getPosition()
                dist = abs(pacman_pos[0] - ghost_pos[0]) + abs(pacman_pos[1] - ghost_pos[1])
                min_danger_dist = min(min_danger_dist, dist)
        
        if min_danger_dist < float('inf'):
            if min_danger_dist <= 1:
                # VERY CLOSE - strong penalty (this leads to death)
                reward -= 5.0
            elif min_danger_dist <= 2:
                # Close - moderate danger
                reward -= 1.5
            elif min_danger_dist <= 3:
                # Approaching danger
                reward -= 0.5
        
        # Reward for escaping danger
        curr_min_ghost_dist = self._get_min_dangerous_ghost_distance()
        if (self.prev_min_ghost_dist < float('inf') and 
            curr_min_ghost_dist < float('inf') and
            self.prev_min_ghost_dist < 4 and 
            curr_min_ghost_dist > self.prev_min_ghost_dist):
            reward += (curr_min_ghost_dist - self.prev_min_ghost_dist) * 0.5
        
        # ============== CAPSULE REWARDS ==============
        capsule_eaten = self.prev_capsule_count - len(self.game_state.getCapsules())
        if capsule_eaten > 0:
            reward += 2.0
            if min_danger_dist < 5:
                reward += 1.0  # Strategic capsule usage
        
        # ============== TIME PRESSURE ==============
        reward -= 0.01  # Slightly higher step penalty to encourage speed
        
        return reward
    
    def _get_min_food_distance(self) -> float:
        """Get Manhattan distance to nearest food."""
        pacman_pos = self.game_state.getPacmanPosition()
        food_grid = self.game_state.getFood()
        
        min_dist = float('inf')
        for x in range(food_grid.width):
            for y in range(food_grid.height):
                if food_grid[x][y]:
                    dist = abs(pacman_pos[0] - x) + abs(pacman_pos[1] - y)
                    min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _get_observation(self) -> np.ndarray:
        """
        Create 33-dimensional observation vector with carefully selected features.
        
        This is a cleaner, more focused observation space than the previous 53-dim version.
        Key improvements:
        - Removed noisy ghost direction features
        - Removed redundant multi-distance wall features  
        - Added clearer danger/food signals per direction
        - Added escape-relevant features
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        
        pacman_pos = self.game_state.getPacmanPosition()
        ghost_states = self.game_state.getGhostStates()
        walls = self.game_state.getWalls()
        food = self.game_state.getFood()
        
        # [0-1] Pacman position normalized to [-1, 1]
        obs[0] = (pacman_pos[0] / self.width) * 2 - 1
        obs[1] = (pacman_pos[1] / self.height) * 2 - 1
        
        # [2-9] Ghost relative positions (4 ghosts × 2 coords)
        # Normalized by map dimensions, clamped to [-1, 1]
        for i in range(4):
            if i < len(ghost_states):
                ghost_pos = ghost_states[i].getPosition()
                obs[2 + i*2] = np.clip((ghost_pos[0] - pacman_pos[0]) / self.max_dist * 2, -1, 1)
                obs[3 + i*2] = np.clip((ghost_pos[1] - pacman_pos[1]) / self.max_dist * 2, -1, 1)
            else:
                obs[2 + i*2] = 0.0
                obs[3 + i*2] = 0.0
        
        # [10-13] Ghost scared timers (4 ghosts, normalized to [0, 1])
        for i in range(4):
            if i < len(ghost_states):
                obs[10 + i] = ghost_states[i].scaredTimer / SCARED_TIME if SCARED_TIME > 0 else 0
            else:
                obs[10 + i] = 0.0
        
        # [14-17] Danger level per direction (N, S, E, W)
        # Higher value = more dangerous (inverse distance to nearest dangerous ghost in that direction)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
        
        for d_idx, (dx, dy) in enumerate(directions):
            max_danger = 0.0
            for ghost_state in ghost_states:
                if ghost_state.scaredTimer > 0:
                    continue  # Scared ghosts are not dangerous
                
                ghost_pos = ghost_state.getPosition()
                rel_x = ghost_pos[0] - pacman_pos[0]
                rel_y = ghost_pos[1] - pacman_pos[1]
                
                # Check if ghost is in this direction (within a cone)
                in_direction = False
                if dx != 0:  # East or West
                    if rel_x * dx > 0 and abs(rel_y) <= abs(rel_x):
                        in_direction = True
                        dist = abs(rel_x) + abs(rel_y)
                else:  # North or South
                    if rel_y * dy > 0 and abs(rel_x) <= abs(rel_y):
                        in_direction = True
                        dist = abs(rel_x) + abs(rel_y)
                
                if in_direction:
                    danger = 1.0 / (dist + 1)  # Inverse distance
                    max_danger = max(max_danger, danger)
            
            obs[14 + d_idx] = max_danger
        
        # [18-21] Food signal per direction (N, S, E, W)
        # Higher value = food is closer in that direction
        for d_idx, (dx, dy) in enumerate(directions):
            min_food_dist = float('inf')
            
            # Ray-cast in this direction until we hit a wall or find food
            for dist in range(1, self.max_dist):
                x = int(pacman_pos[0] + dx * dist)
                y = int(pacman_pos[1] + dy * dist)
                
                if not (0 <= x < self.width and 0 <= y < self.height):
                    break
                if walls[x][y]:
                    break
                if food[x][y]:
                    min_food_dist = dist
                    break
            
            obs[18 + d_idx] = 1.0 / (min_food_dist + 1) if min_food_dist < float('inf') else 0.0
        
        # [22-25] Wall immediately adjacent (N, S, E, W) - binary
        for d_idx, (dx, dy) in enumerate(directions):
            x = int(pacman_pos[0] + dx)
            y = int(pacman_pos[1] + dy)
            if 0 <= x < self.width and 0 <= y < self.height:
                obs[22 + d_idx] = 1.0 if walls[x][y] else 0.0
            else:
                obs[22 + d_idx] = 1.0
        
        # [26-27] Direction to nearest food (normalized dx, dy)
        nearest_food = self._find_nearest_food()
        if nearest_food is not None:
            dx = nearest_food[0] - pacman_pos[0]
            dy = nearest_food[1] - pacman_pos[1]
            dist = abs(dx) + abs(dy)
            if dist > 0:
                obs[26] = dx / dist  # Normalized direction
                obs[27] = dy / dist
            else:
                obs[26] = 0.0
                obs[27] = 0.0
            # [28] Nearest food distance (normalized, 1 = far, 0 = close)
            obs[28] = 1.0 - min(dist / self.max_dist, 1.0)
        else:
            obs[26] = 0.0
            obs[27] = 0.0
            obs[28] = 1.0  # No food = close (will win)
        
        # [29] Food remaining ratio (1 = all food, 0 = no food)
        obs[29] = self.game_state.getNumFood() / max(self.original_food, 1)
        
        # [30] Nearest capsule distance (normalized, 1 = close, 0 = far or none)
        capsules = self.game_state.getCapsules()
        if capsules:
            min_cap_dist = min(abs(pacman_pos[0] - cx) + abs(pacman_pos[1] - cy) 
                              for cx, cy in capsules)
            obs[30] = 1.0 - min(min_cap_dist / self.max_dist, 1.0)
        else:
            obs[30] = 0.0  # No capsules
        
        # [31] Any ghost scared? (binary - important for strategy!)
        obs[31] = 1.0 if any(gs.scaredTimer > 0 for gs in ghost_states) else 0.0
        
        # [32] Progress (steps / max_steps)
        obs[32] = self.step_count / self.max_steps
        
        # Safety: Replace any NaN or Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return np.clip(obs, -1.0, 1.0)
    
    def _find_nearest_food(self) -> Optional[Tuple[int, int]]:
        """Find the position of the nearest food pellet."""
        pacman_pos = self.game_state.getPacmanPosition()
        food = self.game_state.getFood()
        
        min_dist = float('inf')
        nearest = None
        
        for x in range(food.width):
            for y in range(food.height):
                if food[x][y]:
                    dist = abs(pacman_pos[0] - x) + abs(pacman_pos[1] - y)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (x, y)
        
        return nearest
    
    def _get_nearest_food_direction(self) -> np.ndarray:
        """Get one-hot direction to nearest food (legacy compatibility)."""
        direction = np.zeros(4, dtype=np.float32)
        nearest = self._find_nearest_food()
        
        if nearest:
            pacman_pos = self.game_state.getPacmanPosition()
            dx = nearest[0] - pacman_pos[0]
            dy = nearest[1] - pacman_pos[1]
            if abs(dy) >= abs(dx):
                direction[0 if dy > 0 else 1] = 1.0
            else:
                direction[2 if dx > 0 else 3] = 1.0
        
        return direction
    
    def action_masks(self) -> np.ndarray:
        """Return valid action mask for MaskablePPO."""
        legal_actions = self.game_state.getLegalPacmanActions()
        mask = np.zeros(5, dtype=bool)
        for action in legal_actions:
            if action in self.ACTION_TO_INDEX:
                mask[self.ACTION_TO_INDEX[action]] = True
        return mask
    
    def render(self):
        """Render the environment. Display is updated during step()."""
        pass
    
    def close(self):
        """Clean up resources."""
        if self.display:
            try:
                self.display.finish()
            except:
                pass
            self.display = None
            self._display_initialized = False


def make_pacman_env(
    layout_name: str = "mediumClassic",
    ghost_type: str = "random",
    num_ghosts: Optional[int] = None,
    max_steps: int = 500,
    render_mode: Optional[str] = None,
    frame_time: float = 0.05
) -> PacmanEnv:
    """Create a Pacman environment."""
    return PacmanEnv(
        layout_name=layout_name,
        ghost_type=ghost_type,
        num_ghosts=num_ghosts,
        max_steps=max_steps,
        render_mode=render_mode,
        frame_time=frame_time
    )


# Alias for backward compatibility
make_masked_pacman_env = make_pacman_env
