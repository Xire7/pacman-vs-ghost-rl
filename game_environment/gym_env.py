"""
Gymnasium-compatible Pacman environment with action masking support.
Uses MaskablePPO from sb3-contrib for valid action enforcement.
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


class PacmanEnv(gym.Env):
    """
    Gymnasium environment for Pacman with action masking.
    
    Observation: 45-dimensional vector with normalized features
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
        frame_time: float = 0.05
    ):
        super().__init__()
        
        self.layout_name = layout_name
        self.ghost_type = ghost_type
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.frame_time = frame_time
        
        # Load layout
        self.layout = getLayout(layout_name)
        if self.layout is None:
            raise ValueError(f"Layout '{layout_name}' not found")
        
        self.width = self.layout.width
        self.height = self.layout.height
        
        # Ghost setup
        self.num_ghosts = num_ghosts if num_ghosts is not None else self.layout.getNumGhosts()
        
        # Observation and action spaces
        # 45-dimensional observation space
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(45,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        
        # State tracking
        self.game_state = None
        self.step_count = 0
        self.prev_score = 0
        self.prev_food_count = 0
        self.prev_capsule_count = 0
        self.prev_distance_to_food = float('inf')
        
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
    
    def _init_display(self):
        """Initialize display for rendering."""
        if self._display_initialized:
            return
        if self.render_mode == "human":
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
        
        # Reset tracking
        self.step_count = 0
        self.prev_score = 0
        self.prev_food_count = self.game_state.getNumFood()
        self.prev_capsule_count = len(self.game_state.getCapsules())
        self.prev_distance_to_food = self._get_min_food_distance()
        
        # Initialize display if needed
        if self.render_mode == "human":
            self._init_display()
            if self.display:
                self.display.update(self.game_state.data)
        
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
        
        # Execute ghost actions
        for ghost_idx in range(1, self.num_ghosts + 1):
            if self.game_state.isWin() or self.game_state.isLose():
                break
            ghost_agent = self._create_ghosts()[ghost_idx - 1]
            ghost_action = ghost_agent.getAction(self.game_state)
            self.game_state = self.game_state.generateSuccessor(ghost_idx, ghost_action)
        
        # Update display
        if self.render_mode == "human" and self.display:
            self.display.update(self.game_state.data)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self.game_state.isWin() or self.game_state.isLose()
        truncated = self.step_count >= self.max_steps
        
        info = {
            "score": self.game_state.getScore(),
            "win": self.game_state.isWin(),
            "steps": self.step_count
        }
        
        # Update tracking
        self.prev_score = self.game_state.getScore()
        self.prev_food_count = self.game_state.getNumFood()
        self.prev_capsule_count = len(self.game_state.getCapsules())
        self.prev_distance_to_food = self._get_min_food_distance()
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _calculate_reward(self) -> float:
        """Calculate shaped reward with dense signals."""
        reward = 0.0
        
        # Score delta (normalized)
        score_delta = self.game_state.getScore() - self.prev_score
        reward += score_delta * 0.1
        
        # Win/lose bonuses
        if self.game_state.isWin():
            reward += 50.0
        elif self.game_state.isLose():
            reward -= 30.0
        
        # Food collection bonus
        food_eaten = self.prev_food_count - self.game_state.getNumFood()
        reward += food_eaten * 2.0
        
        # Capsule bonus
        capsule_eaten = self.prev_capsule_count - len(self.game_state.getCapsules())
        reward += capsule_eaten * 5.0
        
        # Distance to food reward (encourage moving toward food)
        curr_dist = self._get_min_food_distance()
        if self.prev_distance_to_food < float('inf') and curr_dist < float('inf'):
            dist_improvement = self.prev_distance_to_food - curr_dist
            reward += dist_improvement * 0.2
        
        # Small step penalty to encourage efficiency
        reward -= 0.01
        
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
        Create 45-dimensional observation vector with:
        - Pacman position (2)
        - Ghost positions relative to pacman (8 = 4 ghosts * 2)
        - Ghost scared times (4)
        - Wall sensors in 4 directions at 3 distances (12)
        - Food sensors in 4 directions (4)
        - Capsule sensors in 4 directions (4)
        - Nearest ghost danger in 4 directions (4)
        - Directional features: nearest food direction (4)
        - Game progress: food ratio, score normalized (3)
        Total: 45
        """
        obs = np.zeros(45, dtype=np.float32)
        
        pacman_pos = self.game_state.getPacmanPosition()
        
        # Normalize position to [-1, 1]
        obs[0] = (pacman_pos[0] / self.width) * 2 - 1
        obs[1] = (pacman_pos[1] / self.height) * 2 - 1
        
        # Ghost relative positions and scared times
        ghost_states = self.game_state.getGhostStates()
        for i in range(4):  # Max 4 ghosts
            if i < len(ghost_states):
                ghost_pos = ghost_states[i].getPosition()
                obs[2 + i*2] = (ghost_pos[0] - pacman_pos[0]) / self.width
                obs[3 + i*2] = (ghost_pos[1] - pacman_pos[1]) / self.height
                obs[10 + i] = ghost_states[i].scaredTimer / SCARED_TIME if SCARED_TIME > 0 else 0
            else:
                obs[2 + i*2] = 0
                obs[3 + i*2] = 0
                obs[10 + i] = 0
        
        # Wall sensors (4 directions, 3 distances each)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
        walls = self.game_state.getWalls()
        
        for d_idx, (dx, dy) in enumerate(directions):
            for dist in range(1, 4):
                x, y = int(pacman_pos[0] + dx * dist), int(pacman_pos[1] + dy * dist)
                if 0 <= x < self.width and 0 <= y < self.height:
                    obs[14 + d_idx*3 + (dist-1)] = 1.0 if walls[x][y] else 0.0
                else:
                    obs[14 + d_idx*3 + (dist-1)] = 1.0  # Out of bounds = wall
        
        # Food sensors in 4 directions (nearest food in each direction)
        food = self.game_state.getFood()
        for d_idx, (dx, dy) in enumerate(directions):
            min_dist = float('inf')
            for dist in range(1, max(self.width, self.height)):
                x, y = int(pacman_pos[0] + dx * dist), int(pacman_pos[1] + dy * dist)
                if 0 <= x < food.width and 0 <= y < food.height:
                    if food[x][y]:
                        min_dist = dist
                        break
                    if walls[x][y]:
                        break
                else:
                    break
            obs[26 + d_idx] = 1.0 / (min_dist + 1) if min_dist < float('inf') else 0.0
        
        # Capsule sensors in 4 directions
        capsules = self.game_state.getCapsules()
        for d_idx, (dx, dy) in enumerate(directions):
            min_dist = float('inf')
            for cap_x, cap_y in capsules:
                if dx != 0 and (cap_x - pacman_pos[0]) / dx > 0 and cap_y == pacman_pos[1]:
                    min_dist = min(min_dist, abs(cap_x - pacman_pos[0]))
                elif dy != 0 and (cap_y - pacman_pos[1]) / dy > 0 and cap_x == pacman_pos[0]:
                    min_dist = min(min_dist, abs(cap_y - pacman_pos[1]))
            obs[30 + d_idx] = 1.0 / (min_dist + 1) if min_dist < float('inf') else 0.0
        
        # Ghost danger in 4 directions (nearest non-scared ghost)
        for d_idx, (dx, dy) in enumerate(directions):
            min_danger = 0.0
            for ghost_state in ghost_states:
                if ghost_state.scaredTimer > 0:
                    continue  # Skip scared ghosts
                ghost_pos = ghost_state.getPosition()
                rel_x = ghost_pos[0] - pacman_pos[0]
                rel_y = ghost_pos[1] - pacman_pos[1]
                # Check if ghost is in this direction
                if dx != 0 and rel_x * dx > 0 and abs(rel_y) < 1:
                    dist = abs(rel_x)
                    min_danger = max(min_danger, 1.0 / (dist + 1))
                elif dy != 0 and rel_y * dy > 0 and abs(rel_x) < 1:
                    dist = abs(rel_y)
                    min_danger = max(min_danger, 1.0 / (dist + 1))
            obs[34 + d_idx] = min_danger
        
        # Directional features: nearest food direction (one-hot)
        nearest_food_dir = self._get_nearest_food_direction()
        obs[38:42] = nearest_food_dir
        
        # Game progress
        total_food = self.layout.totalFood if hasattr(self.layout, 'totalFood') else self.prev_food_count
        if total_food > 0:
            obs[42] = self.game_state.getNumFood() / max(total_food, 1)
        obs[43] = np.tanh(self.game_state.getScore() / 100.0)
        obs[44] = self.step_count / self.max_steps
        
        return obs
    
    def _get_nearest_food_direction(self) -> np.ndarray:
        """Get one-hot direction to nearest food."""
        direction = np.zeros(4, dtype=np.float32)
        pacman_pos = self.game_state.getPacmanPosition()
        food = self.game_state.getFood()
        
        min_dist = float('inf')
        nearest_food = None
        
        for x in range(food.width):
            for y in range(food.height):
                if food[x][y]:
                    dist = abs(pacman_pos[0] - x) + abs(pacman_pos[1] - y)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_food = (x, y)
        
        if nearest_food:
            dx = nearest_food[0] - pacman_pos[0]
            dy = nearest_food[1] - pacman_pos[1]
            if abs(dy) >= abs(dx):
                direction[0 if dy > 0 else 1] = 1.0  # N or S
            else:
                direction[2 if dx > 0 else 3] = 1.0  # E or W
        
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
        """Render the environment."""
        if self.render_mode == "human" and self.display:
            self.display.update(self.game_state.data)
    
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
