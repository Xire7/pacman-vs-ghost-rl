"""
State extraction utilities for Pac-Man RL agents.

This module provides functions to extract observation vectors from game states
for both Pac-Man and Ghost agents. All observations are normalized to [-1, 1]
or [0, 1] ranges for stable neural network training.
"""

import numpy as np
from typing import List, Tuple, Optional

from game import Actions, Directions
from util import manhattanDistance


# =============================================================================
# Constants
# =============================================================================

PACMAN_OBS_DIM = 33
GHOST_OBS_DIM = 25
SCARED_TIME = 40  # Default scared timer duration
MAX_GHOSTS = 4

# Direction vectors: North, South, East, West
DIRECTION_VECTORS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


# =============================================================================
# Pac-Man Observation Extraction
# =============================================================================

def extract_pacman_observation(
    game_state, 
    original_food_count: int,
    step_count: int = 0,
    max_steps: int = 500
) -> np.ndarray:
    """
    Extract a 33-dimensional observation vector for Pac-Man.
    
    Features (all normalized to [-1, 1]):
        [0-1]   Pac-Man position (x, y)
        [2-9]   Ghost relative positions (4 ghosts × 2)
        [10-13] Ghost scared timers (4 ghosts)
        [14-17] Danger level per direction (N, S, E, W)
        [18-21] Food signal per direction (N, S, E, W)
        [22-25] Wall adjacent (N, S, E, W)
        [26-27] Direction to nearest food (dx, dy)
        [28]    Nearest food distance (inverted: 1=close)
        [29]    Food remaining ratio
        [30]    Nearest capsule distance (inverted)
        [31]    Any ghost scared (binary)
        [32]    Time progress
    
    Args:
        game_state: The current Pac-Man game state
        original_food_count: Total food count at episode start
        step_count: Current step in episode
        max_steps: Maximum steps per episode
    
    Returns:
        np.ndarray: 33-dimensional float32 observation vector
    """
    layout = game_state.data.layout
    width, height = layout.width, layout.height
    max_dist = width + height
    walls = game_state.getWalls()
    food = game_state.getFood()
    
    obs = np.zeros(PACMAN_OBS_DIM, dtype=np.float32)
    
    pacman_pos = game_state.getPacmanPosition()
    ghost_states = game_state.getGhostStates()
    
    idx = 0
    
    # [0-1] Pac-Man position
    obs[idx] = (pacman_pos[0] / width) * 2 - 1
    obs[idx + 1] = (pacman_pos[1] / height) * 2 - 1
    idx += 2
    
    # [2-9] Ghost relative positions (4 ghosts × 2)
    for i in range(MAX_GHOSTS):
        if i < len(ghost_states):
            gpos = ghost_states[i].getPosition()
            obs[idx] = np.clip((gpos[0] - pacman_pos[0]) / max_dist * 2, -1, 1)
            obs[idx + 1] = np.clip((gpos[1] - pacman_pos[1]) / max_dist * 2, -1, 1)
        idx += 2
    
    # [10-13] Ghost scared timers
    for i in range(MAX_GHOSTS):
        if i < len(ghost_states):
            obs[idx] = ghost_states[i].scaredTimer / SCARED_TIME
        idx += 1
    
    # [14-17] Danger per direction (N, S, E, W)
    for dx, dy in DIRECTION_VECTORS:
        obs[idx] = _get_direction_danger(pacman_pos, ghost_states, dx, dy)
        idx += 1
    
    # [18-21] Food per direction (N, S, E, W)
    for dx, dy in DIRECTION_VECTORS:
        obs[idx] = _get_direction_food(pacman_pos, walls, food, dx, dy, width, height, max_dist)
        idx += 1
    
    # [22-25] Adjacent walls (N, S, E, W)
    for dx, dy in DIRECTION_VECTORS:
        x, y = int(pacman_pos[0] + dx), int(pacman_pos[1] + dy)
        obs[idx] = 1.0 if (not (0 <= x < width and 0 <= y < height) or walls[x][y]) else 0.0
        idx += 1
    
    # [26-28] Nearest food direction and distance
    nearest_food = _find_nearest_food(pacman_pos, food)
    if nearest_food:
        dx = nearest_food[0] - pacman_pos[0]
        dy = nearest_food[1] - pacman_pos[1]
        dist = abs(dx) + abs(dy)
        if dist > 0:
            obs[idx] = dx / dist
            obs[idx + 1] = dy / dist
        obs[idx + 2] = 1.0 - min(dist / max_dist, 1.0)
    else:
        obs[idx + 2] = 1.0  # No food = about to win
    idx += 3
    
    # [29] Food remaining ratio
    obs[idx] = game_state.getNumFood() / max(original_food_count, 1)
    idx += 1
    
    # [30] Nearest capsule distance (inverted)
    capsules = game_state.getCapsules()
    if capsules:
        min_cap = min(abs(pacman_pos[0] - c[0]) + abs(pacman_pos[1] - c[1]) for c in capsules)
        obs[idx] = 1.0 - min(min_cap / max_dist, 1.0)
    idx += 1
    
    # [31] Any ghost scared
    obs[idx] = 1.0 if any(g.scaredTimer > 0 for g in ghost_states) else 0.0
    idx += 1
    
    # [32] Time progress
    obs[idx] = step_count / max_steps
    
    return np.clip(np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)


def _get_direction_danger(pacman_pos, ghost_states, dx: int, dy: int) -> float:
    """Calculate danger level in a direction (higher = more dangerous)."""
    max_danger = 0.0
    for ghost in ghost_states:
        if ghost.scaredTimer > 0:
            continue
        gpos = ghost.getPosition()
        rel_x, rel_y = gpos[0] - pacman_pos[0], gpos[1] - pacman_pos[1]
        
        # Check if ghost is in this direction
        in_direction = False
        if dx != 0 and rel_x * dx > 0 and abs(rel_y) <= abs(rel_x):
            in_direction = True
        elif dy != 0 and rel_y * dy > 0 and abs(rel_x) <= abs(rel_y):
            in_direction = True
        
        if in_direction:
            dist = abs(rel_x) + abs(rel_y)
            max_danger = max(max_danger, 1.0 / (dist + 1))
    
    return max_danger


def _get_direction_food(pacman_pos, walls, food, dx: int, dy: int, 
                        width: int, height: int, max_dist: int) -> float:
    """Get food signal in a direction (ray-cast to first food or wall)."""
    for dist in range(1, max_dist):
        x = int(pacman_pos[0] + dx * dist)
        y = int(pacman_pos[1] + dy * dist)
        if not (0 <= x < width and 0 <= y < height) or walls[x][y]:
            break
        if food[x][y]:
            return 1.0 / (dist + 1)
    return 0.0


def _find_nearest_food(pacman_pos, food) -> Optional[Tuple[int, int]]:
    """Find position of nearest food pellet."""
    nearest, min_dist = None, float('inf')
    for x in range(food.width):
        for y in range(food.height):
            if food[x][y]:
                dist = abs(pacman_pos[0] - x) + abs(pacman_pos[1] - y)
                if dist < min_dist:
                    min_dist, nearest = dist, (x, y)
    return nearest


# =============================================================================
# Ghost Observation Extraction  
# =============================================================================

def extract_ghost_observation(game_state, ghost_index: int) -> np.ndarray:
    """
    Extract a 25-dimensional observation vector for a ghost agent.
    
    Features (all normalized to [0, 1]):
        [0-1]   This ghost's position (x, y)
        [2-3]   Scared state (is_scared, timer_normalized)
        [4-5]   Pac-Man position (x, y)
        [6-7]   Relative position to Pac-Man (dx, dy)
        [8]     Manhattan distance to Pac-Man
        [9-12]  Wall sensors (N, S, E, W)
        [13-16] Legal action flags (N, S, E, W)
        [17-20] Other ghost distances (up to 4 values)
        [21-24] Direction to Pac-Man one-hot (N, S, E, W)
    
    Args:
        game_state: The current game state
        ghost_index: Ghost agent index (1-based, as 0 is Pac-Man)
    
    Returns:
        np.ndarray: 25-dimensional float32 observation vector
    
    Raises:
        ValueError: If ghost_index is invalid
    """
    if ghost_index == 0:
        raise ValueError("ghost_index=0 is Pac-Man. Use index >= 1 for ghosts.")
    if ghost_index >= game_state.getNumAgents():
        raise ValueError(f"ghost_index={ghost_index} exceeds agent count ({game_state.getNumAgents()})")
    
    layout = game_state.data.layout
    width, height = layout.width, layout.height
    walls = game_state.getWalls()
    max_dist = width + height
    
    obs = np.zeros(GHOST_OBS_DIM, dtype=np.float32)
    idx = 0
    
    # [0-1] Ghost position
    ghost_pos = game_state.getGhostPosition(ghost_index)
    obs[idx] = ghost_pos[0] / width
    obs[idx + 1] = ghost_pos[1] / height
    idx += 2
    
    # [2-3] Scared state
    ghost_state = game_state.getGhostState(ghost_index)
    obs[idx] = 1.0 if ghost_state.scaredTimer > 0 else 0.0
    obs[idx + 1] = ghost_state.scaredTimer / SCARED_TIME
    idx += 2
    
    # [4-5] Pac-Man position
    pacman_pos = game_state.getPacmanPosition()
    obs[idx] = pacman_pos[0] / width
    obs[idx + 1] = pacman_pos[1] / height
    idx += 2
    
    # [6-7] Relative position to Pac-Man
    obs[idx] = (pacman_pos[0] - ghost_pos[0]) / width
    obs[idx + 1] = (pacman_pos[1] - ghost_pos[1]) / height
    idx += 2
    
    # [8] Distance to Pac-Man
    obs[idx] = manhattanDistance(ghost_pos, pacman_pos) / max_dist
    idx += 1
    
    # [9-12] Wall sensors (N, S, E, W)
    gx, gy = int(ghost_pos[0]), int(ghost_pos[1])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in directions:
        nx, ny = gx + dx, gy + dy
        obs[idx] = 1.0 if (not (0 <= nx < width and 0 <= ny < height) or walls[nx][ny]) else 0.0
        idx += 1
    
    # [13-16] Legal actions (N, S, E, W)
    legal_actions = game_state.getLegalActions(ghost_index)
    action_dirs = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
    for d in action_dirs:
        obs[idx] = 1.0 if d in legal_actions else 0.0
        idx += 1
    
    # [17-20] Other ghost distances
    num_agents = game_state.getNumAgents()
    other_dists = []
    for i in range(1, num_agents):
        if i != ghost_index:
            other_pos = game_state.getGhostPosition(i)
            other_dists.append(manhattanDistance(ghost_pos, other_pos) / max_dist)
    
    # Pad to 4 values
    while len(other_dists) < 4:
        other_dists.append(1.0)
    for i in range(4):
        obs[idx] = other_dists[i]
        idx += 1
    
    # [21-24] Direction to Pac-Man one-hot
    dx = pacman_pos[0] - ghost_pos[0]
    dy = pacman_pos[1] - ghost_pos[1]
    if abs(dy) >= abs(dx):
        obs[idx if dy > 0 else idx + 1] = 1.0  # North or South
    else:
        obs[idx + 2 if dx > 0 else idx + 3] = 1.0  # East or West
    
    return obs.astype(np.float32)


# =============================================================================
# Action Mapping & Policy Utilities
# =============================================================================

ACTION_INDEX_TO_DIR = {
    0: Directions.NORTH, 
    1: Directions.SOUTH,
    2: Directions.EAST, 
    3: Directions.WEST, 
    4: Directions.STOP
}

DIR_TO_ACTION_INDEX = {v: k for k, v in ACTION_INDEX_TO_DIR.items()}


def get_legal_action_from_policy(
    policy, 
    obs: np.ndarray, 
    legal_actions: List[str],
    deterministic: bool = True
) -> str:
    """
    Get a legal action from a policy model.
    
    Uses SB3's predict() method and falls back to random legal action
    if the predicted action is not legal.
    
    Args:
        policy: SB3 model (PPO, MaskablePPO, DQN, etc.)
        obs: Observation array
        legal_actions: List of legal Directions
        deterministic: If True, pick greedily; else sample
    
    Returns:
        str: The selected legal action (a Direction)
    """
    if policy is None or not legal_actions:
        return np.random.choice(legal_actions) if legal_actions else Directions.STOP
    
    # Use SB3's built-in predict method
    action, _ = policy.predict(obs, deterministic=deterministic)
    direction = ACTION_INDEX_TO_DIR.get(int(action), Directions.STOP)
    
    # Return predicted action if legal, otherwise random legal action
    if direction in legal_actions:
        return direction
    return np.random.choice(legal_actions)
