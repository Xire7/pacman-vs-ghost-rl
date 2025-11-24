import numpy as np
from util import manhattanDistance

def extract_pacman_state(game_state):
    """Extract Pac-Man-centric state features from GameState.
    
    This produces a feature vector from Pac-Man's perspective, including:
    - Pac-Man's own position
    - All ghost positions and scared states
    - Food/capsule information
    - Current score
    
    Feature vector layout:
        [0-1]   Pac-Man position (x, y) normalized
        [2-17]  Up to 4 ghosts: (x, y, scared_flag, scared_timer) × 4
        [18]    Nearest food distance normalized
        [19]    Remaining food count normalized
        [20]    Remaining capsules normalized
        [21]    Current score normalized
    
    Returns:
        np.ndarray: Feature vector with all values in [0, 1]
    """
    layout = game_state.data.layout
    width = layout.width
    height = layout.height
    
    state = []
    
    # 1. Pac-Man position
    pacman_pos = game_state.getPacmanPosition()
    state.extend([pacman_pos[0] / width, pacman_pos[1] / height])
    
    # 2. Ghost information (up to 4 ghosts, 4 features each)
    ghost_states = game_state.getGhostStates()
    for ghost in ghost_states[:4]:  # Max 4 ghosts
        pos = ghost.getPosition()
        state.extend([
            pos[0] / width,
            pos[1] / height,
            1.0 if ghost.scaredTimer > 0 else 0.0,
            ghost.scaredTimer / 40.0
        ])
    
    # Pad with zeros if fewer than 4 ghosts
    for _ in range(4 - len(ghost_states)):
        state.extend([0.0, 0.0, 0.0, 0.0])
    
    # 3. Nearest food distance
    food_positions = game_state.getFood().asList()
    if food_positions:
        distances = [manhattanDistance(pacman_pos, food) for food in food_positions]
        nearest_food = min(distances)
        state.append(nearest_food / (width + height))
    else:
        state.append(0.0)  # No food remaining
    
    # 4. Total food pellets remaining
    state.append(game_state.getNumFood() / 100.0)
    
    # 5. Power capsules remaining (max 4)
    capsules = game_state.getCapsules()
    state.append(len(capsules) / 4.0)
    
    # 6. Current score (normalized)
    state.append(game_state.getScore() / 1000.0)
    
    return np.array(state, dtype=np.float32)


def extract_ghost_state(game_state, ghost_index):
    """Extract ghost-centric state features for training ghost agents.
    
    Feature vector layout:
        [0-1]   This ghost's position (x, y) normalized
        [2-3]   This ghost's scared state (flag, timer_normalized)
        [4-5]   Pac-Man position (x, y) normalized
        [6-7]   Relative position to Pac-Man (dx, dy) normalized
        [8]     Manhattan distance to Pac-Man normalized
    
    Returns:
        np.ndarray: Feature vector with all values in [-1, 1], dtype float32
    """
    # Validate ghost index
    if ghost_index == 0:
        raise ValueError("ghost_index=0 is Pac-Man, not a ghost. Use index >= 1.")
    if ghost_index >= game_state.getNumAgents():
        raise ValueError(f"ghost_index={ghost_index} exceeds number of agents "
                        f"({game_state.getNumAgents()}).")
    
    layout = game_state.data.layout
    width = layout.width
    height = layout.height
    
    state = []
    
    # 1. This ghost's position
    ghost_pos = game_state.getGhostPosition(ghost_index)
    state.extend([ghost_pos[0] / width, ghost_pos[1] / height])
    
    # 2. This ghost's scared state
    ghost_state = game_state.getGhostState(ghost_index)
    state.extend([
        1.0 if ghost_state.scaredTimer > 0 else 0.0,
        ghost_state.scaredTimer / 40.0
    ])
    
    # 3. Pac-Man position
    pacman_pos = game_state.getPacmanPosition()
    state.extend([pacman_pos[0] / width, pacman_pos[1] / height])
    
    # 4. Relative position to Pac-Man
    rel_x = (pacman_pos[0] - ghost_pos[0]) / width
    rel_y = (pacman_pos[1] - ghost_pos[1]) / height
    state.extend([rel_x, rel_y])
    
    # 5. Manhattan distance to Pac-Man
    dist = manhattanDistance(ghost_pos, pacman_pos)
    state.append(dist / (width + height))
    
    return np.array(state, dtype=np.float32)