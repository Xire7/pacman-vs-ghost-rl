import numpy as np
from util import manhattanDistance

def extract_pacman_state(game_state):
    """
    Extract Pac-Man's state representation
    Returns: np.array of shape (state_dim,)
    """
    layout = game_state.data.layout
    width = layout.width
    height = layout.height
    
    state = []
    
    # 1. Pac-Man position
    pacman_pos = game_state.getPacmanPosition()
    state.extend([pacman_pos[0] / width, pacman_pos[1] / height])
    
    # 2. Ghost information
    ghost_states = game_state.getGhostStates()
    for ghost in ghost_states[:4]:  # Max 4 ghosts
        pos = ghost.getPosition()
        state.extend([
            pos[0] / width,
            pos[1] / height,
            1.0 if ghost.scaredTimer > 0 else 0.0,
            ghost.scaredTimer / 40.0
        ])
    
    # Pad to 4 ghosts
    for _ in range(4 - len(ghost_states)):
        state.extend([0.0, 0.0, 0.0, 0.0])
    
    # 3. Nearest food distance
    food_positions = game_state.getFood().asList()
    if food_positions:
        distances = [manhattanDistance(pacman_pos, food) for food in food_positions]
        nearest_food = min(distances)
        state.append(nearest_food / (width + height))
    else:
        state.append(0.0)
    
    # 4. Pellets remaining
    state.append(game_state.getNumFood() / 100.0)
    
    # 5. Power pellets
    capsules = game_state.getCapsules()
    state.append(len(capsules) / 4.0)
    
    # 6. Current score (normalized)
    state.append(game_state.getScore() / 1000.0)
    
    return np.array(state, dtype=np.float32)


def extract_ghost_state(game_state, ghost_index):
    """
    Extract Ghost's state representation
    Returns: np.array of shape (state_dim,)
    """
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