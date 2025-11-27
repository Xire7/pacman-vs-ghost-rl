import numpy as np
from game import Actions, Directions
from util import manhattanDistance

def extract_pacman_observation(game_state, original_food_count):
        # 30 dimension feature vector, all values in [-1, 1]
        layout = game_state.data.layout
        width = layout.width
        height = layout.height
        max_dist = width + height

        features = []

        # Pac-Man position (2)
        pacman_pos = game_state.getPacmanPosition()
        features.append(2.0 * pacman_pos[0] / (width - 1) - 1.0)
        features.append(2.0 * pacman_pos[1] / (height - 1) - 1.0)

        # Pac-Man direction (2)
        pacman_state = game_state.data.agentStates[0]
        direction = pacman_state.getDirection()
        dx, dy = Actions.directionToVector(direction)
        features.append(float(dx))
        features.append(float(dy))

        # Ghost info (16 = 4 ghosts × 4 features)
        ghost_states = game_state.getGhostStates()
        for i in range(4):
            if i < len(ghost_states):
                ghost = ghost_states[i]
                ghost_pos = ghost.getPosition()
                features.append(2.0 * ghost_pos[0] / (width - 1) - 1.0)
                features.append(2.0 * ghost_pos[1] / (height - 1) - 1.0)
                features.append(-1.0 if ghost.scaredTimer > 0 else 1.0)
                features.append(ghost.scaredTimer / 40.0)
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        # Food info (3)
        food_positions = game_state.getFood().asList()
        num_food = len(food_positions)

        if original_food_count > 0:
            features.append(2.0 * num_food / original_food_count - 1.0)
        else:
            features.append(-1.0)

        capsules = game_state.getCapsules()
        features.append(len(capsules) / 4.0 * 2.0 - 1.0)

        if food_positions:
            distances = [abs(pacman_pos[0] - fx) + abs(pacman_pos[1] - fy) 
                        for fx, fy in food_positions]
            nearest_food_dist = min(distances)
            features.append(1.0 - 2.0 * nearest_food_dist / max_dist)
        else:
            features.append(1.0)

        # Ghost distances (2)
        dangerous_ghost_dists = []
        scared_ghost_dists = []
        for ghost in ghost_states:
            ghost_pos = ghost.getPosition()
            dist = abs(pacman_pos[0] - ghost_pos[0]) + abs(pacman_pos[1] - ghost_pos[1])
            if ghost.scaredTimer > 0:
                scared_ghost_dists.append(dist)
            else:
                dangerous_ghost_dists.append(dist)

        if dangerous_ghost_dists:
            nearest_danger = min(dangerous_ghost_dists)
            features.append(2.0 * nearest_danger / max_dist - 1.0)
        else:
            features.append(1.0)

        if scared_ghost_dists:
            nearest_scared = min(scared_ghost_dists)
            features.append(1.0 - 2.0 * nearest_scared / max_dist)
        else:
            features.append(-1.0)

        # Score (1)
        score = game_state.getScore()
        normalized_score = np.clip(score / 500.0, -1.0, 1.0)
        features.append(normalized_score)

        # Direction to nearest food (4)
        if food_positions:
            distances_with_pos = [(abs(pacman_pos[0] - fx) + abs(pacman_pos[1] - fy), (fx, fy))
                                    for fx, fy in food_positions]
            _, nearest_food_pos = min(distances_with_pos, key=lambda x: x[0])
            
            dx = nearest_food_pos[0] - pacman_pos[0]
            dy = nearest_food_pos[1] - pacman_pos[1]
            
            mag = abs(dx) + abs(dy)
            if mag > 0:
                features.append(dx / mag)
                features.append(dy / mag)
            else:
                features.append(0.0)
                features.append(0.0)
            
            features.append(1.0)
            features.append(1.0)
        else:
            features.extend([0.0, 0.0, -1.0, -1.0])

        while len(features) < 30:
            features.append(0.0)

        obs = np.array(features[:30], dtype=np.float32)
        obs = np.clip(obs, -1.0, 1.0)

        return obs


def extract_ghost_observation(game_state, ghost_index):
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
    state.extend([ghost_pos[0] / width, ghost_pos[1] / height]) # normalized
    
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



def get_best_legal_action(policy, obs, legal_actions):
    """
    Get the best legal action from a policy using action masking.
    
    Args:
        policy: SB3 model (PPO, DQN, etc.)
        obs: Observation array
        legal_actions: List of legal Directions (e.g., [North, South, East])
    
    Returns:
        Direction: Best legal action
    """

    action_map = {
        0: Directions.NORTH, 1: Directions.SOUTH,
        2: Directions.EAST, 3: Directions.WEST, 4: Directions.STOP
    }
    # Get action probabilities or Q-values from the policy
    if hasattr(policy.policy, 'get_distribution'):
        # For PPO (stochastic policies)
        with policy.policy.set_training_mode(False):
            obs_tensor = policy.policy.obs_to_tensor(obs)[0]
            distribution = policy.policy.get_distribution(obs_tensor)
            action_probs = distribution.distribution.probs.detach().cpu().numpy()[0]
    else:
        # For DQN (Q-values)
        obs_tensor = policy.policy.obs_to_tensor(obs)[0]
        q_values = policy.q_net(obs_tensor).detach().cpu().numpy()[0]
        action_probs = q_values  # Higher Q-value = better action
    
    # Create mask for legal actions
    legal_indices = [idx for idx, direction in action_map.items() 
                     if direction in legal_actions]
    
    if not legal_indices:
        # Fallback if no legal actions (shouldn't happen)
        return np.random.choice(legal_actions)
    
    # Mask illegal actions (set to -inf so they're never chosen)
    masked_probs = np.full_like(action_probs, -np.inf)
    masked_probs[legal_indices] = action_probs[legal_indices]
    
    # Choose action with highest probability/Q-value among legal actions
    best_action_idx = np.argmax(masked_probs)
    
    return action_map[best_action_idx]
