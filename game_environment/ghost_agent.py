import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import Directions
from state_extractor import extract_pacman_state, extract_ghost_state
from gym_env import PacmanEnv
from stable_baselines3 import DQN

class GhostEnv(gym.Env):
    """
    Gym wrapper that lets ONE ghost be controlled by RL.
    Now includes rendering support.
    """

    metadata = {"render_modes": ["human", "text", "rgb_array"]}

    def __init__(self, ghost_index=1, pacman_policy=None, render_mode=None):
        super().__init__()

        self.ghost_index = ghost_index
        self.pacman_policy = pacman_policy
        self.render_mode = render_mode

        # Pac-Man environment handles game state + display
        # IMPORTANT: Pass render_mode to PacmanEnv
        self.pacman_env = PacmanEnv(render_mode=render_mode)

        self.num_agents = None
        self.prev_dist = None

        # ---- OBSERVATION SPACE ----
        dummy_state, _ = self.pacman_env.reset()
        dummy_obs = extract_ghost_state(self.pacman_env.game_state, ghost_index)
        obs_shape = np.array(dummy_obs).shape

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # ---- ACTION SPACE ----
        self.action_space = spaces.Discrete(4)  # N,S,E,W

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        obs, info = self.pacman_env.reset()
        state = self.pacman_env.game_state
        self.num_agents = state.getNumAgents()

        # track distance to Pac-Man
        pac = state.getPacmanPosition()
        gh  = state.getGhostPosition(self.ghost_index)
        self.prev_dist = abs(pac[0] - gh[0]) + abs(pac[1] - gh[1])

        ghost_obs = extract_ghost_state(state, self.ghost_index).astype(np.float32)

        if self.render_mode == "human":
            self.render()

        return ghost_obs, {}

    def step(self, action):
        # Handle different action formats
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = int(action.item())
            else:
                action = int(action[0])
        else:
            action = int(action)

        state = self.pacman_env.game_state

        action_map = {
            0: Directions.NORTH,
            1: Directions.SOUTH,
            2: Directions.EAST,
            3: Directions.WEST
        }

        for agent_index in range(self.num_agents):
            if state.isWin() or state.isLose():
                break

            legal = state.getLegalActions(agent_index)

            if agent_index == 0:  
                # PACMAN
                if self.pacman_policy is None:
                    chosen = np.random.choice(legal)
                else:
                    # Use PacmanEnv's 30-dim observation
                    temp_state = self.pacman_env.game_state
                    self.pacman_env.game_state = state
                    pac_obs = self.pacman_env._extract_observation()
                    self.pacman_env.game_state = temp_state
                    
                    a, _ = self.pacman_policy.predict(pac_obs, deterministic=False)
                    pac_map = {
                        0: Directions.NORTH,
                        1: Directions.SOUTH,
                        2: Directions.EAST,
                        3: Directions.WEST,
                        4: Directions.STOP
                    }
                    chosen = pac_map[int(a)]
                    if chosen not in legal:
                        chosen = np.random.choice(legal)

            elif agent_index == self.ghost_index:
                # RL Ghost
                chosen = action_map[action]
                if chosen not in legal:
                    chosen = np.random.choice(legal)

            else:
                # Other ghosts
                chosen = np.random.choice(legal)

            state = state.generateSuccessor(agent_index, chosen)

        # Update game state ONCE after all agents move
        self.pacman_env.game_state = state

        # ------- Reward shaping -------
        pac = state.getPacmanPosition()
        gh  = state.getGhostPosition(self.ghost_index)
        dist = abs(pac[0] - gh[0]) + abs(pac[1] - gh[1])

        reward = 0.0
        if state.isLose():
            reward += 100.0  # caught pacman

        if self.prev_dist is not None:
            reward += (self.prev_dist - dist) * 0.5

        reward -= 0.1  # tiny time penalty

        self.prev_dist = dist

        terminated = state.isWin() or state.isLose()
        truncated = False

        next_obs = extract_ghost_state(state, self.ghost_index).astype(np.float32)

        # Render ONLY at the end of the full turn
        if self.render_mode == "human":
            self.render()

        return next_obs, reward, terminated, truncated, {}

    def render(self):
        """Render the current state via PacmanEnv's display."""
        if self.render_mode == 'human':
            # Use PacmanEnv's render method which handles the display
            if self.pacman_env.display is not None:
                try:
                    # Make sure game_state.data is properly set
                    import graphicsUtils
                    self.pacman_env.display.update(self.pacman_env.game_state.data)
                    graphicsUtils.refresh()
                except Exception as e:
                    # Fallback to text if display fails
                    print(f"Display error: {e}")
                    print(str(self.pacman_env.game_state))
        elif self.render_mode == 'text':
            print(str(self.pacman_env.game_state))
    
    def close(self):
        self.pacman_env.close()