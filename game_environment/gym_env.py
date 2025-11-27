"""
Gymnasium wrapper for Berkeley Pac-Man environment.
Compatible with stable-baselines3 for PPO training.
"""

from state_extractor import extract_pacman_observation, extract_ghost_observation, get_best_legal_action
import gymnasium as gym
import numpy as np
import random
from pacman import GameState
from layout import getLayout
from game import Directions, Actions
import ghostAgents
import graphicsDisplay
import graphicsUtils


class PacmanEnv(gym.Env):
    """
    Gymnasium wrapper for Berkeley Pac-Man.

    Now supports trained ghost policies for adversarial training.
    
    Observation Space: 30-dimensional Box[-1, 1]
    Action Space: Discrete(5) - NORTH, SOUTH, EAST, WEST, STOP
    """
    
    metadata = {"render_modes": ["human", "text", "rgb_array"], "render_fps": 10}
    
    ACTION_MAP = {
        0: Directions.NORTH,
        1: Directions.SOUTH,
        2: Directions.EAST,
        3: Directions.WEST,
        4: Directions.STOP
    }
    
    DIRECTION_TO_ACTION = {v: k for k, v in ACTION_MAP.items()}
    
    def __init__(self, layout_name='mediumGrid', ghost_agents=None, max_steps=500, 
                 render_mode=None, reward_shaping=True, ghost_policies=None):
        super().__init__()
        
        self.layout_name = layout_name
        self.layout = getLayout(layout_name)
        if self.layout is None:
            raise ValueError(f"Layout '{layout_name}' not found.")
        
        self._ghost_agent_config = ghost_agents
        self.num_ghosts = len(ghost_agents) if ghost_agents else self.layout.getNumGhosts()
        
        self.ghost_policies = ghost_policies # Dict of {ghost idx: DQN model}

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(30,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)
        
        self.game_state = None
        self.current_score = 0.0
        self.steps = 0
        self.max_steps = max_steps
        self.original_food = 0
        self.prev_food_count = 0
        self.prev_capsule_count = 0
        self.prev_pacman_pos = None
        self.reward_shaping = reward_shaping
        
        self.render_mode = render_mode
        self.display = None
        self._display_initialized = False

    def _create_ghost_agents(self):
        if self._ghost_agent_config is not None:
            return [type(g)(g.index) for g in self._ghost_agent_config]
        else:
            return [ghostAgents.RandomGhost(i + 1) for i in range(self.num_ghosts)]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.game_state = GameState()
        self.game_state.initialize(self.layout, numGhostAgents=self.num_ghosts)
        self.ghost_agents = self._create_ghost_agents()
        
        self.current_score = 0.0
        self.steps = 0
        self.original_food = self.game_state.getNumFood()
        self.prev_food_count = self.original_food
        self.prev_capsule_count = len(self.game_state.getCapsules())
        self.prev_pacman_pos = self.game_state.getPacmanPosition()
        
        if self.render_mode == 'human' and not self._display_initialized:
            self.display = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.05)
            self.display.initialize(self.game_state.data)
            self._display_initialized = True
            graphicsUtils.refresh()
        elif self.render_mode == 'human' and self._display_initialized:
            self.display.initialize(self.game_state.data)
            graphicsUtils.refresh()
        
        obs = extract_pacman_observation(self.game_state, self.original_food).astype(np.float32)
        info = {
            'raw_score': 0,
            'food_remaining': self.original_food,
            'capsules_remaining': self.prev_capsule_count,
        }
        
        return obs, info
    
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item() if action.ndim == 0 else int(action.flat[0])
        else:
            action = int(action)
        
        direction = self.ACTION_MAP[action]
        legal_actions = self.game_state.getLegalActions(0)
        
        if direction not in legal_actions:
            direction = random.choice(legal_actions) if legal_actions else Directions.STOP
        
        prev_score = self.game_state.getScore()
        prev_ghost_dists = self._get_ghost_distances()
        
        self.game_state = self.game_state.generatePacmanSuccessor(direction)
        
        if self.render_mode == 'human' and self.display is not None:
            self.display.update(self.game_state.data)
            graphicsUtils.refresh()
        
        # ghost move
        for ghost_idx in range(1, self.game_state.getNumAgents()):
            if self.game_state.isWin() or self.game_state.isLose():
                break

            legal_ghost = self.game_state.getLegalActions(ghost_idx)

            if self.ghost_policies is not None and ghost_idx in self.ghost_policies:
                ghost_obs = extract_ghost_observation(self.game_state, ghost_idx)
                
                ghost_direction = get_best_legal_action(
                    policy=self.ghost_policies[ghost_idx],
                    obs=ghost_obs,
                    legal_actions=legal_ghost
                )

            elif self._ghost_agent_config and ghost_idx - 1 < len(self.ghost_agents):
                # use scripted behavior (original behavior)
                ghost_direction = self.ghost_agents[ghost_idx - 1].getAction(self.game_state)

                if ghost_direction not in legal_ghost:
                    ghost_direction = random.choice(legal_ghost) if legal_ghost else Directions.STOP
            else:
                #random ghost (fallback)
                ghost_direction = random.choice(legal_ghost) if legal_ghost else Directions.STOP
            
            self.game_state = self.game_state.generateSuccessor(ghost_idx, ghost_direction)
            
            if self.render_mode == 'human' and self.display is not None:
                self.display.update(self.game_state.data)
                graphicsUtils.refresh()
        
        new_score = self.game_state.getScore()
        reward = new_score - prev_score
        
        if self.reward_shaping:
            reward = self._shape_reward(reward, prev_ghost_dists)
        
        self.current_score = new_score
        self.steps += 1
        
        terminated = self.game_state.isWin() or self.game_state.isLose()
        truncated = self.max_steps is not None and self.steps >= self.max_steps and not terminated
        
        self.prev_food_count = self.game_state.getNumFood()
        self.prev_capsule_count = len(self.game_state.getCapsules())
        self.prev_pacman_pos = self.game_state.getPacmanPosition()
        
        obs = extract_pacman_observation(self.game_state, self.original_food).astype(np.float32)
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
        pacman_pos = self.game_state.getPacmanPosition()
        distances = []
        for ghost_state in self.game_state.getGhostStates():
            ghost_pos = ghost_state.getPosition()
            dist = abs(pacman_pos[0] - ghost_pos[0]) + abs(pacman_pos[1] - ghost_pos[1])
            distances.append((dist, ghost_state.scaredTimer > 0))
        return distances
    
    def _shape_reward(self, base_reward, prev_ghost_dists):
        shaped_reward = 0.0
        
        ate_food = base_reward >= 9
        ate_ghost = base_reward >= 199
        won = self.game_state.isWin()
        died = self.game_state.isLose()
        
        shaped_reward -= 0.01
        
        if ate_food and not won:
            shaped_reward += 0.5
        
        if ate_ghost:
            shaped_reward += 2.0
        
        if won:
            shaped_reward += 10.0
        
        if died:
            shaped_reward -= 5.0
        
        current_ghost_dists = self._get_ghost_distances()
        for i, (curr_dist, is_scared) in enumerate(current_ghost_dists):
            if i < len(prev_ghost_dists):
                prev_dist, was_scared = prev_ghost_dists[i]
                if is_scared and was_scared:
                    dist_delta = prev_dist - curr_dist
                    if dist_delta > 0:
                        shaped_reward += 0.1 * dist_delta
                elif not is_scared and not was_scared:
                    if curr_dist < 2:
                        shaped_reward -= 0.05
        
        return shaped_reward
    
    def render(self):
        if self.render_mode == 'human':
            if self.display is not None:
                self.display.update(self.game_state.data)
        elif self.render_mode == 'text':
            print(str(self.game_state))
        return None
    
    def close(self):
        if self.display is not None:
            try:
                self.display.finish()
            except:
                pass
            self._display_initialized = False
            self.display = None
    
    def get_legal_action_mask(self):
        legal_actions = self.game_state.getLegalActions(0)
        mask = np.zeros(5, dtype=np.bool_)
        for action in legal_actions:
            if action in self.DIRECTION_TO_ACTION:
                mask[self.DIRECTION_TO_ACTION[action]] = True
        return mask


def make_pacman_env(layout_name='smallGrid', ghost_type='random', num_ghosts=None,
                    max_steps=500, render_mode=None, reward_shaping=True):
    """Factory function to create a PacmanEnv with specified configuration."""
    layout = getLayout(layout_name)
    if layout is None:
        raise ValueError(f"Layout '{layout_name}' not found")
    
    if num_ghosts is None:
        num_ghosts = layout.getNumGhosts()
    else:
        num_ghosts = min(num_ghosts, layout.getNumGhosts())
    
    if ghost_type == 'random':
        ghosts = [ghostAgents.RandomGhost(i + 1) for i in range(num_ghosts)]
    elif ghost_type == 'directional':
        ghosts = [ghostAgents.DirectionalGhost(i + 1) for i in range(num_ghosts)]
    else:
        ghosts = None
    
    return PacmanEnv(
        layout_name=layout_name,
        ghost_agents=ghosts,
        max_steps=max_steps,
        render_mode=render_mode,
        reward_shaping=reward_shaping
    )
