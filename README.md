# Pac-Man vs Ghost RL

Adversarial reinforcement learning project training PPO-based Pac-Man agents against DQN-based Ghost agents using the Berkeley Pac-Man game engine.

## Project Structure

```
game-environment/
├── gym_env.py           # Gymnasium wrapper for Pac-Man environment
├── state_extractor.py   # Feature extraction utilities
├── pacman.py            # Berkeley Pac-Man game engine (base code)
├── game.py              # Core game classes and logic
├── layout.py            # Layout loading and parsing
├── ghostAgents.py       # Ghost agent implementations
└── layouts/             # Game board layout files (.lay)
```

## Core Components

### 1. `gym_env.py` - Gymnasium Environment Wrapper

A standard Gymnasium-compatible environment for training Pac-Man agents with RL algorithms.

#### Key Features:
- **Observation Space**: 30-dimensional continuous space (Box) with values in [0, 1]
- **Action Space**: 5 discrete actions (North, South, East, West, Stop)
- **Multi-ghost support**: Handles variable number of ghosts (defaults to layout definition)
- **Episode truncation**: Optional `max_steps` parameter for timeout
- **Separated termination flags**: `terminated` (win/lose) vs `truncated` (timeout)

#### Observation Vector Layout (30 dimensions):
```
[0-1]   Pac-Man position (x, y) normalized
[2-17]  Up to 4 ghosts: (x, y, scared_flag, scared_timer) × 4 ghosts
[18]    Remaining food count normalized
[19]    Remaining capsules normalized  
[20]    Nearest food distance normalized
[21]    Current score normalized
[22]    Original food count (for progress tracking)
[23]    Time fraction (steps/max_steps)
[24-29] Padding/reserved
```

#### Usage Example:
```python
from game_environment.gym_env import PacmanEnv

# Create environment
env = PacmanEnv(
    layout_name='mediumClassic',
    ghost_agents=None,        # None = random ghosts, or provide list of agents
    max_steps=1000            # Optional timeout
)

# Standard Gymnasium loop
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Or use your RL agent
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode finished: Win={info['win']}, Score={info['raw_score']}")
        break

env.close()
```

#### Custom Ghost Agents:
```python
from ghostAgents import RandomGhost, DirectionalGhost

# Use specific ghost behaviors
ghosts = [RandomGhost(1), DirectionalGhost(2)]
env = PacmanEnv('mediumClassic', ghost_agents=ghosts)
```

---

### 2. `state_extractor.py` - Feature Extraction Utilities

Standalone functions for extracting state features from Berkeley `GameState` objects. Useful for:
- Training ghost agents (each ghost needs its own perspective)
- Custom training loops outside Gymnasium wrapper
- Analysis and debugging

#### Functions:

##### `extract_pacman_state(game_state)` → np.ndarray
Extracts Pac-Man-centric features (22+ dimensions, depends on ghost count).

**Returns:**
```
[0-1]   Pac-Man position (x, y) normalized to [0, 1]
[2-17]  Ghost info: (x, y, scared_flag, scared_timer) × 4 (padded)
[18]    Nearest food distance normalized
[19]    Food pellets remaining normalized
[20]    Power capsules remaining normalized
[21]    Current score normalized
```

**Usage:**
```python
from game_environment.state_extractor import extract_pacman_state
from pacman import GameState
from layout import getLayout

# Create game state
game_state = GameState()
layout = getLayout('mediumClassic')
game_state.initialize(layout, numGhostAgents=2)

# Extract features
features = extract_pacman_state(game_state)
print(f"Feature shape: {features.shape}")  # (22,)
print(f"Pac-Man position: {features[0:2]}")
```

##### `extract_ghost_state(game_state, ghost_index)` → np.ndarray
Extracts ghost-centric features for a specific ghost (9 dimensions).

**Parameters:**
- `game_state`: Berkeley GameState object
- `ghost_index`: Integer ≥ 1 (1=first ghost, 2=second ghost, etc.)
  - Note: Index 0 is Pac-Man, so ghosts start at 1

**Returns:**
```
[0-1]   This ghost's position (x, y) normalized to [0, 1]
[2-3]   Scared state (binary flag, timer) normalized
[4-5]   Pac-Man position (x, y) normalized
[6-7]   Relative position to Pac-Man (dx, dy) in [-1, 1]
[8]     Manhattan distance to Pac-Man normalized
```

**Usage:**
```python
from game_environment.state_extractor import extract_ghost_state

# For training individual ghost agents
for ghost_idx in range(1, game_state.getNumAgents()):
    ghost_features = extract_ghost_state(game_state, ghost_idx)
    # Feed to ghost's DQN agent
    action = ghost_dqn_models[ghost_idx - 1].predict(ghost_features)
```

**Error Handling:**
```python
# Raises ValueError if invalid ghost_index
try:
    extract_ghost_state(game_state, 0)  # Error: 0 is Pac-Man
except ValueError as e:
    print(e)  # "ghost_index=0 is Pac-Man, not a ghost"

try:
    extract_ghost_state(game_state, 10)  # Error: out of range
except ValueError as e:
    print(e)  # "ghost_index=10 exceeds number of agents"
```

---

## Available Layouts

Layouts are stored in `game-environment/layouts/` directory:

| Layout | Size | Ghosts | Difficulty |
|--------|------|--------|------------|
| `smallGrid` | 7×7 | 1 | Easy |
| `mediumGrid` | 10×16 | 2 | Medium |
| `mediumClassic` | 20×11 | 2 | Medium |
| `originalClassic` | 20×11 | 4 | Hard |
| `capsuleClassic` | 20×11 | 4 | Hard |

---

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Xire7/pacman-vs-ghost-rl.git
cd pacman-vs-ghost-rl

# Install dependencies (TODO: create requirements.txt)
pip install gymnasium numpy stable-baselines3 torch
```

### Test Environment
```bash
cd game-environment
python gym_env.py  # Runs smoke test with random actions
```

### Test Feature Extraction
```bash
cd game-environment
python state_extractor.py  # Validates both extraction functions
```

---

## Technical Details

### Normalization
All features are normalized to facilitate neural network training:
- **Positions**: Divided by layout width/height → [0, 1]
- **Distances**: Divided by max possible distance (width + height) → [0, 1]
- **Counts**: Divided by reasonable upper bounds (100 for food, 4 for capsules)
- **Timers**: Divided by 40 (standard scared timer duration) → [0, 1]

### Action Validation
The gym environment validates actions before applying them:
- If an illegal action is selected, a random legal action is substituted
- Prevents crashes from wall collisions or invalid moves

### Episode Termination
- **Terminated (Natural)**: Pac-Man wins (all food eaten) or loses (caught by ghost)
- **Truncated (Timeout)**: Max steps reached without natural termination
- Important for advantage estimation in PPO algorithm

---

## Implementation Notes

- **Berkeley Base Code**: `pacman.py`, `game.py`, `layout.py` are unmodified base code from UC Berkeley CS188
- **Gymnasium API**: Uses modern v0.26+ API with separated `terminated`/`truncated` flags
- **Ghost Indexing**: Pac-Man is always index 0, ghosts start at index 1
- **Fixed Observation Size**: 30 dimensions for gym_env (with padding), variable for state_extractor
- **Scared Timer**: 40 ticks after eating power capsule (Berkeley default)

---

## License

Based on UC Berkeley CS188 Pacman Projects.