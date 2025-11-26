import sys
import numpy as np

import ghostAgents
import time
from gym_env import PacmanEnv

def test_basic_environment():
    """Test the gym environment with random actions"""
    print("=" * 60)
    print("Test 1: basic environment with random agents")
    print("=" * 60)


    env = PacmanEnv(
        layout_name='smallGrid',
        ghost_agents=None,
        max_steps=100,
        render_mode='human'
    )

    # begin running 1 episode

    obs, info = env.reset()
    print(f"Initial observation: {obs}\n")
    print(f"Initial observation shape: {obs.shape}\n")

    total_reward = 0
    step_count = 0

    print("Running episode with random actions")

    while True:
        action = env.action_space.sample()
        print(f"Taking action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        print(f"Step {step_count}: action={action}, reward={reward:.2f}, Score={info['raw_score']:.2f}\n")

        if terminated or truncated:
            print(f"\nEpisode finished after {step_count} steps.")
            print(f"    Total reward: {total_reward:.2f}")
            print(f"    Final score: {info['raw_score']:.2f}")
            print(f"    Win: {info['win']}")
            print(f"    Lose: {info['lose']}")
            print(f"    Truncated (timeout): {truncated}")
            print(f"{'=' * 60}\n")
            break
        
    env.close()




if __name__ == "__main__":
    test_basic_environment()