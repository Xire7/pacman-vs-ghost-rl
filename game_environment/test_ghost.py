from ghost_agent import GhostEnv
from stable_baselines3 import DQN
import numpy as np

# No Pac-Man policy = random baseline
env = GhostEnv(ghost_index=1, pacman_policy=None)

ghost_model = DQN(
   "MlpPolicy",
   env,
   learning_rate=1e-3,
   buffer_size=50000,
   learning_starts=1000,
   batch_size=64,
   gamma=0.99,
   target_update_interval=1000,
   verbose=1
)

# Train the agent
print("Starting training...")
ghost_model.learn(total_timesteps=200_000)
print("Training complete!")

# Test for multiple episodes
num_episodes = 20
total_rewards = []
total_steps = []
wins = 0  # Ghost catches Pac-Man

print("\n" + "="*60)
print("🎮 TESTING TRAINED GHOST AGENT")
print("="*60 + "\n")

for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    
    while not done:
        env.render()
        # Get action from trained model
        action, _states = ghost_model.predict(obs, deterministic=True)
        action = int(action)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        steps += 1
    
    # Check if ghost won (caught Pac-Man)
    if env.pacman_env.game_state.isLose():
        wins += 1
        result = "👻 GHOST WON!"
    else:
        result = "🟡 Pac-Man Won"
    
    total_rewards.append(episode_reward)
    total_steps.append(steps)
    
    print(f"Episode {episode + 1:2d}: Reward = {episode_reward:7.2f} | Steps = {steps:3d} | {result}")

env.close()

# Print summary statistics
print("\n" + "="*60)
print("📊 SUMMARY STATISTICS")
print("="*60)
print(f"Average Reward:  {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
print(f"Average Steps:   {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
print(f"Win Rate:        {wins}/{num_episodes} ({100*wins/num_episodes:.1f}%)")
print(f"Best Reward:     {max(total_rewards):.2f}")
print(f"Worst Reward:    {min(total_rewards):.2f}")
print("="*60 + "\n")