from gym_env import PacmanEnv
from stable_baselines3 import PPO
import numpy as np

# Make sure this matches your training run output directory
pacman_v1 = PPO.load("training_output/run_20251127_194415/models/pacman_v1")

# ✅ FIX: Use the SAME layout you trained on!
env = PacmanEnv(layout_name='smallGrid', render_mode=None)  

wins = 0
scores = []
lengths = []

print("Testing Pac-Man v1 vs Random Ghosts on smallGrid (50 episodes)...\n")

for i in range(50):
    obs, _ = env.reset()
    done = False
    steps = 0
    
    while not done:
        action, _ = pacman_v1.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    
    if env.game_state.isWin():
        wins += 1
    
    scores.append(env.game_state.getScore())
    lengths.append(steps)
    
    if (i + 1) % 10 == 0:
        print(f"Episode {i+1}/50: Wins so far: {wins}")

env.close()

print(f"\n{'='*60}")
print(f"PAC-MAN v1 PERFORMANCE (smallGrid)")
print(f"{'='*60}")
print(f"Win Rate:     {wins}/50 ({100*wins/50:.1f}%)")
print(f"Avg Score:    {np.mean(scores):.1f}")
print(f"Avg Length:   {np.mean(lengths):.1f} steps")
print(f"{'='*60}\n")

if wins < 15:
    print("❌ Still not converged")
elif wins < 35:
    print("⚠️  Partially converged")
else:
    print("✅ Well converged!")