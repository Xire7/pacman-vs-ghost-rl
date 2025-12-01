from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

def record_gameplay_video(
    pacman_model_path,
    ghost_model_dir,
    ghost_version,
    layout_name='mediumClassic',
    num_ghosts=2,
    output_path='gameplay.mp4',
    max_steps=500,
    fps=10,
    format='mp4'
):
    """
    Record gameplay by capturing the Tkinter rendering window.
    """
    print(f"\n{'='*60}")
    print("Loading models...")
    print(f"{'='*60}")

    # ✅ Load Pac-Man model
    pacman_model = MaskablePPO.load(pacman_model_path)
    
    # ✅ Load VecNormalize stats (CRITICAL!)
    vecnorm_path = None
    
    # Try same directory as model
    model_dir = os.path.dirname(pacman_model_path)
    candidate1 = os.path.join(model_dir, 'vecnormalize.pkl')
    
    # Try parent directory (for best/best_model.zip structure)
    candidate2 = os.path.join(os.path.dirname(model_dir), 'vecnormalize.pkl')
    
    # Try two levels up
    candidate3 = os.path.join(os.path.dirname(os.path.dirname(model_dir)), 'vecnormalize.pkl')
    
    for candidate in [candidate1, candidate2, candidate3]:
        if os.path.exists(candidate):
            vecnorm_path = candidate
            print(f"Found VecNormalize stats: {vecnorm_path}")
            break
    
    if not vecnorm_path:
        print("⚠️  WARNING: VecNormalize stats not found!")
        print("   Model will perform poorly without normalization.")
        print(f"   Searched: {candidate1}, {candidate2}, {candidate3}")
    
    # ✅ Load ghost models
    ghost_models = {}
    for i in range(1, num_ghosts + 1):
        ghost_path = os.path.join(ghost_model_dir, f"ghost_{i}_v{ghost_version}")
        if os.path.exists(ghost_path + ".zip"):
            ghost_models[i] = DQN.load(ghost_path)
            print(f"Loaded Ghost {i}: {ghost_path}")
        else:
            print(f"Warning: Ghost {i} model not found: {ghost_path}")

    # ✅ Create environment WITH VecNormalize wrapper
    base_env = PacmanEnv(
        layout_name=layout_name,
        ghost_policies=ghost_models,
        max_steps=max_steps,
        render_mode='human',
    )
    
    # ✅ Wrap in DummyVecEnv
    env = DummyVecEnv([lambda: base_env])
    
    # ✅ Load VecNormalize wrapper if available
    if vecnorm_path:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False  # Don't update stats during visualization
        env.norm_reward = False  # Don't normalize rewards
        print("✅ VecNormalize wrapper loaded and applied")
    else:
        print("⚠️  Proceeding WITHOUT VecNormalize (expect poor performance)")

    print("Starting game (will capture window)...")
    print("Do not minimize or cover the game window during recording!\n")
    
    # ✅ Reset returns vectorized obs
    obs = env.reset()

    # Wait for the window to stabilize
    time.sleep(2.0)

    frames = []
    step_count = 0
    done = False
    final_info = {}

    print(f"Recording... (max {max_steps} steps)")
    while step_count < max_steps:
        # Capture frame
        try:
            bbox = None
            frame = ImageGrab.grab(bbox=bbox)
            frames.append(np.array(frame))
        except Exception as e:
            print(f"Frame capture error at step {step_count}: {e}")
            break

        if done:
            if step_count - len(frames) > 10:
                break
        else:
            # ✅ Get action masks from base environment
            action_masks = base_env.action_masks()
            
            # ✅ Predict with normalized observations
            action, _ = pacman_model.predict(
                obs, 
                deterministic=True, 
                action_masks=action_masks
            )
            
            # ✅ Convert action
            if hasattr(action, 'item'):
                action = int(action.item())
            elif isinstance(action, np.ndarray):
                action = int(action[0])  # VecEnv returns array
            
            # ✅ Step through VecEnv
            obs, reward, done_vec, info_vec = env.step([action])
            
            # ✅ Extract values from vectorized format
            done = done_vec[0]
            if done and len(info_vec) > 0:
                final_info = info_vec[0]

        step_count += 1
        time.sleep(1.0 / fps)

    env.close()

    print(f"Captured {len(frames)} frames.")
    if len(frames) == 0:
        print("No frames captured! Exiting.")
        return None

    print(f"Saving video to: {output_path}")
    try:
        if format == 'gif' or output_path.endswith('.gif'):
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
        else:
            imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
        print(f"Saved {format.upper()} to: {output_path}")
        print(f"Duration: {len(frames) / fps:.1f} seconds")
        if done and final_info:
            if final_info.get('win', False):
                print(f"PAC-MAN WINS! Score: {final_info.get('raw_score', 0):.0f}")
            elif final_info.get('lose', False):
                print(f"GHOSTS WIN! Score: {final_info.get('raw_score', 0):.0f}")
        else:
            print(f"Stopping recording at {step_count} steps.")
        return output_path
    except Exception as e:
        print(f"Error saving video: {e}")
        return None