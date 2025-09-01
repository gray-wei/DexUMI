#!/usr/bin/env python3
"""
Debug script to analyze inference issues
"""

import numpy as np
import pickle
import zarr

def analyze_action_magnitude():
    """Analyze action magnitude in training data vs expected inference"""
    
    # Load training stats
    stats_path = 'data/weight/vision_tactile_propio/stats.pickle'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Check hand action statistics
    print("=== Hand Action Statistics ===")
    hand_min = stats['hand_action']['min']
    hand_max = stats['hand_action']['max']
    hand_range = hand_max - hand_min
    
    print(f"Hand action min: {hand_min}")
    print(f"Hand action max: {hand_max}")
    print(f"Hand action range: {hand_range}")
    
    # Load actual episode data to verify
    data_path = 'data/xhand_dataset_aligned.zarr'
    z = zarr.open(data_path, 'r')
    
    # Sample first episode
    first_ep = z['episode_0']
    hand_actions = first_ep['hand_action'][:]
    
    print("\n=== Actual Hand Actions (Episode 0) ===")
    print(f"Shape: {hand_actions.shape}")
    print(f"First frame: {hand_actions[0]}")
    print(f"Last frame: {hand_actions[-1]}")
    print(f"Max change between frames: {np.abs(np.diff(hand_actions, axis=0)).max()}")
    
    # Check if actions are absolute positions or velocities
    print("\n=== Action Type Analysis ===")
    print("Checking if hand_actions represent absolute positions...")
    
    # If absolute positions, values should be relatively stable with small changes
    changes = np.diff(hand_actions, axis=0)
    mean_change = np.abs(changes).mean()
    max_change = np.abs(changes).max()
    
    print(f"Mean absolute change between frames: {mean_change:.4f}")
    print(f"Max absolute change between frames: {max_change:.4f}")
    
    if mean_change < 0.1:  # Small changes suggest absolute positions
        print("→ Hand actions appear to be ABSOLUTE POSITIONS")
    else:
        print("→ Hand actions appear to be VELOCITIES or LARGE MOVEMENTS")
    
    # Analyze initial hand pose
    print("\n=== Initial Hand Pose Analysis ===")
    INIT_HAND_POSE = np.array([
        1.5125739574432373, 0.5075849890708923, 0.014543981291353703, -0.0013437544694170356,
        0.013089584186673164, 0.017452778294682503, 0.008726389147341251, 0.01018078625202179,
        0.017452778294682503, 0.02617916837334633, 0.013089584186673164, 0.04072314500808716
    ])
    
    print(f"INIT_HAND_POSE: {INIT_HAND_POSE}")
    print(f"Mean hand action in data: {hand_actions.mean(axis=0)}")
    print(f"Difference: {hand_actions.mean(axis=0) - INIT_HAND_POSE}")
    
    # Check if model is trained with absolute or relative actions
    print("\n=== Training Configuration ===")
    print("relative_hand_action: False (using absolute hand positions)")
    print("This means the model predicts absolute hand positions, not relative movements")
    
    return stats, hand_actions

def suggest_fixes():
    """Suggest fixes based on analysis"""
    
    print("\n" + "="*60)
    print("SUGGESTED FIXES")
    print("="*60)
    
    print("""
1. **Hand Action Issue**: 
   The model is trained with absolute hand positions (relative_hand_action=False),
   but inference code adds predictions to INIT_HAND_POSE. This is incorrect.
   
   FIX in eval_realsense_http_novis.py line 438:
   ```python
   # Change from:
   hand_target = INIT_HAND_POSE + hand_actions[0] / XHAND_HAND_MOTOR_SCALE_FACTOR
   # To:
   hand_target = hand_actions[0] / XHAND_HAND_MOTOR_SCALE_FACTOR
   ```

2. **Action Magnitude Issue**:
   The predicted actions might be too small after unnormalization.
   
   DEBUG: Add logging to check action magnitudes:
   ```python
   print(f"Raw action from model: {action}")
   print(f"TCP action magnitude: {np.linalg.norm(tcp_actions[0, :3]):.4f}")
   print(f"Hand action magnitude: {np.linalg.norm(hand_actions[0]):.4f}")
   ```

3. **Action Horizon Issue**:
   Model predicts 16 steps but only first step is used.
   
   EXPERIMENT: Try using multiple action steps:
   ```python
   # Use first 4 actions instead of just first one
   for i in range(min(4, len(action))):
       tcp_actions = action[i:i+1, :6]
       hand_actions = action[i:i+1, 6:]
       # Send commands...
       time.sleep(dt)
   ```

4. **Proprioception Issue**:
   Verify proprioception is correctly formatted.
   
   CHECK: Print shapes and values during inference to ensure 14-dim vector.
""")

if __name__ == "__main__":
    stats, hand_actions = analyze_action_magnitude()
    suggest_fixes()