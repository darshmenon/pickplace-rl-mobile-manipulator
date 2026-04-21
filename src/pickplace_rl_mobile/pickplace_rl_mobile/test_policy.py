#!/usr/bin/env python3

import argparse
import time
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sb3_contrib import TQC
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pickplace_rl_mobile.pickplace_env import PickPlaceEnv
import threading

class ImageRecorder(Node):
    def __init__(self, save_dir='images'):
        super().__init__('image_recorder')
        self.bridge = CvBridge()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.latest_image = None
        self.lock = threading.Lock()
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            
    def save_snapshot(self, filename):
        with self.lock:
            if self.latest_image is not None:
                path = os.path.join(self.save_dir, filename)
                cv2.imwrite(path, self.latest_image)
                return True
        return False


def _infer_observation_mode(model_path: str) -> str:
    data, _, _ = load_from_zip_file(model_path, device='auto')
    obs_shape = tuple(data['observation_space'].shape)
    if obs_shape == (27,):
        return 'legacy27'
    return 'full'


def _extract_object_position(obs: np.ndarray) -> np.ndarray:
    # object position is stable at 16:19 in both the full and legacy27 layouts
    if obs.shape[0] < 19:
        raise ValueError(f"Observation too short to extract object position: shape={obs.shape}")
    return np.asarray(obs[16:19], dtype=np.float32)

def test_policy(model_path, num_episodes=5, curriculum_stage=0):
    """
    Test a trained policy and record images.
    """
    # Initialize ROS for image recording
    if not rclpy.ok():
        rclpy.init()
    
    recorder = ImageRecorder()
    
    # Spin recorder in separate thread
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(recorder)
    spinner_thread = threading.Thread(target=executor.spin, daemon=True)
    spinner_thread.start()
    
    # Load model
    print(f"Loading model from {model_path}...")
    model_dir = os.path.dirname(model_path) or '.'
    vecnorm_candidates = [
        os.path.join(model_dir, 'vecnormalize.pkl'),
        os.path.join(model_dir, 'best_vecnormalize.pkl'),
    ]
    vecnorm_path = next((p for p in vecnorm_candidates if os.path.exists(p)), None)

    try:
        observation_mode = _infer_observation_mode(model_path)
        env = DummyVecEnv([
            lambda: PickPlaceEnv(
                curriculum_stage=curriculum_stage,
                observation_mode=observation_mode,
                enable_domain_randomization=False,
            )
        ])
        if not vecnorm_path:
            raise FileNotFoundError(
                f"VecNormalize stats not found next to {model_path}. "
                "Testing without the saved normalization stats would produce invalid observations."
            )

        print(f"Loading VecNormalize stats from {vecnorm_path}...")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

        model = TQC.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run episodes
    episode_rewards = []
    success_count = 0
    phase_counts = {}
    verified_grasps = 0
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Save start image
        time.sleep(1.0) # Wait for camera
        recorder.save_snapshot(f"episode_{episode+1}_start.png")
        
        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done_flags, infos = env.step(action)
            info = infos[0]
            episode_reward += float(reward[0])
            done = bool(done_flags[0])
            step += 1
            
            # Print progress
            if step % 50 == 0:
                print(
                    f"  Step {step}: Reward = {float(reward[0]):.2f}, "
                    f"phase = {info.get('phase')}, verified_grasp = {info.get('grasp_verified')}"
                )
        
        # Episode summary
        episode_rewards.append(episode_reward)
        max_phase = int(info.get('max_phase', info.get('phase', -1)))
        phase_counts[max_phase] = phase_counts.get(max_phase, 0) + 1
        verified_grasps += int(bool(info.get('grasp_verified', False)) or int(info.get('verified_grasps', 0)) > 0)
        
        # Check success
        terminal_obs = info.get('terminal_observation')
        if terminal_obs is None:
            final_obs = env.get_original_obs()[0] if isinstance(env, VecNormalize) else obs[0]
        elif isinstance(env, VecNormalize):
            final_obs = env.unnormalize_obs(np.asarray([terminal_obs]))[0]
        else:
            final_obs = terminal_obs
        obj_pos = _extract_object_position(np.asarray(final_obs))
        target_pos = np.asarray(env.venv.envs[0].target_pos if isinstance(env, VecNormalize) else env.envs[0].target_pos)
        distance_to_target = np.linalg.norm(obj_pos - target_pos)
        
        if bool(info.get('is_success', False)) or distance_to_target < 0.15:
            success_count += 1
            print(f"✓ Episode {episode + 1} SUCCESS!")
            recorder.save_snapshot(f"episode_{episode+1}_success.png")
        else:
            print(f"✗ Episode {episode + 1} failed.")
            recorder.save_snapshot(f"episode_{episode+1}_fail.png")
        print(
            f"  Episode reward: {episode_reward:.2f}, max_phase: {max_phase}, "
            f"verified_grasps: {info.get('verified_grasps', 0)}, final_dist_to_target: {distance_to_target:.3f}"
        )
            
    # Final statistics
    print("\n" + "="*50)
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Verified grasps: {verified_grasps}/{num_episodes}")
    print(f"Max phase histogram: {phase_counts}")
    print(f"Curriculum stage: {curriculum_stage}")
    print("="*50)
    
    # Cleanup
    env.close()
    recorder.destroy_node()
    rclpy.shutdown()

def main():
    parser = argparse.ArgumentParser(description='Test trained RL policy')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of test episodes')
    parser.add_argument('--curriculum-stage', type=int, default=0,
                        help='Curriculum stage used for evaluation env')
    
    args = parser.parse_args()
    
    test_policy(args.model, args.episodes, args.curriculum_stage)

if __name__ == '__main__':
    main()
