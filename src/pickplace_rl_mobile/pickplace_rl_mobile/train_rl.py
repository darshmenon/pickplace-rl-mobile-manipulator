#!/usr/bin/env python3

import argparse
import os
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from pickplace_rl_mobile.pickplace_env import PickPlaceEnv


def make_env(ros_domain_id=None, gz_partition=None):
    """Factory that returns a thunk creating a monitored PickPlaceEnv."""
    def _init():
        env = PickPlaceEnv(ros_domain_id=ros_domain_id, gz_partition=gz_partition)
        return Monitor(env)
    return _init


# Base ROS domain ID for parallel worlds (20-29 avoids conflict with default 0)
_MULTI_WORLD_BASE_DOMAIN = 20


def train(total_timesteps=500000, save_dir='./models', n_envs=1):
    os.makedirs(save_dir, exist_ok=True)

    print(f"Creating {n_envs} parallel pick-and-place environment(s)...")
    if n_envs > 1:
        # Each worker gets its own ROS domain + GZ partition for full isolation.
        # SubprocVecEnv runs each env in a separate OS process, so os.environ
        # set inside _init() is safe and doesn't affect other workers.
        env = SubprocVecEnv([
            make_env(ros_domain_id=_MULTI_WORLD_BASE_DOMAIN + i,
                     gz_partition=f'sim_{i}')
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([make_env()])

    # Eval env: single world, uses same domain as env 0 if multi
    eval_env = DummyVecEnv([
        make_env(ros_domain_id=_MULTI_WORLD_BASE_DOMAIN if n_envs > 1 else None,
                 gz_partition='sim_0' if n_envs > 1 else None)
    ])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=save_dir,
        name_prefix='pickplace_model'
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=max(20000 // n_envs, 1),
        deterministic=True,
        render=False
    )

    print("Initializing TQC model...")
    model = TQC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=500000,
        learning_starts=1000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=2,
        top_quantiles_to_drop_per_net=2,
        verbose=1,
        device='cuda',
        tensorboard_log=os.path.join(save_dir, 'tensorboard')
    )

    print(f"Starting TQC training for {total_timesteps} timesteps across {n_envs} env(s)...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    final_model_path = os.path.join(save_dir, 'pickplace_final_model')
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")

    env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description='Train RL agent for pick-and-place')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps (default: 500000)')
    parser.add_argument('--save-dir', type=str, default='./rl_models',
                        help='Directory to save models (default: ./rl_models)')
    parser.add_argument('--n-envs', type=int, default=1,
                        help='Number of parallel Gazebo worlds (default: 1)')

    args, unknown = parser.parse_known_args()

    train(total_timesteps=args.timesteps, save_dir=args.save_dir, n_envs=args.n_envs)


if __name__ == '__main__':
    main()
