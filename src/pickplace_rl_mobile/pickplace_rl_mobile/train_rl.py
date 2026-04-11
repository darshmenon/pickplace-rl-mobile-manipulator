#!/usr/bin/env python3

import argparse
import math
import os
import torch
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from pickplace_rl_mobile.pickplace_env import PickPlaceEnv


class SaveVecNormalizeCallback(BaseCallback):
    """Save VecNormalize stats and replay buffer alongside every model checkpoint."""

    def __init__(self, save_path: str, save_freq: int):
        super().__init__()
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(os.path.join(self.save_path, 'vecnormalize.pkl'))
            self.model.save_replay_buffer(os.path.join(self.save_path, 'replay_buffer'))
        return True


class SaveBestVecNormalizeCallback(BaseCallback):
    """Persist normalization stats whenever EvalCallback finds a new best model."""

    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        if isinstance(self.training_env, VecNormalize):
            self.training_env.save(os.path.join(self.save_path, 'best_vecnormalize.pkl'))
        return True


def make_env(monitor_path=None, ros_domain_id=None, gz_partition=None):
    """Factory that returns a thunk creating a monitored PickPlaceEnv."""
    def _init():
        env = PickPlaceEnv(ros_domain_id=ros_domain_id, gz_partition=gz_partition)
        # Log additional info metrics in monitor.csv
        kw = (
            'phase',
            'max_phase',
            'grasp_attempts',
            'verified_grasps',
            'grasp_verified',
            'real_object_z',
            'object_height_delta',
            'dist_to_obj',
            'finger_joint',
            'is_success',
        )
        return Monitor(env, filename=monitor_path, info_keywords=kw)
    return _init


# Base ROS domain ID for parallel worlds (20-29 avoids conflict with default 0)
_MULTI_WORLD_BASE_DOMAIN = 20


def train(total_timesteps=500000, save_dir='./models', n_envs=1, load_model=None):
    os.makedirs(save_dir, exist_ok=True)

    vecnorm_path = os.path.join(save_dir, 'vecnormalize.pkl')
    monitor_dir = os.path.join(save_dir, 'monitor')
    eval_monitor_dir = os.path.join(save_dir, 'eval_monitor')
    best_model_dir = os.path.join(save_dir, 'best_model')
    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(eval_monitor_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    print(f"Creating {n_envs} parallel pick-and-place environment(s)...")
    if n_envs > 1:
        raw_env = SubprocVecEnv([
            make_env(
                     monitor_path=os.path.join(monitor_dir, f'train_env_{i}.monitor.csv'),
                     ros_domain_id=_MULTI_WORLD_BASE_DOMAIN + i,
                     gz_partition=f'sim_{i}')
            for i in range(n_envs)
        ])
    else:
        raw_env = DummyVecEnv([
            make_env(monitor_path=os.path.join(monitor_dir, 'train_env_0.monitor.csv'))
        ])

    # VecNormalize: normalises obs and rewards online — critical when obs spans
    # joint angles (rad), positions (m), and velocities (rad/s) at very different scales.
    if load_model and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path}...")
        env = VecNormalize.load(vecnorm_path, raw_env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_raw_env = DummyVecEnv([
        make_env(monitor_path=os.path.join(eval_monitor_dir, 'eval_env_0.monitor.csv'))
    ])
    if os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, eval_raw_env)
    else:
        eval_env = VecNormalize(eval_raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    ckpt_freq = max(10000 // n_envs, 1)
    eval_freq = max(5000 // n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=save_dir,
        name_prefix='pickplace_model'
    )
    vecnorm_callback = SaveVecNormalizeCallback(save_path=save_dir, save_freq=ckpt_freq)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=save_dir,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=SaveBestVecNormalizeCallback(best_model_dir),
    )

    # 3-layer 512-unit network: bigger than default [256,256] to capture
    # the complex mapping from 24-dim obs to 9-dim continuous actions.
    policy_kwargs = dict(net_arch=[512, 512, 512])

    if load_model:
        print(f"Loading existing model from {load_model}...")
        model = TQC.load(
            load_model,
            env=env,
            tensorboard_log=os.path.join(save_dir, 'tensorboard'),
        )

        # Keep ent_coef at 0.1 on resume — 0.3 was too aggressive and hurt convergence.
        model.ent_coef = 0.1
        model.ent_coef_tensor = torch.tensor(0.1, device=model.device)
        if model.log_ent_coef is not None:
            with torch.no_grad():
                model.log_ent_coef.data.fill_(math.log(0.1))
        print("Set ent_coef=0.1 on resume")

        model.gradient_steps = 4
        print("Set gradient_steps=4 on resume")

        # Load replay buffer if available — avoids cold-start problem entirely.
        replay_buf_path = os.path.join(save_dir, 'replay_buffer.pkl')
        if os.path.exists(replay_buf_path):
            model.load_replay_buffer(replay_buf_path)
            print(f"Loaded replay buffer ({model.replay_buffer.size()} transitions)")
        else:
            # No saved buffer — pre-fill with policy rollouts before first update
            # so TQC has a diverse starting distribution to learn from.
            pre_fill = 20000
            model.learning_starts = model.num_timesteps + pre_fill
            print(f"No replay buffer found — pre-filling with {pre_fill} steps before updates")
    else:
        print("Initializing new TQC model...")
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
            gradient_steps=1,
            top_quantiles_to_drop_per_net=2,
            ent_coef=0.3,           # fixed — auto-tuning blows up for this env
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='cuda',
            tensorboard_log=os.path.join(save_dir, 'tensorboard')
        )

    print(f"Starting TQC training for {total_timesteps} timesteps across {n_envs} env(s)...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, vecnorm_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=not bool(load_model),
    )

    final_model_path = os.path.join(save_dir, 'pickplace_final_model')
    model.save(final_model_path)
    env.save(vecnorm_path)
    model.save_replay_buffer(os.path.join(save_dir, 'replay_buffer'))
    print(f"Training complete! Model → {final_model_path}, VecNormalize → {vecnorm_path}")

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
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to a saved model to resume training (default: None)')

    args, unknown = parser.parse_known_args()

    train(total_timesteps=args.timesteps, save_dir=args.save_dir, n_envs=args.n_envs, load_model=args.load_model)


if __name__ == '__main__':
    main()
