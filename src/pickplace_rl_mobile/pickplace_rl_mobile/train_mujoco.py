#!/usr/bin/env python3
"""
Train a policy on the MuJoCo pick-place backend with parallel envs.

No ROS/Gazebo needed. Use --whole-body to let the policy drive the base in
every phase (it is locked in phases 1-3 otherwise).

  python3 -m pickplace_rl_mobile.train_mujoco --algo tqc --n-envs 8 \
      --timesteps 200000 --whole-body
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from pickplace_rl_mobile.pickplace_env_mujoco import PickPlaceEnvMuJoCo


def _make_env(whole_body, assist):
    return lambda: PickPlaceEnvMuJoCo(
        enable_assist=assist, whole_body=whole_body)


def _build_model(algo, venv, device):
    if algo == 'ppo':
        return PPO('MlpPolicy', venv, n_steps=512, batch_size=512,
                   device=device, verbose=1,
                   policy_kwargs=dict(net_arch=[256, 256]))
    if algo == 'sac':
        return SAC('MlpPolicy', venv, learning_starts=1000, batch_size=512,
                   device=device, verbose=1)
    from sb3_contrib import TQC
    return TQC('MlpPolicy', venv, learning_starts=1000, batch_size=512,
               gradient_steps=4, device=device, verbose=1,
               policy_kwargs=dict(net_arch=[512, 512, 512]))


def evaluate(model, venv, steps=150):
    """Roll out one deterministic episode with assist off; return base travel."""
    env = PickPlaceEnvMuJoCo(enable_assist=False, whole_body=venv.get_attr('whole_body')[0])
    obs, _ = env.reset(seed=0)
    path, total = [env.base_pose[:2].copy()], 0.0
    for _ in range(steps):
        norm = venv.normalize_obs(obs[None])
        act, _ = model.predict(norm, deterministic=True)
        obs, r, term, trunc, _ = env.step(act[0])
        total += r
        path.append(env.base_pose[:2].copy())
        if term or trunc:
            break
    travel = float(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1).sum())
    return total, travel, env.current_phase


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=['tqc', 'sac', 'ppo'], default='tqc')
    p.add_argument('--n-envs', type=int, default=8)
    p.add_argument('--timesteps', type=int, default=200_000)
    p.add_argument('--whole-body', action='store_true')
    p.add_argument('--no-assist', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--save-dir', default='./rl_models/mujoco')
    args = p.parse_args()

    venv = make_vec_env(_make_env(args.whole_body, not args.no_assist),
                        n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, clip_obs=10.0)
    model = _build_model(args.algo, venv, args.device)
    model.learn(args.timesteps)

    os.makedirs(args.save_dir, exist_ok=True)
    model.save(os.path.join(args.save_dir, f'{args.algo}_model'))
    venv.save(os.path.join(args.save_dir, f'{args.algo}_vecnormalize.pkl'))

    ret, travel, phase = evaluate(model, venv)
    print(f'eval return={ret:.1f} base_travel={travel:.2f}m phase={phase}')


if __name__ == '__main__':
    main()
