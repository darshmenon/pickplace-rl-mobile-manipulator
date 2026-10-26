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

# Must be imported before tensorflow/jax get pulled in transitively below
# (via gymnasium/stable_baselines3) — otherwise torch._dynamo's lazy import
# from inside Adam's optimizer __init__ segfaults on this install.
import torch
import torch._dynamo

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from pickplace_rl_mobile.pickplace_env_mujoco import PickPlaceEnvMuJoCo

# Same thresholds train_rl.py's CurriculumCallback uses for the Gazebo
# backend (config/curriculum.yaml defaults) — both backends share the same
# reward function, so eval-reward scale should transfer reasonably well.
_ADVANCE_THRESHOLDS = {1: 25.0, 2: 120.0, 3: 350.0, 4: 600.0}
_REVERT_THRESHOLDS = {2: 60.0, 3: 200.0, 4: 450.0, 5: 700.0}


def _make_env(whole_body, assist, curriculum_stage):
    return lambda: PickPlaceEnvMuJoCo(
        enable_assist=assist, whole_body=whole_body, curriculum_stage=curriculum_stage)


class CurriculumCallback(BaseCallback):
    """Advance/revert the staged curriculum based on EvalCallback results.

    Without this, training stays at whatever --curriculum-stage was passed
    for the entire run. Stage 0 (the default PickPlaceEnvMuJoCo curriculum_stage)
    is the full, hardest task with no staging — plain PPO essentially never
    stumbles into that compound success within a few hundred thousand steps,
    which is why an early overnight run with no curriculum sat at
    success_rate=0 for its whole duration despite training looking otherwise
    healthy (stable losses, no crashes). Starting staged (stage 1) and
    advancing on eval performance, like the Gazebo backend already does, is
    what actually lets the policy bootstrap up to the full task.
    """

    def __init__(self, eval_callback, training_venv, eval_venv, starting_stage):
        super().__init__()
        self.eval_callback = eval_callback
        self.training_venv = training_venv
        self.eval_venv = eval_venv
        self.current_stage = int(starting_stage)
        self._last_eval_calls = 0

    def _set_stage(self, stage, mean_reward, reason):
        self.training_venv.env_method('set_curriculum_stage', stage)
        self.eval_venv.env_method('set_curriculum_stage', stage)
        self.current_stage = stage
        print(f"Curriculum {reason} -> stage {stage} (eval reward {mean_reward:.1f})")

    def _on_step(self):
        if self.current_stage <= 0 or self.current_stage >= 5:
            return True
        if self.eval_callback.n_calls == self._last_eval_calls:
            return True
        if self.eval_callback.n_calls % self.eval_callback.eval_freq != 0:
            return True
        self._last_eval_calls = self.eval_callback.n_calls
        mean_reward = self.eval_callback.last_mean_reward

        revert_t = _REVERT_THRESHOLDS.get(self.current_stage)
        if revert_t is not None and mean_reward < revert_t and self.current_stage > 1:
            self._set_stage(self.current_stage - 1, mean_reward, "REVERT")
            return True

        advance_t = _ADVANCE_THRESHOLDS.get(self.current_stage)
        if advance_t is not None and mean_reward >= advance_t:
            self._set_stage(self.current_stage + 1, mean_reward, "ADVANCE")
        return True


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


def evaluate(model, venv, curriculum_stage=0, steps=150):
    """Roll out one deterministic episode with assist off; return base travel."""
    env = PickPlaceEnvMuJoCo(enable_assist=False, whole_body=venv.get_attr('whole_body')[0],
                              curriculum_stage=curriculum_stage)
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
    p.add_argument('--checkpoint-freq', type=int, default=20_000,
                    help='env steps between checkpoints (0 disables)')
    p.add_argument('--curriculum-stage', type=int, default=1,
                    help='starting stage (0=full task, no staging; 1-4=staged, '
                         'auto-advances on eval performance like the Gazebo backend)')
    p.add_argument('--eval-freq', type=int, default=20_000,
                    help='env steps between curriculum-eval rollouts')
    args = p.parse_args()

    venv = make_vec_env(_make_env(args.whole_body, not args.no_assist, args.curriculum_stage),
                        n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
                        vec_env_kwargs=dict(start_method='spawn'))
    venv = VecNormalize(venv, clip_obs=10.0)
    model = _build_model(args.algo, venv, args.device)

    os.makedirs(args.save_dir, exist_ok=True)
    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.n_envs),
            save_path=args.save_dir, name_prefix=args.algo,
            save_vecnormalize=True, verbose=1,
        ))

    eval_venv = DummyVecEnv([_make_env(args.whole_body, not args.no_assist, args.curriculum_stage)])
    eval_venv = VecNormalize(eval_venv, clip_obs=10.0, training=False, norm_reward=False)
    eval_callback = EvalCallback(
        eval_venv, eval_freq=max(1, args.eval_freq // args.n_envs),
        n_eval_episodes=5, deterministic=True,
        best_model_save_path=os.path.join(args.save_dir, 'best'), verbose=1,
    )
    callbacks.append(eval_callback)
    callbacks.append(CurriculumCallback(eval_callback, venv, eval_venv, args.curriculum_stage))

    model.learn(args.timesteps, callback=CallbackList(callbacks))

    model.save(os.path.join(args.save_dir, f'{args.algo}_model'))
    venv.save(os.path.join(args.save_dir, f'{args.algo}_vecnormalize.pkl'))

    ret, travel, phase = evaluate(model, venv, curriculum_stage=args.curriculum_stage)
    print(f'eval return={ret:.1f} base_travel={travel:.2f}m phase={phase}')


if __name__ == '__main__':
    main()
