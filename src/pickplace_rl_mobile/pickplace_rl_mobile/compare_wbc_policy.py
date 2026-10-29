#!/usr/bin/env python3
"""
Head-to-head evaluation of the whole-body controller against trained policies
on the MuJoCo pick-place env, same seeds, same metrics.

  python3 -m pickplace_rl_mobile.compare_wbc_policy --episodes 20 \
      --policy rl_models/mujoco_overnight/best/best_model.zip
"""

import argparse
import json
import os

# Must precede the stable_baselines3 import (see train_mujoco.py).
import torch
import torch._dynamo

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pickplace_rl_mobile.pickplace_env_mujoco import PickPlaceEnvMuJoCo
from pickplace_rl_mobile.wbc_baseline import WholeBodyController, run_episode

_ALGOS = {'ppo': PPO, 'sac': SAC}


def _load_policy(path, algo, vecnorm=None):
    model = _ALGOS[algo].load(path, device='cpu')
    d, base = os.path.dirname(path), os.path.basename(path)
    candidates = [vecnorm] if vecnorm else []
    # ppo_40000_steps.zip -> ppo_vecnormalize_40000_steps.pkl, ppo_model.zip -> ppo_vecnormalize.pkl
    if base.endswith('_steps.zip'):
        prefix, steps = base[:-len('_steps.zip')].rsplit('_', 1)
        candidates.append(os.path.join(d, f'{prefix}_vecnormalize_{steps}_steps.pkl'))
    candidates += [os.path.join(d, base.replace('_model.zip', '_vecnormalize.pkl')),
                   os.path.join(d, 'best_vecnormalize.pkl'), os.path.join(d, 'vecnormalize.pkl')]
    for c in candidates:
        if c and os.path.exists(c):
            dummy = DummyVecEnv([lambda: PickPlaceEnvMuJoCo(enable_assist=False)])
            vn = VecNormalize.load(c, dummy)
            vn.training = False
            return model, vn
    raise FileNotFoundError(f'no VecNormalize stats found for {path}; pass --vecnorm')


def run_policy_episode(env, model, vn, seed):
    obs, _ = env.reset(seed=seed)
    path = [env.base_pose[:2].copy()]
    total, steps, done, info = 0.0, 0, False, {}
    while not done:
        o = vn.normalize_obs(obs[None]) if vn is not None else obs[None]
        act, _ = model.predict(o, deterministic=True)
        obs, r, term, trunc, info = env.step(act[0])
        total += r
        steps += 1
        path.append(env.base_pose[:2].copy())
        done = term or trunc
    travel = float(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1).sum())
    obj = env.real_object_pos
    return dict(success=bool(info.get('is_success')), max_phase=int(info.get('max_phase', 0)),
                verified=int(info.get('verified_grasps', 0)), steps=steps, reward=total,
                base_travel=travel,
                obj_target_err=float(np.linalg.norm(obj[:2] - env.target_pos[:2])),
                obj_z=float(obj[2]))


def summarize(name, rows):
    n = len(rows)
    return dict(
        name=name, episodes=n,
        success=sum(r['success'] for r in rows) / n,
        grasp_verified=sum(r['verified'] > 0 for r in rows) / n,
        mean_max_phase=float(np.mean([r['max_phase'] for r in rows])),
        mean_reward=float(np.mean([r['reward'] for r in rows])),
        mean_steps=float(np.mean([r['steps'] for r in rows])),
        mean_base_travel=float(np.mean([r['base_travel'] for r in rows])),
        mean_obj_target_err=float(np.mean([r['obj_target_err'] for r in rows])),
        obj_within_10cm=sum(r['obj_target_err'] < 0.10 and r['obj_z'] < 0.2 for r in rows) / n,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--policy', action='append', default=[],
                   help='path to a model zip (repeatable); normaliser pkl is auto-found')
    p.add_argument('--vecnorm', default=None, help='VecNormalize pkl for the policies')
    p.add_argument('--algo', choices=list(_ALGOS), default='ppo')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    results = []
    env = PickPlaceEnvMuJoCo(enable_assist=False, whole_body=True, curriculum_stage=0)
    ctrl = WholeBodyController(env)
    rows = [run_episode(env, ctrl, args.seed + i) for i in range(args.episodes)]
    results.append(summarize('wbc', rows))

    for path in args.policy:
        model, vn = _load_policy(path, args.algo, args.vecnorm)
        for wb in (True, False):
            penv = PickPlaceEnvMuJoCo(enable_assist=False, whole_body=wb, curriculum_stage=0)
            rows = [run_policy_episode(penv, model, vn, args.seed + i) for i in range(args.episodes)]
            results.append(summarize(f'{os.path.relpath(path)} whole_body={wb}', rows))

    for r in results:
        print(json.dumps(r))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
