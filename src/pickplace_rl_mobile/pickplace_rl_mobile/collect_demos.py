#!/usr/bin/env python3
"""
Record expert demonstrations from the whole-body controller (wbc_baseline.py)
on the MuJoCo pick-place env, for behavior-cloning / demo-seeded RL.

The controller acts through the env's normal 9-D action, so every recorded
(obs, action) pair is something a policy can reproduce exactly. Exploration
noise is added to the *executed* action (DART-style) to widen state coverage,
while the *clean* expert action is stored as the BC label. Only successful
episodes are kept.

  python3 -m pickplace_rl_mobile.collect_demos --episodes 160 --workers 16 \
      --out demos/wbc_pinch.npz
"""

import argparse
import multiprocessing as mp
import os

import numpy as np

_FIELDS = ('obs', 'act_clean', 'act_exec', 'rew', 'next_obs', 'done', 'ep')


def _collect(args):
    seeds, noise = args
    from pickplace_rl_mobile.pickplace_env_mujoco import PickPlaceEnvMuJoCo
    from pickplace_rl_mobile.wbc_baseline import WholeBodyController

    env = PickPlaceEnvMuJoCo(enable_assist=False, whole_body=True,
                             curriculum_stage=0, ee_site='pinch')
    ctrl = WholeBodyController(env, ee_site='pinch')
    out = {k: [] for k in _FIELDS}
    kept = tried = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=int(seed))
        ctrl.reset()
        ep = {k: [] for k in _FIELDS}
        done = False
        while not done:
            clean = ctrl.act()
            execd = clean.copy()
            # noise on arm + base only; the gripper command stays decisive
            execd[:6] += rng.normal(0, noise, 6)
            execd[7:] += rng.normal(0, 5 * noise, 2)
            execd = np.clip(execd, -1, 1).astype(np.float32)
            nxt, r, term, trunc, info = env.step(execd)
            ep['obs'].append(obs); ep['act_clean'].append(clean); ep['act_exec'].append(execd)
            ep['rew'].append(r); ep['next_obs'].append(nxt); ep['done'].append(float(term))
            ep['ep'].append(int(seed))
            obs = nxt
            done = term or trunc
        tried += 1
        if info.get('is_success'):
            kept += 1
            for k in _FIELDS:
                out[k].extend(ep[k])
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}, kept, tried


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=160)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--noise', type=float, default=0.02,
                   help='std of action noise on executed arm actions (base gets 5x)')
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--out', default='demos/wbc_pinch.npz')
    args = p.parse_args()

    seeds = np.arange(args.seed, args.seed + args.episodes)
    chunks = [(c, args.noise) for c in np.array_split(seeds, args.workers) if len(c)]
    with mp.get_context('spawn').Pool(len(chunks)) as pool:
        results = pool.map(_collect, chunks)

    data = {k: np.concatenate([r[0][k] for r in results]) for k in _FIELDS}
    kept = sum(r[1] for r in results)
    tried = sum(r[2] for r in results)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f'kept {kept}/{tried} successful episodes, {len(data["obs"])} transitions -> {args.out}')


if __name__ == '__main__':
    main()
