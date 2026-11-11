#!/usr/bin/env python3
"""Where does the cloned policy fail? Runs it on many held-out seeds and tabulates the failures.

For every seed it records the furthest phase reached and how the episode ended, then prints a
table of failures by phase, so the next fix targets the stage that actually breaks.

  python3 -m pickplace_rl_mobile.failure_benchmark --seeds 200 --out docs/failure_benchmark.md
"""

import argparse
import collections

import torch
import torch._dynamo  # noqa: F401  (must precede sb3 imports, see train_mujoco.py)

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pickplace_rl_mobile.bc_finetune import PickPlaceEnvMuJoCo, _make_env, build_model, load_demos

PHASES = {0: 'approach', 1: 'pre-grasp', 2: 'descend', 3: 'grasp', 4: 'lift', 5: 'place'}


def run(clone_path, demos_path, seed_start, n_seeds, max_steps=400):
    venv = VecNormalize(make_vec_env(_make_env, n_envs=1, vec_env_cls=DummyVecEnv),
                        norm_obs=True, norm_reward=False, clip_obs=10.0)
    clone = build_model(venv, load_demos(demos_path))
    clone.set_parameters(clone_path)
    actor = clone.policy.actor.eval()
    env = PickPlaceEnvMuJoCo(enable_assist=False, whole_body=True, curriculum_stage=0, ee_site='pinch')
    rows = []
    for s in range(seed_start, seed_start + n_seeds):
        obs, _ = env.reset(seed=s)
        done, n, info = False, 0, {}
        while not done and n < max_steps:
            o = venv.normalize_obs(obs[None]).astype(np.float32)
            with torch.no_grad():
                a = actor(torch.as_tensor(o), deterministic=True).numpy()
            obs, _, term, trunc, info = env.step(np.clip(a, -1.0, 1.0)[0])
            done, n = term or trunc, n + 1
        rows.append(dict(seed=s, ok=bool(info.get('is_success')), phase=int(info.get('max_phase', 0)),
                         steps=n, timeout=n >= max_steps))
    venv.close()
    return rows


def report(rows):
    n = len(rows)
    ok = [r for r in rows if r['ok']]
    bad = [r for r in rows if not r['ok']]
    by_phase = collections.Counter(PHASES.get(r['phase'], str(r['phase'])) for r in bad)
    lines = [f'Clone on {n} held-out seeds (seeds {rows[0]["seed"]} to {rows[-1]["seed"]}).', '',
             f'- success: {len(ok)}/{n} = {len(ok) / n:.3f}',
             f'- mean steps on success: {np.mean([r["steps"] for r in ok]):.0f}' if ok else '- no successes',
             f'- failures that hit the step limit: {sum(r["timeout"] for r in bad)}/{len(bad)}', '',
             '| Furthest phase reached on failure | Failures |', '|---|---|']
    lines += [f'| {k} | {v} |' for k, v in sorted(by_phase.items(), key=lambda kv: -kv[1])]
    if bad:
        lines += ['', 'Failed seeds: ' + ', '.join(str(r['seed']) for r in bad)]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clone', default='rl_models/mujoco_ft_v3/tqc_bc.zip')
    p.add_argument('--stats-demos', default='demos/wbc_pinch_v3.npz')
    p.add_argument('--seeds', type=int, default=200)
    p.add_argument('--seed-start', type=int, default=5000)
    p.add_argument('--out', default=None, help='write the table to this markdown file')
    args = p.parse_args()
    text = report(run(args.clone, args.stats_demos, args.seed_start, args.seeds))
    print(text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write('# Clone failure benchmark\n\n' + text + '\n')


if __name__ == '__main__':
    main()
