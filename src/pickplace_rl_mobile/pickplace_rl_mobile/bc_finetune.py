#!/usr/bin/env python3
"""
Behavior-cloning warm start from whole-body-controller demos, then optional
RL fine-tuning with TQC on the MuJoCo pick-place env (ee_site='pinch').

  stage 1  BC: regress the TQC actor onto the expert's clean actions
  stage 2  critic warm-up on the demo transitions with the actor frozen, so the
           first RL updates do not wipe out the cloned behaviour
  stage 3  online TQC with the demos kept in the replay buffer; observation
           normalisation is frozen at the demo statistics so buffered demo
           observations stay valid

  python3 -m pickplace_rl_mobile.bc_finetune --demos demos/wbc_pinch.npz \
      --save-dir rl_models/mujoco_bc --finetune-steps 100000
"""

import argparse
import os

import torch
import torch._dynamo  # noqa: F401  (must precede sb3 imports, see train_mujoco.py)

import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from pickplace_rl_mobile.pickplace_env_mujoco import PickPlaceEnvMuJoCo

REWARD_SCALE = 0.01  # raw rewards reach ~1e3 per step (grasp / success bonuses)


class _ScaledReward(gym.RewardWrapper):
    def reward(self, reward):
        return reward * REWARD_SCALE


def _make_env():
    return _ScaledReward(PickPlaceEnvMuJoCo(
        enable_assist=False, whole_body=True, curriculum_stage=0, ee_site='pinch'))


def load_demos(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def build_model(venv, demos, device='cpu'):
    # freeze normalisation at the demo statistics
    venv.obs_rms.mean = demos['obs'].mean(0).astype(np.float64)
    venv.obs_rms.var = demos['obs'].var(0).astype(np.float64) + 1e-6
    venv.obs_rms.count = float(len(demos['obs']))
    venv.training = False
    return TQC('MlpPolicy', venv, learning_starts=0, batch_size=256, gradient_steps=1,
               ent_coef='auto_0.005', learning_rate=3e-4, device=device, verbose=0,
               policy_kwargs=dict(net_arch=[256, 256]))


def behavior_clone(model, venv, demos, steps=8000, batch=512, lr=1e-3):
    actor = model.policy.actor
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    obs = torch.as_tensor(venv.normalize_obs(demos['obs']), dtype=torch.float32)
    act = torch.as_tensor(demos['act_clean'], dtype=torch.float32)
    n = len(obs)
    for i in range(steps):
        idx = torch.randint(0, n, (batch,))
        loss = torch.nn.functional.mse_loss(actor(obs[idx], deterministic=True), act[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 2000 == 0 or i == steps - 1:
            print(f'  BC step {i} loss {loss.item():.5f}', flush=True)


def dagger(model, venv, demos, iters, eps, max_steps, bc_steps, eval_seeds, beta0=0.5):
    """Roll out the learner, let the whole-body controller label every visited
    state, aggregate, and retrain. Fixes the drift that plain BC suffers from."""
    from pickplace_rl_mobile.wbc_baseline import WholeBodyController

    env = PickPlaceEnvMuJoCo(enable_assist=False, whole_body=True, curriculum_stage=0,
                             ee_site='pinch')
    ctrl = WholeBodyController(env, ee_site='pinch')
    rng = np.random.default_rng(0)
    keys = ('obs', 'act_clean', 'act_exec', 'rew', 'next_obs', 'done', 'ep')
    for it in range(iters):
        beta = beta0 * (0.5 ** it)
        new = {k: [] for k in keys}
        wins = 0
        for e in range(eps):
            seed = 5000 + it * eps + e
            obs, _ = env.reset(seed=seed)
            ctrl.reset()
            for _ in range(max_steps):
                clean = ctrl.act()
                pol, _ = model.predict(venv.normalize_obs(obs[None]), deterministic=True)
                execd = clean if rng.random() < beta else pol[0]
                nxt, r, term, trunc, info = env.step(execd)
                for k, v in zip(keys, (obs, clean, execd, r, nxt, float(term), seed)):
                    new[k].append(v)
                obs = nxt
                if term or trunc:
                    break
            wins += bool(info.get('is_success'))
        for k in keys:
            demos[k] = np.concatenate([demos[k], np.asarray(new[k], dtype=np.float32)])
        behavior_clone(model, venv, demos, bc_steps)
        sr, ph, st = evaluate(model, venv, eval_seeds)
        print(f'DAgger iter {it} beta={beta:.2f} mixed-rollout wins={wins}/{eps} '
              f'data={len(demos["obs"])} -> policy eval success={sr:.2f} '
              f'max_phase={ph:.2f} steps={st:.0f}', flush=True)
    return demos


def fill_buffer(model, venv, demos):
    obs = venv.normalize_obs(demos['obs']).astype(np.float32)
    nxt = venv.normalize_obs(demos['next_obs']).astype(np.float32)
    n = model.replay_buffer.n_envs
    rew = (np.asarray(demos['rew']) * REWARD_SCALE).astype(np.float32)
    done = np.asarray(demos['done'], dtype=np.float32)
    for i in range(0, len(obs) - n + 1, n):  # the buffer stores one transition per env per add
        sl = slice(i, i + n)
        model.replay_buffer.add(obs[sl], nxt[sl], demos['act_exec'][sl], rew[sl], done[sl],
                                [{'TimeLimit.truncated': False}] * n)


def evaluate(model, venv, seeds, max_steps=400):
    env = PickPlaceEnvMuJoCo(enable_assist=False, whole_body=True, curriculum_stage=0,
                             ee_site='pinch')
    ok, phases, steps = 0, [], []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        done, n, info = False, 0, {}
        while not done and n < max_steps:
            act, _ = model.predict(venv.normalize_obs(obs[None]), deterministic=True)
            obs, _, term, trunc, info = env.step(act[0])
            done, n = term or trunc, n + 1
        ok += bool(info.get('is_success'))
        phases.append(info.get('max_phase', 0)); steps.append(n)
    return ok / len(seeds), float(np.mean(phases)), float(np.mean(steps))


class EvalSave(BaseCallback):
    def __init__(self, venv, save_dir, every, seeds):
        super().__init__()
        self.venv, self.save_dir, self.every, self.seeds = venv, save_dir, every, seeds
        self.best = -1.0
        self._last = 0

    def _on_step(self):
        if self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        sr, ph, st = evaluate(self.model, self.venv, self.seeds)
        print(f'[{self.num_timesteps}] eval success={sr:.2f} max_phase={ph:.2f} steps={st:.0f}',
              flush=True)
        self.model.save(os.path.join(self.save_dir, 'tqc_latest'))
        if sr >= self.best:
            self.best = sr
            self.model.save(os.path.join(self.save_dir, 'tqc_best'))
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--demos', default='demos/wbc_pinch.npz')
    p.add_argument('--save-dir', default='rl_models/mujoco_bc')
    p.add_argument('--bc-steps', type=int, default=8000)
    p.add_argument('--warmup-steps', type=int, default=3000)
    p.add_argument('--stats-demos', default=None,
                   help='demos whose statistics fix the obs normalisation (use the original set when resuming)')
    p.add_argument('--init-model', default=None, help='resume from this TQC zip (skips plain BC)')
    p.add_argument('--dagger-beta', type=float, default=0.5)
    p.add_argument('--dagger-iters', type=int, default=0)
    p.add_argument('--dagger-eps', type=int, default=16)
    p.add_argument('--dagger-max-steps', type=int, default=400)
    p.add_argument('--finetune-steps', type=int, default=0)
    p.add_argument('--n-envs', type=int, default=8)
    p.add_argument('--eval-seeds', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=10_000)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    demos = load_demos(args.demos)
    print(f'{len(demos["obs"])} demo transitions', flush=True)
    eval_seeds = np.arange(100, 100 + args.eval_seeds)  # same seeds as the comparison

    venv = VecNormalize(make_vec_env(_make_env, n_envs=args.n_envs,
                                     vec_env_cls=SubprocVecEnv,
                                     vec_env_kwargs=dict(start_method='spawn')),
                        norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = build_model(venv, load_demos(args.stats_demos) if args.stats_demos else demos)

    if args.init_model:
        model.set_parameters(args.init_model)
        print(f'resumed from {args.init_model}', flush=True)
    else:
        print('stage 1 behavior cloning', flush=True)
        behavior_clone(model, venv, demos, args.bc_steps)
    sr, ph, st = evaluate(model, venv, eval_seeds)
    print(f'BC policy eval success={sr:.2f} max_phase={ph:.2f} steps={st:.0f}', flush=True)
    model.save(os.path.join(args.save_dir, 'tqc_bc'))
    venv.save(os.path.join(args.save_dir, 'vecnormalize.pkl'))

    if args.dagger_iters > 0:
        print('stage 1b DAgger', flush=True)
        demos = dagger(model, venv, demos, args.dagger_iters, args.dagger_eps,
                       args.dagger_max_steps, args.bc_steps // 2, eval_seeds,
                       beta0=args.dagger_beta)
        model.save(os.path.join(args.save_dir, 'tqc_dagger'))
        np.savez_compressed(os.path.join(args.save_dir, 'demos_dagger.npz'), **demos)

    if args.finetune_steps > 0:
        print('stage 2 critic warm-up (actor frozen)', flush=True)
        fill_buffer(model, venv, demos)
        # SB3 resets every optimiser lr from its schedule on each train() call, so zeroing the lr
        # does not freeze the actor. Switch off its gradients for the critic warm-up instead.
        for p_ in model.actor.parameters():
            p_.requires_grad_(False)
        model._n_updates = 0
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(None, ['stdout']))  # train() logs, and learn() has not set one up yet
        model.policy.set_training_mode(True)
        chunk = 1000
        for done in range(0, args.warmup_steps, chunk):
            model.train(gradient_steps=min(chunk, args.warmup_steps - done), batch_size=256)
            print(f'  warm-up {min(done + chunk, args.warmup_steps)}/{args.warmup_steps}', flush=True)
        for p_ in model.actor.parameters():
            p_.requires_grad_(True)
        model.lr_schedule = lambda _: 1e-4  # gentle online fine-tuning, the schedule is what sets the lr
        sr, ph, st = evaluate(model, venv, eval_seeds)
        print(f'after warm-up eval success={sr:.2f} max_phase={ph:.2f}', flush=True)

        print('stage 3 online fine-tuning', flush=True)
        model.learn(args.finetune_steps, reset_num_timesteps=True,
                    callback=EvalSave(venv, args.save_dir, args.eval_every, eval_seeds))
        model.save(os.path.join(args.save_dir, 'tqc_final'))


if __name__ == '__main__':
    main()
