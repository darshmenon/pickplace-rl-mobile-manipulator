#!/usr/bin/env python3
"""Residual online fine-tuning of a cloned policy.

Plain online TQC on top of the cloned actor loses the clone within a few thousand steps (the critic
has only seen demonstrations). Here the clone stays frozen and a second TQC learns a small residual
added to its action, so the clone's behaviour can't be unlearned (ResFiT, Ankile et al. 2025):

    env action = clip(clone(obs) + residual_scale * residual(obs), -1, 1)

  python3 -m pickplace_rl_mobile.residual_finetune \\
      --clone rl_models/mujoco_ft_v3/tqc_bc.zip --stats-demos demos/wbc_pinch_v3.npz \\
      --save-dir rl_models/mujoco_res --steps 60000
"""

import argparse
import os

import torch
import torch._dynamo  # noqa: F401  (must precede sb3 imports, see train_mujoco.py)

import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper, VecNormalize

from pickplace_rl_mobile.bc_finetune import PickPlaceEnvMuJoCo, _make_env, build_model, load_demos


class ResidualVecEnv(VecEnvWrapper):
    """Adds the learner's action, scaled, to the frozen clone's action for the same observation."""

    def __init__(self, venv, base_actor, scale, penalty=0.0):
        super().__init__(venv)
        self.base_actor, self.scale, self.penalty = base_actor, scale, penalty
        self._obs = None
        self._res = 0.0

    def _base(self, obs):
        with torch.no_grad():
            return self.base_actor(torch.as_tensor(obs, dtype=torch.float32), deterministic=True).numpy()

    def reset(self):
        self._obs = self.venv.reset()
        return self._obs

    def step_async(self, actions):
        self._res = np.asarray(actions)
        act = np.clip(self._base(self._obs) + self.scale * self._res, -1.0, 1.0)
        self.venv.step_async(act)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        self._obs = obs
        if self.penalty:  # pull the residual toward zero unless it earns its keep
            rew = rew - self.penalty * np.mean(np.square(self._res), axis=-1)
        return obs, rew, done, info


def evaluate(model, base_actor, venv, scale, seeds, max_steps=400):
    env = PickPlaceEnvMuJoCo(enable_assist=False, whole_body=True, curriculum_stage=0, ee_site='pinch')
    ok, phases, steps = 0, [], []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        done, n, info = False, 0, {}
        while not done and n < max_steps:
            o = venv.normalize_obs(obs[None]).astype(np.float32)
            with torch.no_grad():
                base = base_actor(torch.as_tensor(o), deterministic=True).numpy()
            res = model.predict(o, deterministic=True)[0] if model is not None else 0.0
            obs, _, term, trunc, info = env.step(np.clip(base + scale * res, -1.0, 1.0)[0])
            done, n = term or trunc, n + 1
        ok += bool(info.get('is_success'))
        phases.append(info.get('max_phase', 0))
        steps.append(n)
    return ok / len(seeds), float(np.mean(phases)), float(np.mean(steps))


class EvalSave(BaseCallback):
    def __init__(self, base_actor, venv, scale, save_dir, every, seeds, actor_delay=0):
        super().__init__()
        self.actor_delay, self._actor_on = actor_delay, actor_delay <= 0
        self.base_actor, self.venv, self.scale = base_actor, venv, scale
        self.save_dir, self.every, self.seeds = save_dir, every, seeds
        self.best, self._last = -1.0, 0

    def _on_training_start(self):
        self._last = self.num_timesteps  # non-zero when resuming
        if not self._actor_on and self.num_timesteps >= self.actor_delay:
            self._actor_on = True
        if not self._actor_on:  # let the critic learn from the clone's rollouts first
            # SB3 resets the learning rate on every update, so skip the optimizer step instead
            self.model.actor.optimizer.step = lambda *a, **k: None

    def _on_step(self):
        if not self._actor_on and self.num_timesteps >= self.actor_delay:
            self._actor_on = True
            self.model.actor.optimizer.__dict__.pop('step', None)
            print(f'[{self.num_timesteps}] actor updates enabled', flush=True)
        if self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        sr, ph, st = evaluate(self.model, self.base_actor, self.venv, self.scale, self.seeds)
        print(f'[{self.num_timesteps}] eval success={sr:.2f} max_phase={ph:.2f} steps={st:.0f}', flush=True)
        self.model.save(os.path.join(self.save_dir, 'res_latest'))
        self.model.save_replay_buffer(os.path.join(self.save_dir, 'res_replay'))  # lets --resume continue after a kill
        self.model.save(os.path.join(self.save_dir, f'res_{self.num_timesteps}'))  # pick on held-out seeds later
        if sr >= self.best:
            self.best = sr
            self.model.save(os.path.join(self.save_dir, 'res_best'))
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clone', default='rl_models/mujoco_ft_v3/tqc_bc.zip')
    p.add_argument('--stats-demos', default='demos/wbc_pinch_v3.npz',
                   help='demos whose statistics fix the obs normalisation the clone was trained with')
    p.add_argument('--save-dir', default='rl_models/mujoco_res')
    p.add_argument('--steps', type=int, default=60_000)
    p.add_argument('--scale', type=float, default=0.1, help='residual action scale')
    p.add_argument('--res-penalty', type=float, default=0.0, help='reward cost on the mean squared residual action')
    p.add_argument('--seed-start', type=int, default=1000, help='first evaluation seed')
    p.add_argument('--resume', action='store_true', help='continue from res_latest and res_replay in --save-dir')
    p.add_argument('--n-envs', type=int, default=3)
    p.add_argument('--eval-every', type=int, default=5_000)
    p.add_argument('--eval-seeds', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--learning-starts', type=int, default=2_000)
    p.add_argument('--actor-delay', type=int, default=0,
                   help='steps to keep the residual actor frozen while the critic warms up')
    p.add_argument('--residual', default=None,
                   help='saved residual model to score against the clone on the same seeds, then exit')
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.eval_seeds))

    venv = VecNormalize(make_vec_env(_make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
                                     vec_env_kwargs=dict(start_method='spawn')),
                        norm_obs=True, norm_reward=False, clip_obs=10.0)
    clone = build_model(venv, load_demos(args.stats_demos))
    clone.set_parameters(args.clone)
    base_actor = clone.policy.actor.eval()
    venv.save(os.path.join(args.save_dir, 'vecnormalize.pkl'))

    sr, ph, st = evaluate(None, base_actor, venv, args.scale, seeds)
    print(f'clone alone eval success={sr:.2f} max_phase={ph:.2f} steps={st:.0f}', flush=True)

    rvenv = ResidualVecEnv(venv, base_actor, args.scale, args.res_penalty)
    latest = os.path.join(args.save_dir, 'res_latest.zip')
    replay = os.path.join(args.save_dir, 'res_replay.pkl')
    resuming = args.resume and os.path.exists(latest) and os.path.exists(replay)  # an empty buffer can't train
    if resuming:
        model = TQC.load(latest, env=rvenv, device='cpu')
        model.load_replay_buffer(replay)
        print(f'resuming at {model.num_timesteps} steps', flush=True)
    else:
        model = TQC('MlpPolicy', rvenv, learning_starts=args.learning_starts, batch_size=256, gradient_steps=1,
                    buffer_size=max(args.steps, 100_000), ent_coef='auto_0.005', learning_rate=args.lr,
                    device='cpu', verbose=0, policy_kwargs=dict(net_arch=[256, 256]))
    if not resuming:
        with torch.no_grad():  # start from "no correction": the residual head outputs ~0
            model.actor.mu.weight.mul_(0.01)
            model.actor.mu.bias.zero_()
    if args.residual:
        model.set_parameters(args.residual)
        sr, ph, st = evaluate(model, base_actor, venv, args.scale, seeds)
        print(f'clone + {args.residual} eval success={sr:.2f} max_phase={ph:.2f} steps={st:.0f}', flush=True)
        venv.close()
        return
    if not resuming:
        sr, ph, st = evaluate(model, base_actor, venv, args.scale, seeds)
        print(f'clone + untrained residual eval success={sr:.2f} max_phase={ph:.2f} steps={st:.0f}', flush=True)

    model.learn(max(args.steps - model.num_timesteps, 0), reset_num_timesteps=False, callback=EvalSave(base_actor, venv, args.scale, args.save_dir,
                                              args.eval_every, seeds, args.actor_delay))
    model.save(os.path.join(args.save_dir, 'res_final'))


if __name__ == '__main__':
    main()
