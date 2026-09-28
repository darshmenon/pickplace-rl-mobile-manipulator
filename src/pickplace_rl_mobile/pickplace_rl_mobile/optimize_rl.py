#!/usr/bin/env python3
"""Optuna hyperparameter search for pickplace RL training.

Each trial trains a fresh model in a single Gazebo world for a short budget
(--timesteps-per-trial) and is scored on its final eval reward. Optuna's
median pruner kills trials that are falling behind the pack partway through,
so a bad learning rate or tau doesn't burn its full budget. Modeled on
drl_grasping's optimize.py, adapted to this project's single-process,
single-Gazebo-world training loop (make_env/create_model from train_rl.py
and agent_factory.py) instead of rl-zoo's ExperimentManager.

A short trial cannot reach the full 5-phase task — it is only meant to be
directionally informative about which hyperparameters learn fastest early
on. Treat the winning config as a starting point for a full run, not a
final answer; verify it with a normal train_rl.py run before trusting it.
"""

import argparse
import os
import subprocess
import time

import optuna
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from pickplace_rl_mobile import agent_factory
from pickplace_rl_mobile.train_rl import make_env

_TUNABLE_ALGOS = ('tqc', 'sac', 'crossq')

# The train env below runs in-process against whichever Gazebo world the user
# already launched (ambient ROS_DOMAIN_ID, no override). Eval needs its own
# world on a distinct ROS domain / Gazebo transport partition — otherwise its
# rollout drives the *same* live Gazebo world the training env is stepping,
# desyncing training's env state once eval hands back control (mirrors why
# train_rl.py puts single-env eval in its own SubprocVecEnv; see train_rl.py
# around 'Single-env training runs the rollout env in-process...').
_EVAL_DOMAIN = int(os.environ.get('PICKPLACE_OPTIMIZE_EVAL_DOMAIN', 21))
_EVAL_PARTITION = os.environ.get('PICKPLACE_OPTIMIZE_EVAL_PARTITION', 'sim_optuna_eval')


def _launch_eval_gazebo(world: str) -> subprocess.Popen:
    """Start a second, headless Gazebo world dedicated to eval rollouts."""
    env = dict(os.environ, ROS_DOMAIN_ID=str(_EVAL_DOMAIN), GZ_PARTITION=_EVAL_PARTITION)
    proc = subprocess.Popen(
        ['ros2', 'launch', 'pickplace_rl_mobile', 'gazebo.launch.py',
         'headless:=true', f'world:={world}'],
        env=env,
    )
    time.sleep(8)  # give Gazebo time to come up before any env tries to connect
    return proc


def _sample_hparams(trial: optuna.Trial, algo: str) -> dict:
    hp = dict(
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        tau=trial.suggest_float('tau', 0.001, 0.02, log=True),
        gamma=trial.suggest_float('gamma', 0.95, 0.999),
        batch_size=trial.suggest_categorical('batch_size', [256, 512, 1024, 2048]),
        gradient_steps=trial.suggest_categorical('gradient_steps', [1, 2, 4, 8]),
        train_freq=1,
        learning_starts=1000,
        buffer_size=200_000,
        ent_coef=trial.suggest_categorical('ent_coef', ['auto', 0.1, 0.3, 0.5]),
    )
    if algo == 'tqc':
        hp['top_quantiles_to_drop_per_net'] = trial.suggest_int('top_quantiles_to_drop_per_net', 0, 4)
    return hp


def _sample_net_arch(trial: optuna.Trial):
    choice = trial.suggest_categorical('net_arch', ['256_256', '512_512', '512_512_512'])
    return tuple(int(x) for x in choice.split('_'))


class OptunaPruningEvalCallback(EvalCallback):
    """EvalCallback that also reports each eval's mean reward to the Optuna
    trial and stops training early (returning False) once the trial should
    be pruned, instead of only logging like the base callback."""

    def __init__(self, trial: optuna.Trial, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return continue_training


def objective(trial: optuna.Trial, algo: str, policy_arch: str, curriculum_stage: int,
              timesteps_per_trial: int, eval_freq: int, n_eval_episodes: int, save_dir: str) -> float:
    hparams = _sample_hparams(trial, algo)
    net_arch = _sample_net_arch(trial) if policy_arch == 'mlp' else None

    trial_dir = os.path.join(save_dir, f'trial_{trial.number}')
    os.makedirs(trial_dir, exist_ok=True)

    raw_env = DummyVecEnv([make_env(
        monitor_path=os.path.join(trial_dir, 'train.monitor.csv'),
        curriculum_stage=curriculum_stage,
        enable_domain_randomization=True,
        enable_assist=True,
    )])
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_raw_env = SubprocVecEnv([make_env(
        monitor_path=os.path.join(trial_dir, 'eval.monitor.csv'),
        ros_domain_id=_EVAL_DOMAIN,
        gz_partition=_EVAL_PARTITION,
        curriculum_stage=curriculum_stage,
        enable_domain_randomization=False,
        enable_assist=False,
    )])
    eval_env = VecNormalize(eval_raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    model = agent_factory.create_model(
        algo, env, policy_arch=policy_arch, net_arch=net_arch,
        tensorboard_log=None, device='auto', verbose=0,
        hparam_overrides=hparams,
    )

    eval_callback = OptunaPruningEvalCallback(
        trial, eval_env,
        eval_freq=max(eval_freq, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=0,
    )

    try:
        model.learn(total_timesteps=timesteps_per_trial, callback=eval_callback, progress_bar=False)
    finally:
        env.close()
        eval_env.close()

    if eval_callback.is_pruned:
        raise optuna.TrialPruned()

    reward = eval_callback.last_mean_reward
    return reward if reward is not None else -1e9


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter search for pickplace RL training')
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--timesteps-per-trial', type=int, default=30000,
                         help='Training budget per trial (default: 30000; a full run is ~500000+)')
    parser.add_argument('--eval-freq', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=5)
    parser.add_argument('--algo', type=str, default='tqc', choices=_TUNABLE_ALGOS,
                         help='Only off-policy algorithms are supported (default: tqc)')
    parser.add_argument('--policy-arch', type=str, default='mlp', choices=agent_factory.POLICY_ARCHS)
    parser.add_argument('--curriculum-stage', type=int, default=1,
                         help='Curriculum stage to search at (default: 1, the reach stage — '
                              'cheapest and fastest to get a learning-speed signal from)')
    parser.add_argument('--save-dir', type=str, default='./rl_models/optuna')
    parser.add_argument('--study-name', type=str, default='pickplace_tqc')
    parser.add_argument('--storage', type=str, default=None,
                         help='Optuna storage URL, e.g. sqlite:///rl_models/optuna/study.db '
                              '(default: in-memory — trials are lost if the process is interrupted)')
    parser.add_argument('--n-startup-trials', type=int, default=5,
                         help='Random trials before TPE sampling / pruning kick in')
    parser.add_argument('--world', type=str, default='pickplace_world.world',
                         help='World file for the dedicated eval Gazebo instance this script '
                              'launches (default: pickplace_world.world). The train env talks to '
                              'whichever Gazebo world you already have running.')

    args, _ = parser.parse_known_args()
    os.makedirs(args.save_dir, exist_ok=True)

    sampler = TPESampler(n_startup_trials=args.n_startup_trials)
    pruner = MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage, load_if_exists=bool(args.storage),
        sampler=sampler, pruner=pruner, direction='maximize',
    )

    eval_gazebo = _launch_eval_gazebo(args.world)
    try:
        study.optimize(
            lambda trial: objective(
                trial, algo=args.algo, policy_arch=args.policy_arch,
                curriculum_stage=args.curriculum_stage,
                timesteps_per_trial=args.timesteps_per_trial, eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes, save_dir=args.save_dir,
            ),
            n_trials=args.n_trials,
        )
    except KeyboardInterrupt:
        print("Interrupted — reporting best trial found so far.")
    finally:
        eval_gazebo.terminate()
        try:
            eval_gazebo.wait(timeout=10)
        except subprocess.TimeoutExpired:
            eval_gazebo.kill()

    complete = study.get_trials(states=[TrialState.COMPLETE])
    pruned = study.get_trials(states=[TrialState.PRUNED])
    print(f"\n{len(study.trials)} trials run ({len(complete)} complete, {len(pruned)} pruned)")

    if not complete:
        print("No trial completed — nothing to report.")
        return

    print(f"Best trial: #{study.best_trial.number}, eval reward={study.best_value:.2f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_hparams_path = os.path.join(args.save_dir, f'{args.study_name}_best_hparams.yaml')
    best_params = dict(study.best_params)
    net_arch = best_params.pop('net_arch', None)
    with open(best_hparams_path, 'w') as f:
        yaml.safe_dump({args.algo: best_params}, f, default_flow_style=False)
    print(f"\nBest hyperparameters written to {best_hparams_path}")
    if net_arch:
        print(f"Best net_arch: {net_arch} (not written — set manually via agent_factory.policy_kwargs_for "
              f"or pass --net-arch-equivalent options to train_rl.py once supported)")
    print("Review before merging into config/algo_hparams.yaml — these were found on a short "
          "single-stage budget and are a starting point, not a validated full-training config.")


if __name__ == '__main__':
    main()
