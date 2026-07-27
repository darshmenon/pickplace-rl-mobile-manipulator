#!/usr/bin/env python3

import argparse
import math
import os
import torch
import yaml

# torch.optim.Adam lazily imports torch._dynamo (and transitively triton) on
# construction. If that first happens after TensorFlow is loaded (sb3_contrib
# pulls it in transitively), triton's bundled LLVM collides with TensorFlow's
# and segfaults on import. Constructing a throwaway Adam here forces that
# import to happen first, while the process is still TF-free.
torch.optim.Adam(torch.nn.Linear(1, 1).parameters())

from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from pickplace_rl_mobile.pickplace_env import PickPlaceEnv
from pickplace_rl_mobile import agent_factory


class AtomicCheckpointCallback(CheckpointCallback):
    """CheckpointCallback saves straight to its final path, so a process kill
    mid-write (OOM, manual kill, crash) leaves a truncated, unloadable zip —
    which is exactly what happened to a prior run's checkpoint at 785000 steps.
    Save to a temp path first and rename into place, which is atomic on the
    same filesystem, so a checkpoint file is either the old one or a complete
    new one, never a partial write.
    """

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True
        final_path = self._checkpoint_path(extension='zip')
        tmp_path = final_path + '.tmp'
        self.model.save(tmp_path)
        os.replace(tmp_path, final_path)
        if self.verbose >= 2:
            print(f"Saving model checkpoint to {final_path}")
        return True


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


class EntropyDecayCallback(BaseCallback):
    """Linearly decay entropy coefficient from initial to final over decay_steps."""

    def __init__(self, initial: float = 0.3, final: float = 0.05, decay_steps: int = 100000):
        super().__init__()
        self.initial = initial
        self.final = final
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        t = min(self.num_timesteps, self.decay_steps)
        ent_coef = self.initial + (self.final - self.initial) * (t / self.decay_steps)
        self.model.ent_coef = ent_coef
        self.model.ent_coef_tensor = torch.tensor(ent_coef, device=self.model.device)
        if self.model.log_ent_coef is not None:
            with torch.no_grad():
                self.model.log_ent_coef.data.fill_(math.log(max(ent_coef, 1e-8)))
        return True


def _load_curriculum_config() -> dict:
    """Load stage thresholds from config/curriculum.yaml, falling back to the
    values this project has always used if the file is missing/invalid."""
    defaults = {
        # Lowered from (50/250/600/900) → (30/150/400/700) for domain randomization,
        # then further reduced to account for multi-shape generalization difficulty.
        'advance_thresholds': {1: 25.0, 2: 120.0, 3: 350.0, 4: 600.0},
        'revert_thresholds': {2: 60.0, 3: 200.0, 4: 450.0, 5: 700.0},
        'adaptive': {'window': 5, 'epsilon': 5.0, 'floor_ratio': 0.6},
    }
    path = agent_factory.resolve_config_path('curriculum.yaml')
    if not path:
        return defaults
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key, fallback in defaults.items():
            value = data.get(key)
            if isinstance(value, dict):
                merged = dict(fallback)
                merged.update({int(k): v for k, v in value.items()} if key != 'adaptive'
                              else value)
                defaults[key] = merged
        return defaults
    except Exception as exc:
        print(f"Warning: could not load {path} ({exc}); using built-in curriculum thresholds")
        return defaults


class CurriculumCallback(BaseCallback):
    """Advance or revert staged curricula based on evaluation results.

    Default mode advances/reverts on fixed reward thresholds (from
    config/curriculum.yaml). With adaptive=True, a stage instead advances
    once eval reward plateaus (rolling improvement below `epsilon` over the
    last `window` evals) provided it has cleared `floor_ratio` of the fixed
    advance threshold as a safety floor — this lets stages that learn faster
    than expected advance sooner, and slower stages still get the same floor
    guarantee as fixed-threshold mode.
    """

    def __init__(self, eval_callback: EvalCallback, eval_env, starting_stage: int,
                 adaptive: bool = False):
        super().__init__()
        self.eval_callback = eval_callback
        self.eval_env = eval_env
        self.current_stage = int(starting_stage)
        self.adaptive = adaptive
        self._last_eval_step = 0
        self._history = {}

        cfg = _load_curriculum_config()
        self.ADVANCE_THRESHOLDS = cfg['advance_thresholds']
        self.REVERT_THRESHOLDS = cfg['revert_thresholds']
        self._adapt_window = int(cfg['adaptive']['window'])
        self._adapt_epsilon = float(cfg['adaptive']['epsilon'])
        self._adapt_floor_ratio = float(cfg['adaptive']['floor_ratio'])

    def _set_stage(self, stage: int, mean_reward: float, reason: str) -> None:
        self.training_env.env_method('set_curriculum_stage', stage)
        self.eval_env.env_method('set_curriculum_stage', stage)
        self.current_stage = stage
        print(f"Curriculum {reason} to stage {stage} (eval reward {mean_reward:.2f})")

    def _plateaued(self, mean_reward: float) -> bool:
        history = self._history.setdefault(self.current_stage, [])
        history.append(mean_reward)
        if len(history) < self._adapt_window:
            return False
        window = history[-self._adapt_window:]
        improvement = window[-1] - window[0]
        floor = self.ADVANCE_THRESHOLDS.get(self.current_stage)
        above_floor = floor is None or window[-1] >= floor * self._adapt_floor_ratio
        return improvement < self._adapt_epsilon and above_floor

    def _on_step(self) -> bool:
        if self.current_stage <= 0:
            return True
        if self.eval_callback.n_calls == self._last_eval_step:
            return True
        if self.eval_callback.n_calls % self.eval_callback.eval_freq != 0:
            return True

        self._last_eval_step = self.eval_callback.n_calls
        mean_reward = self.eval_callback.last_mean_reward

        # Revert if reward has regressed significantly
        revert_threshold = self.REVERT_THRESHOLDS.get(self.current_stage)
        if revert_threshold is not None and mean_reward < revert_threshold and self.current_stage > 1:
            self._set_stage(self.current_stage - 1, mean_reward, "REVERT")
            return True

        if self.current_stage >= 5:
            return True

        # Advance if reward is strong enough (fixed threshold), or — in
        # adaptive mode — once reward plateaus above the safety floor.
        advance_threshold = self.ADVANCE_THRESHOLDS.get(self.current_stage)
        if advance_threshold is not None and mean_reward >= advance_threshold:
            self._set_stage(self.current_stage + 1, mean_reward, "ADVANCE")
        elif self.adaptive and self._plateaued(mean_reward):
            self._set_stage(self.current_stage + 1, mean_reward, "ADVANCE (adaptive plateau)")

        return True


def _infer_observation_mode(load_model: str | None) -> str:
    if not load_model:
        return 'full'
    try:
        data, _, _ = load_from_zip_file(load_model, device='auto')
        obs_shape = tuple(data['observation_space'].shape)
    except Exception as exc:
        print(f"Could not inspect checkpoint observation space ({exc}); using full observation mode")
        return 'full'

    if obs_shape == (27,):
        print(f"Detected legacy checkpoint observation space {obs_shape}; enabling legacy27 observation mode")
        return 'legacy27'
    print(f"Detected checkpoint observation space {obs_shape}; using full observation mode")
    return 'full'


def _load_vecnormalize_or_fallback(vecnorm_path: str, raw_env, norm_reward: bool):
    """Load VecNormalize stats when compatible, otherwise start fresh."""
    try:
        env = VecNormalize.load(vecnorm_path, raw_env)
        env.training = True
        env.norm_reward = norm_reward
        return env
    except Exception as exc:
        print(f"Warning: could not load VecNormalize stats from {vecnorm_path} ({exc}); starting fresh")
        return VecNormalize(raw_env, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)


def _reset_replay_buffer(model) -> None:
    """Recreate the replay buffer to match the current env spaces."""
    replay_buffer_type = type(model.replay_buffer)
    handle_timeout_termination = getattr(model.replay_buffer, 'handle_timeout_termination', True)
    model.replay_buffer = replay_buffer_type(
        model.buffer_size,
        model.observation_space,
        model.action_space,
        device=model.device,
        n_envs=model.n_envs,
        optimize_memory_usage=model.optimize_memory_usage,
        handle_timeout_termination=handle_timeout_termination,
    )


def _schedule_replay_prefill(model, pre_fill: int = 20000) -> None:
    model.learning_starts = model.num_timesteps + pre_fill
    print(f"Pre-filling fresh replay buffer with {pre_fill} steps before updates")


def make_env(
    monitor_path=None,
    ros_domain_id=None,
    gz_partition=None,
    curriculum_stage=0,
    observation_mode='full',
    enable_domain_randomization=True,
    enable_assist=True,
):
    """Factory that returns a thunk creating a monitored PickPlaceEnv."""
    def _init():
        env = PickPlaceEnv(
            ros_domain_id=ros_domain_id,
            gz_partition=gz_partition,
            curriculum_stage=curriculum_stage,
            observation_mode=observation_mode,
            enable_domain_randomization=enable_domain_randomization,
            enable_assist=enable_assist,
        )
        # Log additional info metrics in monitor.csv
        kw = (
            'phase',
            'max_phase',
            'object_grasped',
            'grasp_attempts',
            'verified_grasps',
            'grasp_verified',
            'real_object_z',
            'object_height_delta',
            'dist_to_obj',
            'finger_joint',
            'is_success',
            'success_rate',
            'assist_scale',
            'curriculum_stage',
            'curriculum_target_phase',
            'stage_success',
        )
        return Monitor(env, filename=monitor_path, info_keywords=kw)
    return _init


# Base ROS domain ID for parallel worlds (20-29 avoids conflict with default 0).
# Overridable so a second training run (e.g. a different policy_arch experiment)
# can share the machine without colliding on ROS domain / Gazebo transport with
# a run already using the defaults.
_MULTI_WORLD_BASE_DOMAIN = int(os.environ.get('PICKPLACE_DOMAIN_BASE', 20))
_SINGLE_WORLD_TRAIN_PARTITION = os.environ.get('PICKPLACE_TRAIN_PARTITION', 'sim_0')
_SINGLE_WORLD_EVAL_PARTITION = os.environ.get('PICKPLACE_EVAL_PARTITION', 'sim_1')
_DEFAULT_N_EVAL_EPISODES = 10


def train(total_timesteps=500000, save_dir='./models', n_envs=1, load_model=None, curriculum_stage=0,
          algo='tqc', policy_arch='mlp', adaptive_curriculum=False):
    os.makedirs(save_dir, exist_ok=True)
    load_model = load_model.strip() if isinstance(load_model, str) else load_model
    observation_mode = _infer_observation_mode(load_model)

    vecnorm_path = os.path.join(save_dir, 'vecnormalize.pkl')
    monitor_dir = os.path.join(save_dir, 'monitor')
    eval_monitor_dir = os.path.join(save_dir, 'eval_monitor')
    best_model_dir = os.path.join(save_dir, 'best_model')
    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(eval_monitor_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    load_model_dir = os.path.dirname(load_model) if load_model else ''
    vecnorm_candidates = [
        vecnorm_path,
        os.path.join(load_model_dir, 'vecnormalize.pkl') if load_model_dir else '',
        os.path.join(load_model_dir, 'best_vecnormalize.pkl') if load_model_dir else '',
        os.path.join(save_dir, 'best_model', 'best_vecnormalize.pkl'),
    ]
    vecnorm_candidates = [path for path in vecnorm_candidates if path]
    resume_vecnorm_path = next((path for path in vecnorm_candidates if os.path.exists(path)), None)

    print(
        f"Creating {n_envs} parallel pick-and-place environment(s) "
        f"for curriculum stage {curriculum_stage}..."
    )
    if n_envs > 1:
        raw_env = SubprocVecEnv([
            make_env(
                monitor_path=os.path.join(monitor_dir, f'train_env_{i}.monitor.csv'),
                ros_domain_id=_MULTI_WORLD_BASE_DOMAIN + i,
                gz_partition=f'sim_{i}',
                curriculum_stage=curriculum_stage,
                observation_mode=observation_mode,
                enable_domain_randomization=True,
                enable_assist=True,
            )
            for i in range(n_envs)
        ])
    else:
        raw_env = DummyVecEnv([
            make_env(
                monitor_path=os.path.join(monitor_dir, 'train_env_0.monitor.csv'),
                ros_domain_id=_MULTI_WORLD_BASE_DOMAIN,
                gz_partition=_SINGLE_WORLD_TRAIN_PARTITION,
                curriculum_stage=curriculum_stage,
                observation_mode=observation_mode,
                enable_domain_randomization=True,
                enable_assist=True,
            )
        ])

    # VecNormalize: normalises obs and rewards online — critical when obs spans
    # joint angles (rad), positions (m), and velocities (rad/s) at very different scales.
    if load_model and resume_vecnorm_path:
        print(f"Loading VecNormalize stats from {resume_vecnorm_path}...")
        env = _load_vecnormalize_or_fallback(resume_vecnorm_path, raw_env, norm_reward=True)
    else:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if n_envs > 1:
        eval_raw_env = DummyVecEnv([
            make_env(
                monitor_path=os.path.join(eval_monitor_dir, 'eval_env_0.monitor.csv'),
                curriculum_stage=curriculum_stage,
                observation_mode=observation_mode,
                enable_domain_randomization=False,
                enable_assist=False,
            )
        ])
    else:
        # Single-env training runs the rollout env in-process, so evaluation must live in
        # its own subprocess to avoid reusing the same rclpy context / Gazebo world.
        eval_raw_env = SubprocVecEnv([
            make_env(
                monitor_path=os.path.join(eval_monitor_dir, 'eval_env_0.monitor.csv'),
                ros_domain_id=_MULTI_WORLD_BASE_DOMAIN + 1,
                gz_partition=_SINGLE_WORLD_EVAL_PARTITION,
                curriculum_stage=curriculum_stage,
                observation_mode=observation_mode,
                enable_domain_randomization=False,
                enable_assist=False,
            )
        ])
    if load_model and resume_vecnorm_path:
        eval_env = _load_vecnormalize_or_fallback(resume_vecnorm_path, eval_raw_env, norm_reward=False)
    else:
        eval_env = VecNormalize(eval_raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    ckpt_freq = max(10000 // n_envs, 1)
    eval_freq = max(10000 // n_envs, 1)
    checkpoint_callback = AtomicCheckpointCallback(
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
        n_eval_episodes=_DEFAULT_N_EVAL_EPISODES,
        deterministic=True,
        callback_on_new_best=SaveBestVecNormalizeCallback(best_model_dir),
    )
    curriculum_callback = CurriculumCallback(
        eval_callback=eval_callback,
        eval_env=eval_env,
        starting_stage=curriculum_stage,
        adaptive=adaptive_curriculum,
    )

    if load_model:
        print(f"Loading existing {algo.upper()} model from {load_model}...")
        model = agent_factory.load_model(
            algo, load_model, env=env, tensorboard_log=os.path.join(save_dir, 'tensorboard'),
        )

        if agent_factory.is_entropy_tunable(algo):
            # Keep ent_coef at 0.1 on resume — 0.3 was too aggressive and hurt convergence.
            model.ent_coef = 0.1
            model.ent_coef_tensor = torch.tensor(0.1, device=model.device)
            if model.log_ent_coef is not None:
                with torch.no_grad():
                    model.log_ent_coef.data.fill_(math.log(0.1))
            print("Set ent_coef=0.1 on resume")

        if agent_factory.is_off_policy(algo):
            model.gradient_steps = 4
            print("Set gradient_steps=4 on resume")

            # Load replay buffer if available — avoids cold-start problem entirely.
            replay_buf_candidates = [
                os.path.join(save_dir, 'replay_buffer.pkl'),
                os.path.join(load_model_dir, 'replay_buffer.pkl') if load_model_dir else '',
            ]
            replay_buf_path = next((path for path in replay_buf_candidates if path and os.path.exists(path)), None)
            if replay_buf_path:
                try:
                    model.load_replay_buffer(replay_buf_path)
                    replay_obs_shape = tuple(model.replay_buffer.observations.shape[2:])
                    expected_obs_shape = tuple(model.observation_space.shape)
                    if replay_obs_shape != expected_obs_shape:
                        print(
                            f"Warning: replay buffer obs shape {replay_obs_shape} does not match "
                            f"env shape {expected_obs_shape}; resetting buffer"
                        )
                        _reset_replay_buffer(model)
                        _schedule_replay_prefill(model)
                    else:
                        print(f"Loaded replay buffer ({model.replay_buffer.size()} transitions)")
                except Exception as exc:
                    print(f"Warning: could not load replay buffer from {replay_buf_path} ({exc})")
                    _reset_replay_buffer(model)
                    _schedule_replay_prefill(model)
            else:
                # No saved buffer — pre-fill with policy rollouts before first update
                # so the agent has a diverse starting distribution to learn from.
                _schedule_replay_prefill(model)
    else:
        print(f"Initializing new {algo.upper()} model (policy_arch={policy_arch})...")
        model = agent_factory.create_model(
            algo, env, policy_arch=policy_arch,
            tensorboard_log=os.path.join(save_dir, 'tensorboard'),
            device='auto', verbose=1,
        )

    callbacks = [checkpoint_callback, vecnorm_callback, eval_callback, curriculum_callback]
    if not load_model and agent_factory.is_entropy_tunable(algo):
        callbacks.append(EntropyDecayCallback(initial=0.3, final=0.05, decay_steps=100000))

    print(f"Starting {algo.upper()} training for {total_timesteps} timesteps across {n_envs} env(s)...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=not bool(load_model),
    )

    final_model_path = os.path.join(save_dir, 'pickplace_final_model')
    model.save(final_model_path)
    env.save(vecnorm_path)
    if agent_factory.is_off_policy(algo):
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
    parser.add_argument('--curriculum-stage', type=int, default=1,
                        help='Curriculum stage: 0=full, 1=reach, 2=grasp, 3=lift, 4=transport, 5=place')
    parser.add_argument('--algo', type=str, default='tqc', choices=agent_factory.ALGOS,
                        help='RL algorithm (default: tqc)')
    parser.add_argument('--policy-arch', type=str, default='mlp', choices=agent_factory.POLICY_ARCHS,
                        help='Policy feature-extractor architecture (default: mlp)')
    parser.add_argument('--adaptive-curriculum', action='store_true',
                        help='Advance curriculum stages on reward-plateau detection instead of '
                             'fixed thresholds only (see config/curriculum.yaml)')

    args, unknown = parser.parse_known_args()

    train(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        n_envs=args.n_envs,
        load_model=args.load_model,
        curriculum_stage=args.curriculum_stage,
        algo=args.algo,
        policy_arch=args.policy_arch,
        adaptive_curriculum=args.adaptive_curriculum,
    )


if __name__ == '__main__':
    main()
