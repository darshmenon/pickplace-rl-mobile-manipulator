#!/usr/bin/env python3
"""Algorithm and policy-architecture factory for pickplace RL training.

Centralizes model construction so train_rl.py can switch between TQC
(default), SAC, PPO, and PPO+LSTM (recurrent) via --algo, and between an
MLP or Transformer feature-extractor policy via --policy-arch, without
hardcoding a single algorithm's API throughout the training script.

Hyperparameters default to the values train_rl.py used to hardcode for TQC
and are overridable via config/algo_hparams.yaml (see resolve_config_path).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import yaml

# torch.optim.Adam lazily imports torch._dynamo (and transitively triton) on
# construction. If that first happens after TensorFlow is loaded (sb3_contrib
# pulls it in transitively below), triton's bundled LLVM collides with
# TensorFlow's and segfaults on import. Constructing a throwaway Adam here
# forces that import to happen first, while the process is still TF-free —
# see the matching guard at the top of train_rl.py.
torch.optim.Adam(torch.nn.Linear(1, 1).parameters())

from sb3_contrib import TQC, RecurrentPPO
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

ALGOS = ('tqc', 'sac', 'ppo', 'ppo_lstm')
POLICY_ARCHS = ('mlp', 'transformer')

_ALGO_CLASSES = {'tqc': TQC, 'sac': SAC, 'ppo': PPO, 'ppo_lstm': RecurrentPPO}

# Off-policy algorithms use a replay buffer and an auto-tunable entropy
# coefficient tensor; on-policy algorithms (PPO family) use neither.
_OFF_POLICY_ALGOS = {'tqc', 'sac'}
_ENTROPY_TUNABLE_ALGOS = {'tqc', 'sac'}

# Built-in fallback hyperparameters, used when config/algo_hparams.yaml is
# missing or doesn't define a given algorithm. These match the values
# train_rl.py hardcoded for TQC prior to the multi-algorithm refactor.
_DEFAULT_HPARAMS = {
    'tqc': dict(
        learning_rate=3e-4, buffer_size=1_000_000, learning_starts=1000,
        batch_size=1024, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=4,
        top_quantiles_to_drop_per_net=2, ent_coef=0.3,
    ),
    'sac': dict(
        learning_rate=3e-4, buffer_size=1_000_000, learning_starts=1000,
        batch_size=1024, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=4,
        ent_coef=0.3,
    ),
    'ppo': dict(
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    ),
    'ppo_lstm': dict(
        learning_rate=3e-4, n_steps=1024, batch_size=128, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    ),
}

# Named observation groups for PickPlaceEnv's flat observation vectors, used
# to tokenize the observation for the transformer feature extractor. Order
# and sizes must match PickPlaceEnv.get_observation() (see pickplace_env.py).
_OBS_GROUPS_FULL = (
    ('joint_pos', 6), ('joint_vel', 6), ('finger_pos', 1), ('ee_pos', 3),
    ('obj_pos', 3), ('ee_to_obj', 3), ('ee_to_target', 3), ('obj_to_target', 3),
    ('obj_in_base', 3), ('gripper_error', 1), ('grasped', 1), ('phase', 1),
    ('base_pose', 3), ('prev_action', 9),
)
_OBS_GROUPS_LEGACY27 = (
    ('joint_pos', 6), ('joint_vel', 6), ('finger_pos', 1), ('ee_pos', 3),
    ('obj_pos', 3), ('ee_to_obj', 3), ('grasped', 1), ('phase', 1),
    ('base_pose', 3),
)


def resolve_config_path(filename: str):
    """Find a config file in the source tree or installed share dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(os.path.dirname(here), 'config', filename)]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(
            get_package_share_directory('pickplace_rl_mobile'), 'config', filename))
    except Exception:
        pass
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_yaml_hparams() -> dict:
    path = resolve_config_path('algo_hparams.yaml')
    if not path:
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Warning: could not load {path} ({exc}); using built-in hyperparameter defaults")
        return {}


def hparams_for(algo: str, overrides: dict = None) -> dict:
    if algo not in _DEFAULT_HPARAMS:
        raise ValueError(f"Unknown algo '{algo}', expected one of {ALGOS}")
    hp = dict(_DEFAULT_HPARAMS[algo])
    hp.update(_load_yaml_hparams().get(algo, {}))
    if overrides:
        hp.update(overrides)
    return hp


def is_off_policy(algo: str) -> bool:
    return algo in _OFF_POLICY_ALGOS


def is_entropy_tunable(algo: str) -> bool:
    return algo in _ENTROPY_TUNABLE_ALGOS


def is_recurrent(algo: str) -> bool:
    return algo == 'ppo_lstm'


def _obs_groups_for_dim(obs_dim: int):
    if obs_dim == 46:
        return _OBS_GROUPS_FULL
    if obs_dim == 27:
        return _OBS_GROUPS_LEGACY27
    # Unknown/custom observation mode — fall back to per-dim tokens so the
    # extractor still runs, just without semantic grouping.
    return tuple((f'dim{i}', 1) for i in range(obs_dim))


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """Tokenizes the flat observation into named groups (joint angles, EE
    position, object position, ...) and runs a small self-attention encoder
    before mean-pooling — an alternative to SB3's default flat-MLP encoder
    that lets the policy learn cross-group relationships (e.g. ee_pos vs
    obj_pos) directly through attention instead of only via dense layers.
    """

    def __init__(self, observation_space, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = int(np.prod(observation_space.shape))
        groups = _obs_groups_for_dim(obs_dim)

        self._slices = []
        start = 0
        for _, size in groups:
            self._slices.append(slice(start, start + size))
            start += size

        self._token_proj = nn.ModuleList([nn.Linear(size, d_model) for _, size in groups])
        self._pos_embed = nn.Parameter(torch.zeros(1, len(groups), d_model))
        nn.init.normal_(self._pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self._out = nn.Sequential(nn.Linear(d_model, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(
            [proj(observations[:, sl]) for proj, sl in zip(self._token_proj, self._slices)],
            dim=1,
        )
        tokens = tokens + self._pos_embed
        encoded = self._encoder(tokens)
        return self._out(encoded.mean(dim=1))


def policy_kwargs_for(policy_arch: str, net_arch=(512, 512, 512)) -> dict:
    if policy_arch == 'transformer':
        return dict(
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs=dict(d_model=64, n_heads=4, n_layers=2, features_dim=256),
            net_arch=[256, 256],
        )
    if policy_arch != 'mlp':
        raise ValueError(f"Unknown policy_arch '{policy_arch}', expected one of {POLICY_ARCHS}")
    return dict(net_arch=list(net_arch))


def policy_name_for(algo: str) -> str:
    return 'MlpLstmPolicy' if algo == 'ppo_lstm' else 'MlpPolicy'


def create_model(algo: str, env, policy_arch: str = 'mlp', tensorboard_log=None,
                  device='auto', verbose=1, hparam_overrides: dict = None):
    if algo not in _ALGO_CLASSES:
        raise ValueError(f"Unknown algo '{algo}', expected one of {ALGOS}")
    algo_cls = _ALGO_CLASSES[algo]
    return algo_cls(
        policy_name_for(algo),
        env,
        policy_kwargs=policy_kwargs_for(policy_arch),
        verbose=verbose,
        device=device,
        tensorboard_log=tensorboard_log,
        **hparams_for(algo, hparam_overrides),
    )


def load_model(algo: str, path: str, env, tensorboard_log=None, device='auto'):
    if algo not in _ALGO_CLASSES:
        raise ValueError(f"Unknown algo '{algo}', expected one of {ALGOS}")
    return _ALGO_CLASSES[algo].load(path, env=env, tensorboard_log=tensorboard_log, device=device)
