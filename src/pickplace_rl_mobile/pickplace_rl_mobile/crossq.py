#!/usr/bin/env python3
"""CrossQ: a SAC variant that drops the target critic network entirely,
relying on Batch Renormalization instead of Polyak-averaged targets to keep
the Bellman backup stable (Bhatt et al., "CrossQ: Batch Normalization in
Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity",
https://arxiv.org/abs/1902.05605 / ICLR 2024 revision).

Why this matters for this project: it's a second off-policy algorithm option
alongside TQC/SAC (see agent_factory.py) that reportedly matches or beats
SAC's sample efficiency at a fraction of the wall-clock cost per gradient
step (no target-network forward pass, no Polyak update) — worth an A/B via
optimize_rl.py against the existing TQC baseline before committing to it for
a full training run.

Mechanism (see CrossQ.train() below): instead of a separate critic_target
network updated by Polyak averaging, the current transition (s, a) and next
transition (s', a'=pi(s')) are concatenated along the batch dimension and
passed through the *same* critic in ONE forward call, so its BatchRenorm
layers see joint statistics across both halves. The "next" half of the
output is detached before building the Bellman target, so gradients only
flow into the critic through the "current" half — exactly as they would
with a target network, but without maintaining or syncing a second copy of
the weights. A plain critic_target is still constructed by SACPolicy._build
(inherited, unused) since SAC._setup_model() reads batch-norm buffers off
it; the wasted memory is negligible at this project's net sizes.

Only the critic gets BatchRenorm here — the actor stays a standard SAC MLP.
The paper applies it to both; the critic is where it does the load-bearing
work (letting the Bellman target use the online network safely), and this
project's env observations already go through VecNormalize, so the actor's
input-scale problem BatchRenorm would otherwise help with is largely already
handled.
"""

from typing import Any, Optional, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy


class BatchRenorm1d(nn.Module):
    """Batch Renormalization (Ioffe, 2017), the normalization layer CrossQ
    uses in place of plain BatchNorm1d. Plain BatchNorm's running estimates
    lag a slowly-shifting replay buffer under RL's non-i.i.d. data regime;
    Batch Renorm corrects each batch's normalization by an (r, d) factor
    tying it back toward those running estimates, with r/d clipped and
    ramped in linearly over `warmup_steps` so the layer behaves like
    ordinary BatchNorm early on (when the running estimates are still
    inaccurate) and only leans on the correction once they've caught up.
    """

    def __init__(self, num_features: int, eps: float = 1e-3, momentum: float = 0.01,
                 r_max: float = 3.0, d_max: float = 5.0, warmup_steps: int = 100_000):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.r_max = r_max
        self.d_max = d_max
        self.warmup_steps = warmup_steps
        self.weight = nn.Parameter(th.ones(num_features))
        self.bias = nn.Parameter(th.zeros(num_features))
        self.register_buffer('running_mean', th.zeros(num_features))
        self.register_buffer('running_var', th.ones(num_features))
        self.register_buffer('num_batches_tracked', th.tensor(0, dtype=th.long))

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.training:
            batch_mean = x.mean(0)
            batch_var = x.var(0, unbiased=False)
            batch_std = (batch_var + self.eps).sqrt()
            running_std = (self.running_var + self.eps).sqrt()

            progress = min(1.0, self.num_batches_tracked.item() / self.warmup_steps)
            r_max = 1.0 + progress * (self.r_max - 1.0)
            d_max = progress * self.d_max
            r = (batch_std.detach() / running_std).clamp(1.0 / r_max, r_max)
            d = ((batch_mean.detach() - self.running_mean) / running_std).clamp(-d_max, d_max)

            x_hat = (x - batch_mean) / batch_std * r + d

            with th.no_grad():
                self.running_mean += self.momentum * (batch_mean - self.running_mean)
                self.running_var += self.momentum * (batch_var - self.running_var)
                self.num_batches_tracked += 1
        else:
            x_hat = (x - self.running_mean) / (self.running_var + self.eps).sqrt()

        return x_hat * self.weight + self.bias


def _bn_mlp(input_dim: int, output_dim: int, net_arch: list, activation_fn: type) -> list:
    """Like SB3's create_mlp, but with a BatchRenorm1d ahead of the first
    layer and after every hidden Linear — the layout CrossQ's critic uses."""
    layers: list = [BatchRenorm1d(input_dim)]
    prev_dim = input_dim
    for size in net_arch:
        layers += [nn.Linear(prev_dim, size), BatchRenorm1d(size), activation_fn()]
        prev_dim = size
    layers.append(nn.Linear(prev_dim, output_dim))
    return layers


class CrossQContinuousCritic(ContinuousCritic):
    """SB3's ContinuousCritic with each Q-network rebuilt as a BatchRenorm
    MLP (see _bn_mlp). Kept as a subclass (rather than a from-scratch class)
    so isinstance checks and the inherited forward()/q1_forward() — which
    only assume self.q_networks is a list of callables over
    cat([features, actions]) — keep working unchanged."""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box,
                 net_arch: list, features_extractor: BaseFeaturesExtractor, features_dim: int,
                 activation_fn: type = nn.ReLU, normalize_images: bool = True,
                 n_critics: int = 2, share_features_extractor: bool = True):
        super().__init__(
            observation_space, action_space, net_arch, features_extractor, features_dim,
            activation_fn, normalize_images, n_critics, share_features_extractor,
        )
        action_dim = get_action_dim(self.action_space)
        input_dim = features_dim + action_dim
        # Replace the plain-MLP q_networks the super().__init__() above just
        # built with BatchRenorm ones; add_module overwrites the existing
        # "qfN" submodule in place.
        for idx in range(self.n_critics):
            q_net = nn.Sequential(*_bn_mlp(input_dim, 1, net_arch, activation_fn))
            self.add_module(f"qf{idx}", q_net)
            self.q_networks[idx] = q_net


class CrossQPolicy(SACPolicy):
    """SACPolicy with make_critic() swapped to build a CrossQContinuousCritic.
    The actor is unchanged from SACPolicy's default MLP actor."""

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CrossQContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CrossQContinuousCritic(**critic_kwargs).to(self.device)


class CrossQ(SAC):
    """SAC with the target critic replaced by CrossQ's joint-batch trick —
    see the module docstring and train() below. `tau` and
    `target_update_interval` are accepted (inherited from SAC's __init__)
    for CLI/API compatibility with the rest of agent_factory's algo family
    but are unused: there is no target network left to Polyak-update.
    """

    def __init__(self, policy: Union[str, type[CrossQPolicy]] = CrossQPolicy, *args, **kwargs):
        super().__init__(policy, *args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Next action from the current (not target) actor — no target
            # actor exists in CrossQ either.
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)

            # The joint-batch trick: one forward pass over current and next
            # transitions concatenated on the batch dim, so CrossQContinuousCritic's
            # BatchRenorm layers normalize both halves against shared statistics.
            batch_size_actual = replay_data.observations.shape[0]
            all_obs = th.cat([replay_data.observations, replay_data.next_observations], dim=0)
            all_actions = th.cat([replay_data.actions, next_actions], dim=0)
            all_q_values = self.critic(all_obs, all_actions)

            current_q_values = tuple(q[:batch_size_actual] for q in all_q_values)
            next_q_values = tuple(q[batch_size_actual:].detach() for q in all_q_values)

            next_q_values_cat = th.cat(next_q_values, dim=1)
            next_q_values_min, _ = th.min(next_q_values_cat, dim=1, keepdim=True)
            next_q_values_min = next_q_values_min - ent_coef * next_log_prob.reshape(-1, 1).detach()
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values_min

            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Actor loss: evaluate the just-updated critic in eval mode (frozen
            # BatchRenorm running stats) so this pass doesn't further shift
            # them — only the joint critic-update pass above should.
            self.critic.set_training_mode(False)
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            self.critic.set_training_mode(True)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # No target-network Polyak update — CrossQ has no target critic.

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
