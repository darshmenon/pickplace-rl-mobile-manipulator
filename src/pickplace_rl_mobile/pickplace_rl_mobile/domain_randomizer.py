#!/usr/bin/env python3
"""
Domain Randomization Utility for Sim-to-Real Transfer.

Provides randomization functions for the pick-and-place environment
to improve policy robustness and sim-to-real transferability:
- Object position randomization
- Object color/size variation
- Lighting perturbation
- Physics parameter noise
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RandomizationConfig:
    """Configuration for domain randomization parameters."""

    # Object position randomization
    obj_x_range: Tuple[float, float] = (0.4, 0.8)
    obj_y_range: Tuple[float, float] = (-0.3, 0.3)
    obj_z_base: float = 0.055

    # Object size randomization (scale factor)
    obj_size_range: Tuple[float, float] = (0.8, 1.3)  # 80%-130% of nominal
    nominal_obj_size: float = 0.04  # 4cm cube

    # Object color randomization (HSV shifts)
    hue_shift_range: Tuple[float, float] = (-15, 15)  # degrees around red
    saturation_range: Tuple[float, float] = (0.6, 1.0)
    brightness_range: Tuple[float, float] = (0.5, 1.0)

    # Target position randomization
    target_x_range: Tuple[float, float] = (0.4, 0.8)
    target_y_range: Tuple[float, float] = (0.3, 0.7)
    target_z: float = 0.1

    # Physics noise
    mass_noise_std: float = 0.01  # kg
    friction_range: Tuple[float, float] = (0.5, 1.5)

    # Observation noise
    joint_pos_noise_std: float = 0.01  # radians
    odom_pos_noise_std: float = 0.005  # meters
    odom_theta_noise_std: float = 0.01  # radians

    # Action noise (for robustness)
    action_noise_std: float = 0.02

    # Gravity noise
    gravity_noise_std: float = 0.02  # m/s^2

    # Enable flags
    randomize_object_pos: bool = True
    randomize_object_size: bool = False
    randomize_object_color: bool = False
    randomize_target_pos: bool = False
    randomize_physics: bool = False
    randomize_observations: bool = True
    randomize_actions: bool = False


class DomainRandomizer:
    """
    Domain randomization utility for the PickPlace environment.

    Usage:
        randomizer = DomainRandomizer(config)

        # During env.reset():
        obj_pos = randomizer.randomize_object_position()
        target_pos = randomizer.randomize_target_position()

        # During env.step():
        noisy_obs = randomizer.add_observation_noise(obs)
        noisy_action = randomizer.add_action_noise(action)
    """

    def __init__(self, config: RandomizationConfig = None):
        self.config = config or RandomizationConfig()
        self.rng = np.random.default_rng()
        self.current_episode_params = {}
        self._randomize_episode_params()

    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def _randomize_episode_params(self):
        """Randomize parameters that stay constant within one episode."""
        cfg = self.config

        # Object size for this episode
        if cfg.randomize_object_size:
            scale = self.rng.uniform(*cfg.obj_size_range)
        else:
            scale = 1.0

        # Object color for this episode
        if cfg.randomize_object_color:
            hue_shift = self.rng.uniform(*cfg.hue_shift_range)
            saturation = self.rng.uniform(*cfg.saturation_range)
            brightness = self.rng.uniform(*cfg.brightness_range)
        else:
            hue_shift, saturation, brightness = 0.0, 1.0, 1.0

        # Physics for this episode
        if cfg.randomize_physics:
            mass_noise = self.rng.normal(0, cfg.mass_noise_std)
            friction = self.rng.uniform(*cfg.friction_range)
            gravity_noise = self.rng.normal(0, cfg.gravity_noise_std)
        else:
            mass_noise, friction, gravity_noise = 0.0, 1.0, 0.0

        self.current_episode_params = {
            'obj_scale': scale,
            'obj_size': cfg.nominal_obj_size * scale,
            'hue_shift': hue_shift,
            'saturation': saturation,
            'brightness': brightness,
            'mass_noise': mass_noise,
            'friction': friction,
            'gravity_noise': gravity_noise,
        }

    def randomize_object_position(self) -> np.ndarray:
        """Generate a random object position within the configured range."""
        cfg = self.config
        if cfg.randomize_object_pos:
            x = self.rng.uniform(*cfg.obj_x_range)
            y = self.rng.uniform(*cfg.obj_y_range)
            z = cfg.obj_z_base
        else:
            x, y, z = 0.6, 0.0, cfg.obj_z_base
        return np.array([x, y, z])

    def randomize_target_position(self) -> np.ndarray:
        """Generate a random target position."""
        cfg = self.config
        if cfg.randomize_target_pos:
            x = self.rng.uniform(*cfg.target_x_range)
            y = self.rng.uniform(*cfg.target_y_range)
            z = cfg.target_z
        else:
            x, y, z = 0.6, 0.5, cfg.target_z
        return np.array([x, y, z])

    def add_observation_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add noise to observation vector for robustness training."""
        if not self.config.randomize_observations:
            return obs

        noisy_obs = obs.copy()
        cfg = self.config

        # Joint position block is stable across both full (46) and legacy27 layouts.
        noisy_obs[:6] += self.rng.normal(0, cfg.joint_pos_noise_std, 6)

        # EE position is always at indices 13:16 in the current observation layouts.
        noisy_obs[13:16] += self.rng.normal(0, cfg.odom_pos_noise_std, 3)

        # Base pose lives at 24:27 in legacy27 and 34:37 in full mode.
        if noisy_obs.shape[0] >= 37:
            base_slice = slice(34, 37)
        elif noisy_obs.shape[0] >= 27:
            base_slice = slice(24, 27)
        else:
            return noisy_obs

        noisy_obs[base_slice.start:base_slice.start + 2] += self.rng.normal(0, cfg.odom_pos_noise_std, 2)
        noisy_obs[base_slice.stop - 1] += self.rng.normal(0, cfg.odom_theta_noise_std)

        return noisy_obs

    def add_action_noise(self, action: np.ndarray) -> np.ndarray:
        """Add noise to actions for robustness."""
        if not self.config.randomize_actions:
            return action

        noisy_action = action + self.rng.normal(
            0, self.config.action_noise_std, action.shape)
        return np.clip(noisy_action, -1.0, 1.0)

    def reset_episode(self):
        """Call at the start of each episode to re-randomize per-episode params."""
        self._randomize_episode_params()

    def get_episode_params(self) -> dict:
        """Get current episode randomization parameters (for logging)."""
        return self.current_episode_params.copy()

    def get_object_color_rgba(self) -> Tuple[float, float, float, float]:
        """
        Get the randomized object color as RGBA values.
        Useful for dynamically changing object appearance in Gazebo.
        """
        params = self.current_episode_params
        # Base red: H=0, S=1, V=1 -> R=1, G=0, B=0
        # Apply hue shift, saturation, brightness
        h = (params['hue_shift'] % 360) / 360.0
        s = params['saturation']
        v = params['brightness']

        # HSV to RGB conversion
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c

        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return (r + m, g + m, b + m, 1.0)
