#!/usr/bin/env python3
"""
MuJoCo-backed pick-and-place environment.

Reuses every bit of reward/observation/curriculum logic from PickPlaceEnvBase
(pickplace_env_base.py), same as the Gazebo backend (pickplace_env.py). This
class only supplies the MuJoCo-specific I/O.

The arm and gripper are the MuJoCo Menagerie-quality UR3 (mujoco_assets/ur3,
capsule collision, tuned PD actuators) and Robotiq 2F-85 (mujoco_assets/
robotiq_2f85, real mesh geometry, pad friction, tendon-coupled fingers),
composed at load time via MjSpec.attach() rather than hand-built. Grasping
relies on the gripper's own contact physics (pad friction + the driver/
follower 4-bar linkage) — no kinematic "pinning" trick.

Verified empirically: the UR3 model's joint-angle convention matches
PickPlaceEnvBase.ur3_fk()'s DH parameters once the arm is mounted with zero
extra rotation (the model's own geometry already bakes in the 180 deg yaw
that get_end_effector_pos() applies for the Gazebo URDF arm) — see
_build_model() below. This keeps get_end_effector_pos()/get_global_ee_pos()
and the Cartesian-assist Jacobian (both inherited, unmodified) consistent
with the real simulated arm.

Simplifications relative to the Gazebo backend:
  - The mobile base is 3 unconstrained world-frame DOFs (slide x, slide y,
    hinge yaw) driven by directly-set qvel, not real wheel/ground contact.
  - The object is a fixed-size box; per-episode shape/size/friction domain
    randomization is not applied (respawn just repositions it).
"""

import os

import numpy as np
import mujoco
import mujoco.viewer

from pickplace_rl_mobile.pickplace_env_base import (
    PickPlaceEnvBase, _ARM_MOUNT_XYZ, _BASE_SPAWN_Z, _PLATFORM_TOP,
)

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'mujoco_assets')

_ARM_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
_ARM_ACTUATOR_NAMES = [
    'shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3',
]
_GRIPPER_JOINT_NAME = 'gripper_right_driver_joint'  # 0=open .. 0.8=closed
_GRIPPER_ACTUATOR_NAME = 'gripper_fingers_actuator'  # ctrlrange 0-255
_GRIPPER_JOINT_MAX = 0.8

_OBJECT_HALF_SIZE = 0.0325  # 6.5cm cube, matches DomainRandomizer's default obj_size

# Real robot's Gazebo DiffDrive plugin limits (diff_drive_robot-main/urdf/
# gazebo_control.xacro): wheel_separation=0.35, wheel_radius=0.1,
# wheel_acceleration=1.0 rad/s^2. The MuJoCo base is still unconstrained
# world-frame DOFs (no wheel contact), but matching these speed/accel limits
# keeps commanded base motion physically comparable to the real robot instead
# of being instantaneous and unbounded.
_MAX_LINEAR_SPEED = 2.0  # m/s
_MAX_ANGULAR_SPEED = 3.14  # rad/s
_LINEAR_ACCEL = 0.1  # m/s^2, wheel_radius * wheel_acceleration
_ANGULAR_ACCEL = 2 * 0.1 * 1.0 / 0.35  # rad/s^2, 2*wheel_radius*wheel_acceleration/wheel_separation


_WHEEL_RADIUS = 0.075  # matches the Gazebo robot's wheels (visual only here)
_WHEEL_TRACK = 0.39    # distance between the left and right wheel pairs


def _build_scene_spec() -> mujoco.MjSpec:
    """Mobile base + ground + platform + object. The arm/gripper assembly is
    attached onto this scene's `arm_attach` site by _build_model()."""
    scene_xml = f'''
<mujoco model="pickplace_scene">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>

  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="3 3 0.1" rgba="0.5 0.5 0.55 1"/>

    <!-- Pickup platform: top surface at _PLATFORM_TOP -->
    <geom name="platform" type="box" pos="0.6 0.0 {_PLATFORM_TOP / 2:.4f}"
          size="0.25 0.25 {_PLATFORM_TOP / 2:.4f}" rgba="0.6 0.6 0.65 1"/>

    <!-- Target marker (visual only) -->
    <geom name="target_marker" type="cylinder" pos="0.6 0.5 0.001" size="0.08 0.001"
          rgba="0.1 0.9 0.2 0.5" contype="0" conaffinity="0"/>

    <!-- Pickup object: free body, position set programmatically at reset -->
    <body name="pickup_object" pos="0.6 0.0 0.1325">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box"
            size="{_OBJECT_HALF_SIZE} {_OBJECT_HALF_SIZE} {_OBJECT_HALF_SIZE}"
            rgba="0.8 0.15 0.15 1" mass="0.15" friction="1.5 0.005 0.0001"
            solimp="0.95 0.99 0.001" solref="0.004 1"/>
    </body>

    <!-- Mobile base: 3 unconstrained world-frame DOFs, no wheel/contact modeling -->
    <body name="chassis" pos="0 0 {_BASE_SPAWN_Z:.4f}">
      <joint name="base_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="base_y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" damping="0"/>
      <geom type="box" size="0.225 0.175 0.08" rgba="0.40 0.23 0.72 1"/>

      <!-- Visual-only wheels, same size/colour as the Gazebo diff-drive robot
           (r=0.075, w=0.04, black). No contacts and ~zero mass, so they do not
           change the dynamics; they are spun kinematically from the base
           velocity in _advance_physics(). -->
      <body name="wheel_left_rear" pos="-0.15 0.195 -0.005">
        <joint name="wheel_left_rear_joint" type="hinge" axis="0 1 0" damping="0"/>
        <geom type="cylinder" size="0.075 0.02" euler="90 0 0" mass="0.001"
              rgba="0 0 0 1" contype="0" conaffinity="0"/>
      </body>
      <body name="wheel_left_front" pos="0.15 0.195 -0.005">
        <joint name="wheel_left_front_joint" type="hinge" axis="0 1 0" damping="0"/>
        <geom type="cylinder" size="0.075 0.02" euler="90 0 0" mass="0.001"
              rgba="0 0 0 1" contype="0" conaffinity="0"/>
      </body>
      <body name="wheel_right_rear" pos="-0.15 -0.195 -0.005">
        <joint name="wheel_right_rear_joint" type="hinge" axis="0 1 0" damping="0"/>
        <geom type="cylinder" size="0.075 0.02" euler="90 0 0" mass="0.001"
              rgba="0 0 0 1" contype="0" conaffinity="0"/>
      </body>
      <body name="wheel_right_front" pos="0.15 -0.195 -0.005">
        <joint name="wheel_right_front_joint" type="hinge" axis="0 1 0" damping="0"/>
        <geom type="cylinder" size="0.075 0.02" euler="90 0 0" mass="0.001"
              rgba="0 0 0 1" contype="0" conaffinity="0"/>
      </body>

      <!-- Arm mount: no extra rotation here — the UR3 model's own geometry
           already embeds the 180 deg yaw that get_end_effector_pos() applies
           for the Gazebo URDF arm (verified empirically, see module docstring). -->
      <body name="arm_mount" pos="{_ARM_MOUNT_XYZ[0]:.4f} {_ARM_MOUNT_XYZ[1]:.4f} {_ARM_MOUNT_XYZ[2]:.4f}">
        <site name="arm_attach" pos="0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
'''
    return mujoco.MjSpec.from_string(scene_xml)


def _build_model() -> mujoco.MjModel:
    scene = _build_scene_spec()
    ur3 = mujoco.MjSpec.from_file(os.path.join(_ASSETS_DIR, 'ur3', 'ur3.xml'))
    gripper = mujoco.MjSpec.from_file(os.path.join(_ASSETS_DIR, 'robotiq_2f85', '2f85.xml'))
    ur3.attach(gripper, prefix='gripper_', site=ur3.site('attachment_site'))
    scene.attach(ur3, prefix='', site=scene.site('arm_attach'))
    return scene.compile()


class PickPlaceEnvMuJoCo(PickPlaceEnvBase):
    """MuJoCo-backed pick-and-place environment (see module docstring)."""

    def __init__(
        self,
        curriculum_stage=0,
        observation_mode='full',
        enable_domain_randomization=False,
        enable_assist=True,
        render=False,
        whole_body=False,
        ee_site='flange',
    ):
        if ee_site not in ('flange', 'pinch'):
            raise ValueError("ee_site must be 'flange' or 'pinch'")
        self.ee_site = ee_site
        self.model = _build_model()
        self.data = mujoco.MjData(self.model)
        # Scratch MjData for FK at arbitrary joint angles (assist Jacobian).
        self._fk_data = mujoco.MjData(self.model)
        self._pinch_site_id = self.model.site('gripper_pinch').id

        self._arm_qpos_adr = np.array(
            [self.model.joint(name).qposadr[0] for name in _ARM_JOINT_NAMES])
        self._arm_qvel_adr = np.array(
            [self.model.joint(name).dofadr[0] for name in _ARM_JOINT_NAMES])
        self._arm_act_adr = np.array(
            [self.model.actuator(name).id for name in _ARM_ACTUATOR_NAMES])

        self._gripper_qpos_adr = self.model.joint(_GRIPPER_JOINT_NAME).qposadr[0]
        self._gripper_act_adr = self.model.actuator(_GRIPPER_ACTUATOR_NAME).id

        self._base_x_qpos = self.model.joint('base_x').qposadr[0]
        self._base_y_qpos = self.model.joint('base_y').qposadr[0]
        self._base_yaw_qpos = self.model.joint('base_yaw').qposadr[0]
        self._base_x_qvel = self.model.joint('base_x').dofadr[0]
        self._base_y_qvel = self.model.joint('base_y').dofadr[0]
        self._base_yaw_qvel = self.model.joint('base_yaw').dofadr[0]

        self._left_wheel_dofs = [self.model.joint(f'wheel_left_{k}_joint').dofadr[0]
                                 for k in ('rear', 'front')]
        self._right_wheel_dofs = [self.model.joint(f'wheel_right_{k}_joint').dofadr[0]
                                  for k in ('rear', 'front')]

        self._object_body_id = self.model.body('pickup_object').id
        self._object_qpos_adr = self.model.joint('object_free').qposadr[0]
        self._object_qvel_adr = self.model.joint('object_free').dofadr[0]

        mujoco.mj_forward(self.model, self.data)

        self._cmd_linear = 0.0
        self._cmd_angular = 0.0
        self._target_linear = 0.0
        self._target_angular = 0.0

        self._viewer = None
        if render:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        super().__init__(
            curriculum_stage=curriculum_stage,
            observation_mode=observation_mode,
            enable_domain_randomization=enable_domain_randomization,
            enable_assist=enable_assist,
            whole_body=whole_body,
        )
        if ee_site == 'pinch':
            # the cube stops the fingers at ~0.22, so a real grasp never reaches
            # the base-class "closed" threshold of 0.7
            self._grip_closing, self._grip_closed, self._grip_hold_min = 0.08, 0.15, 0.10
            # full arm authority from the start so a demonstrator's joint
            # velocities map 1:1 onto the policy action in every phase
            self._phase_arm_gain = {1: 1.0, 2: 1.0}
        self._pull_state()

    # ------------------------------------------------------------------
    # Internal state sync
    # ------------------------------------------------------------------

    def _pull_state(self):
        d = self.data
        self.joint_positions[:6] = d.qpos[self._arm_qpos_adr]
        self.joint_positions[6] = d.qpos[self._gripper_qpos_adr]
        self.joint_velocities[:6] = d.qvel[self._arm_qvel_adr]
        bx = d.qpos[self._base_x_qpos]
        by = d.qpos[self._base_y_qpos]
        byaw = d.qpos[self._base_yaw_qpos]
        self.base_pose = np.array([bx, by, float(np.arctan2(np.sin(byaw), np.cos(byaw)))])
        self.real_object_pos = d.xpos[self._object_body_id].copy()
        self._joint_states_received = True

    # ------------------------------------------------------------------
    # End-effector point. The base class uses the UR3 flange (DH FK). With
    # ee_site='pinch' the controlled point is the Robotiq pinch site instead,
    # 15.6 cm further along the tool axis: that is where the cube has to be for
    # a physical grasp, so the reward/phase gates then agree with the contact
    # physics. 'flange' keeps the original behaviour for old checkpoints.
    # ------------------------------------------------------------------

    def _ee_world_from_joints(self, joint_angles):
        if self.ee_site == 'flange':
            return super()._ee_world_from_joints(joint_angles)
        fk = self._fk_data
        fk.qpos[:] = self.data.qpos
        fk.qpos[self._arm_qpos_adr] = np.asarray(joint_angles[:6], dtype=float)
        mujoco.mj_kinematics(self.model, fk)
        return fk.site_xpos[self._pinch_site_id].copy()

    def get_end_effector_pos(self):
        if self.ee_site == 'flange':
            return super().get_end_effector_pos()
        world = self._ee_world_from_joints(self.joint_positions[:6])
        bx, by, btheta = self.base_pose
        dx, dy = world[0] - bx, world[1] - by
        c, s = np.cos(btheta), np.sin(btheta)
        return np.array([c * dx + s * dy, -s * dx + c * dy, world[2] - _BASE_SPAWN_Z])

    # ------------------------------------------------------------------
    # Backend hooks (see PickPlaceEnvBase)
    # ------------------------------------------------------------------

    def _drive_arm(self, target_positions, dt):
        self.data.ctrl[self._arm_act_adr] = target_positions

    def _drive_gripper(self, target_open_frac):
        frac = float(np.clip(target_open_frac, 0.0, _GRIPPER_JOINT_MAX))
        self.data.ctrl[self._gripper_act_adr] = frac / _GRIPPER_JOINT_MAX * 255.0

    def _drive_base(self, linear, angular):
        # Speed-limited target; the actual ramp to this target (accel-limited
        # like the real robot's Gazebo DiffDrive plugin) happens once per
        # control step in _advance_physics(), which is where the real dt is
        # available.
        self._target_linear = float(np.clip(linear, -_MAX_LINEAR_SPEED, _MAX_LINEAR_SPEED))
        self._target_angular = float(np.clip(angular, -_MAX_ANGULAR_SPEED, _MAX_ANGULAR_SPEED))

    def _advance_physics(self, dt):
        self._cmd_linear += np.clip(
            self._target_linear - self._cmd_linear, -_LINEAR_ACCEL * dt, _LINEAR_ACCEL * dt)
        self._cmd_angular += np.clip(
            self._target_angular - self._cmd_angular, -_ANGULAR_ACCEL * dt, _ANGULAR_ACCEL * dt)
        theta = float(self.base_pose[2])
        self.data.qvel[self._base_x_qvel] = self._cmd_linear * np.cos(theta)
        self.data.qvel[self._base_y_qvel] = self._cmd_linear * np.sin(theta)
        self.data.qvel[self._base_yaw_qvel] = self._cmd_angular
        # spin the (visual) wheels as a diff drive would
        w_left = (self._cmd_linear - self._cmd_angular * _WHEEL_TRACK / 2) / _WHEEL_RADIUS
        w_right = (self._cmd_linear + self._cmd_angular * _WHEEL_TRACK / 2) / _WHEEL_RADIUS
        self.data.qvel[self._left_wheel_dofs] = w_left
        self.data.qvel[self._right_wheel_dofs] = w_right

        n_sub = max(1, int(round(dt / self.model.opt.timestep)))
        for _ in range(n_sub):
            mujoco.mj_step(self.model, self.data)
            if self._viewer is not None:
                self._viewer.sync()
        self._pull_state()

    def _refresh_state(self):
        self._pull_state()

    def _spawn_object(self, x, y, z):
        qadr = self._object_qpos_adr
        self.data.qpos[qadr:qadr + 3] = [x, y, z]
        self.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
        self.data.qvel[self._object_qvel_adr:self._object_qvel_adr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def _respawn_object_randomized(self, x, y, z):
        # Shape/size/friction randomization isn't modeled here (fixed-size
        # box) — just reposition, same as the no-randomizer fallback.
        self._spawn_object(x, y, z)
        return z

    def _randomize_gravity(self):
        pass

    def _wait_until_ready(self):
        self._cmd_linear = 0.0
        self._cmd_angular = 0.0
        self._target_linear = 0.0
        self._target_angular = 0.0

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
