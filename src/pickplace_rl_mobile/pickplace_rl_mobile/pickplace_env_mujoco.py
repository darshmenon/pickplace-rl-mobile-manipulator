#!/usr/bin/env python3
"""
MuJoCo-backed pick-and-place environment — a quick sanity-check backend.

Reuses every bit of reward/observation/curriculum logic from PickPlaceEnvBase
(pickplace_env_base.py), same as the Gazebo backend (pickplace_env.py). This
class only supplies the MuJoCo-specific I/O.

Simplifications made for speed (this is a sanity check, not a physically
faithful re-implementation of the Gazebo world):
  - The mobile base is 3 unconstrained world-frame DOFs (slide x, slide y,
    hinge yaw) driven by directly-set qvel, not real wheel/ground contact.
  - The gripper's opening fraction is tracked as a plain float with a simple
    first-order lag, mirrored onto two cosmetic finger joints. There is no
    real finger-object contact.
  - Grasping is a kinematic pin: once the gripper is closed near the object,
    the object's freejoint is pinned to follow the EE each substep (like a
    weld), rather than relying on friction/contact to hold it. This is what
    lets the shared grasp-verification logic in PickPlaceEnvBase (which reads
    real_object_pos height) actually see the object rise.
  - The object is a fixed-size box; per-episode shape/size/friction
    domain randomization is not applied (respawn just repositions it).
"""

import numpy as np
import mujoco
import mujoco.viewer

from pickplace_rl_mobile.pickplace_env_base import (
    PickPlaceEnvBase, _UR3_DH, _UR3_JOINT_LOW, _UR3_JOINT_HIGH,
    _ARM_MOUNT_XYZ, _BASE_SPAWN_Z, _PLATFORM_TOP,
)

_ARM_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

_OBJECT_HALF_SIZE = 0.0325  # 6.5cm cube, matches DomainRandomizer's default obj_size


def _quat_x(angle: float) -> str:
    """Quaternion (w x y z) for a rotation of `angle` radians about the local x axis."""
    return f"{np.cos(angle / 2):.8f} {np.sin(angle / 2):.8f} 0 0"


def _build_mjcf_xml() -> str:
    """Build the scene MJCF, walking the UR3 DH chain so MuJoCo's forward
    kinematics agrees exactly with PickPlaceEnvBase.ur3_fk() joint-for-joint.
    """
    arm_xml = []
    close_tags = []
    for i, (a, d, alpha) in enumerate(_UR3_DH):
        lo = float(_UR3_JOINT_LOW[i])
        hi = float(_UR3_JOINT_HIGH[i])
        # hinge_body_i: applies theta_i (the joint variable) about local z.
        # A capsule from the origin to (a, 0, d) stands in for the link.
        arm_xml.append(f'''
<body name="dh_hinge_{i}">
  <joint name="{_ARM_JOINT_NAMES[i]}" type="hinge" axis="0 0 1" pos="0 0 0"
         range="{lo:.6f} {hi:.6f}" damping="4.0" frictionloss="0.2"/>
  <geom type="capsule" fromto="0 0 0 {a:.6f} 0 {d:.6f}" size="0.035"
        rgba="0.85 0.55 0.1 1" contype="0" conaffinity="0"/>
  <body name="dh_offset_{i}" pos="{a:.6f} 0 {d:.6f}" quat="{_quat_x(alpha)}">''')
        close_tags.append('  </body>\n</body>')

    gripper_xml = '''
    <body name="gripper_base" pos="0 0 0">
      <geom type="box" size="0.03 0.05 0.03" rgba="0.2 0.2 0.2 1"
            contype="0" conaffinity="0"/>
      <body name="left_finger" pos="0 0.02 0.06">
        <joint name="left_finger_joint" type="slide" axis="0 1 0" range="-0.02 0.02" damping="2.0"/>
        <geom type="box" size="0.008 0.008 0.03" rgba="0.3 0.3 0.3 1"/>
      </body>
      <body name="right_finger" pos="0 -0.02 0.06">
        <joint name="right_finger_joint" type="slide" axis="0 1 0" range="-0.02 0.02" damping="2.0"/>
        <geom type="box" size="0.008 0.008 0.03" rgba="0.3 0.3 0.3 1"/>
      </body>
    </body>'''

    arm_chain = ''.join(arm_xml) + gripper_xml + '\n' + '\n'.join(reversed(close_tags))

    xml = f'''
<mujoco model="pickplace_mobile_ur3">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <geom contype="1" conaffinity="1" friction="1.0 0.05 0.01"/>
  </default>

  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="3 3 0.1" rgba="0.5 0.5 0.55 1"/>

    <!-- Pickup platform: top surface at _PLATFORM_TOP -->
    <geom name="platform" type="box" pos="0.6 0.0 {_PLATFORM_TOP / 2:.4f}"
          size="0.25 0.25 {_PLATFORM_TOP / 2:.4f}" rgba="0.6 0.6 0.65 1"/>

    <!-- Target marker (visual only) -->
    <geom name="target_marker" type="cylinder" pos="0.6 0.5 0.001" size="0.08 0.001"
          rgba="0.1 0.9 0.2 0.5" contype="0" conaffinity="0"/>

    <!-- Pickup object: free body, position/size set programmatically at reset -->
    <body name="pickup_object" pos="0.6 0.0 0.1325">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="{_OBJECT_HALF_SIZE} {_OBJECT_HALF_SIZE} {_OBJECT_HALF_SIZE}"
            rgba="0.8 0.15 0.15 1" mass="0.15"/>
    </body>

    <!-- Mobile base: 3 unconstrained world-frame DOFs, no wheel/contact modeling -->
    <body name="chassis" pos="0 0 {_BASE_SPAWN_Z:.4f}">
      <joint name="base_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="base_y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" damping="0"/>
      <geom type="box" size="0.225 0.175 0.08" rgba="0.2 0.4 0.8 1"/>

      <!-- Arm mount: offset + 180 deg yaw flip, matching
           PickPlaceEnvBase.get_end_effector_pos()'s x,y flip. -->
      <body name="arm_mount" pos="{_ARM_MOUNT_XYZ[0]:.4f} {_ARM_MOUNT_XYZ[1]:.4f} {_ARM_MOUNT_XYZ[2]:.4f}"
            euler="0 0 3.14159265">
        {arm_chain}
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="act_shoulder_pan" joint="shoulder_pan_joint" kp="200" ctrlrange="-6.28 6.28"/>
    <position name="act_shoulder_lift" joint="shoulder_lift_joint" kp="200" ctrlrange="-6.28 6.28"/>
    <position name="act_elbow" joint="elbow_joint" kp="200" ctrlrange="-3.15 3.15"/>
    <position name="act_wrist_1" joint="wrist_1_joint" kp="150" ctrlrange="-6.28 6.28"/>
    <position name="act_wrist_2" joint="wrist_2_joint" kp="150" ctrlrange="-6.28 6.28"/>
    <position name="act_wrist_3" joint="wrist_3_joint" kp="150" ctrlrange="-6.28 6.28"/>
    <position name="act_left_finger" joint="left_finger_joint" kp="80" ctrlrange="-0.02 0.02"/>
    <position name="act_right_finger" joint="right_finger_joint" kp="80" ctrlrange="-0.02 0.02"/>
  </actuator>
</mujoco>
'''
    return xml


class PickPlaceEnvMuJoCo(PickPlaceEnvBase):
    """MuJoCo-backed pick-and-place environment (see module docstring for
    the simplifications made relative to the Gazebo backend)."""

    def __init__(
        self,
        curriculum_stage=0,
        observation_mode='full',
        enable_domain_randomization=False,
        enable_assist=True,
        render=False,
    ):
        self.model = mujoco.MjModel.from_xml_string(_build_mjcf_xml())
        self.data = mujoco.MjData(self.model)

        self._arm_qpos_adr = np.array(
            [self.model.joint(name).qposadr[0] for name in _ARM_JOINT_NAMES])
        self._arm_qvel_adr = np.array(
            [self.model.joint(name).dofadr[0] for name in _ARM_JOINT_NAMES])
        self._arm_act_adr = np.array([
            self.model.actuator(name).id for name in
            ['act_shoulder_pan', 'act_shoulder_lift', 'act_elbow',
             'act_wrist_1', 'act_wrist_2', 'act_wrist_3']
        ])

        self._base_x_qpos = self.model.joint('base_x').qposadr[0]
        self._base_y_qpos = self.model.joint('base_y').qposadr[0]
        self._base_yaw_qpos = self.model.joint('base_yaw').qposadr[0]
        self._base_x_qvel = self.model.joint('base_x').dofadr[0]
        self._base_y_qvel = self.model.joint('base_y').dofadr[0]
        self._base_yaw_qvel = self.model.joint('base_yaw').dofadr[0]

        self._object_body_id = self.model.body('pickup_object').id
        self._object_qpos_adr = self.model.joint('object_free').qposadr[0]
        self._object_qvel_adr = self.model.joint('object_free').dofadr[0]

        self._left_finger_act = self.model.actuator('act_left_finger').id
        self._right_finger_act = self.model.actuator('act_right_finger').id

        self._gripper_frac = 0.0
        self._gripper_target = 0.0
        self._held = False

        mujoco.mj_forward(self.model, self.data)

        self._viewer = None
        if render:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        super().__init__(
            curriculum_stage=curriculum_stage,
            observation_mode=observation_mode,
            enable_domain_randomization=enable_domain_randomization,
            enable_assist=enable_assist,
        )
        self._pull_state()

    # ------------------------------------------------------------------
    # Internal state sync
    # ------------------------------------------------------------------

    def _pull_state(self):
        d = self.data
        self.joint_positions[:6] = d.qpos[self._arm_qpos_adr]
        self.joint_positions[6] = self._gripper_frac
        self.joint_velocities[:6] = d.qvel[self._arm_qvel_adr]
        bx = d.qpos[self._base_x_qpos]
        by = d.qpos[self._base_y_qpos]
        byaw = d.qpos[self._base_yaw_qpos]
        self.base_pose = np.array([bx, by, float(np.arctan2(np.sin(byaw), np.cos(byaw)))])
        self.real_object_pos = d.xpos[self._object_body_id].copy()
        self._joint_states_received = True

    # ------------------------------------------------------------------
    # Backend hooks (see PickPlaceEnvBase)
    # ------------------------------------------------------------------

    def _drive_arm(self, target_positions, dt):
        self.data.ctrl[self._arm_act_adr] = target_positions

    def _drive_gripper(self, target_open_frac):
        self._gripper_target = float(target_open_frac)
        alpha = 0.3
        self._gripper_frac = (1 - alpha) * self._gripper_frac + alpha * self._gripper_target
        sep = 0.018 * (1.0 - self._gripper_frac / 0.8)
        self.data.ctrl[self._left_finger_act] = sep
        self.data.ctrl[self._right_finger_act] = -sep

    def _drive_base(self, linear, angular):
        theta = float(self.base_pose[2])
        self.data.qvel[self._base_x_qvel] = linear * np.cos(theta)
        self.data.qvel[self._base_y_qvel] = linear * np.sin(theta)
        self.data.qvel[self._base_yaw_qvel] = angular

    def _advance_physics(self, dt):
        n_sub = max(1, int(round(dt / self.model.opt.timestep)))
        for _ in range(n_sub):
            self._pull_state()
            ee = self.get_global_ee_pos()
            obj_pos = self.data.xpos[self._object_body_id]
            dist = float(np.linalg.norm(ee - obj_pos))
            if not self._held and self.joint_positions[6] > 0.7 and dist < 0.06:
                self._held = True
            elif self._held and self.joint_positions[6] < 0.3:
                self._held = False
            if self._held:
                target = ee.copy()
                target[2] -= 0.02
                qadr = self._object_qpos_adr
                self.data.qpos[qadr:qadr + 3] = target
                self.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
                self.data.qvel[self._object_qvel_adr:self._object_qvel_adr + 6] = 0
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
        self._held = False

    def _respawn_object_randomized(self, x, y, z):
        # Shape/size/friction randomization isn't modeled here (fixed-size
        # box) — just reposition, same as the no-randomizer fallback.
        self._spawn_object(x, y, z)
        return z

    def _randomize_gravity(self):
        pass

    def _wait_until_ready(self):
        pass

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
