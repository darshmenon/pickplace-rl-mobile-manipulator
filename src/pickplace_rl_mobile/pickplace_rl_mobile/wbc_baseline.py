#!/usr/bin/env python3
"""
Whole-body controller (WBC) baseline for the MuJoCo mobile-manipulator env.

Resolved-rate control over the stacked 9-DoF body [6 arm joints, base linear
(along heading), base yaw]. The Jacobian comes straight from MuJoCo
(mj_jacSite), so the base and arm are solved together in one weighted
damped-least-squares problem instead of a locked-base arm IK plus a separate
base P-controller:

    min_qd  |J qd - e|^2 + qd^T W qd         W penalises base motion more than
                                              arm motion, so the base only
                                              moves when the arm runs out of
                                              reach
    qd += N * k (q_nom - q)                   N = null-space projector, pulls
                                              the arm toward a nominal posture

Task rows: EE position (3) plus tool-axis alignment (3, rank 2). A small
state machine over the env's own phase counter picks the target and gripper
command, and the result is mapped back into the env's normalised 9-D action
so the controller goes through exactly the same step() interface as a policy.

  python3 -m pickplace_rl_mobile.wbc_baseline --episodes 5 --ee-site pinch
"""

import argparse

import numpy as np
import mujoco

from pickplace_rl_mobile.pickplace_env_mujoco import PickPlaceEnvMuJoCo

# Position-actuator gains of the six arm joints (mujoco_assets/ur3), used for
# gravity feedforward: env.step() anchors each target to the *measured* joint
# angle, so without it gravity droop accumulates every control step.
_ARM_KP = np.array([1500.0, 1500.0, 1500.0, 500.0, 500.0, 500.0])
_ARM_MAX_VEL = 0.5    # rad/s, the largest joint speed env.step() can realise from an action
_GRAV_FF = 1.0        # gravity feedforward scale (1 = full)
_ACTION_TO_VEL = 2.5   # env.step(): joint vel = clip(2.5 * action, +-0.5) rad/s
_ARM_DT = 0.2          # env.step() integrates joint velocity over this long
_LIFT_Z = 0.25         # flange height the env's lift phase wants
_HOVER_HEIGHT = 0.06   # pinch height above the cube centre before the final descent

# Top-down reach posture (tool axis straight down, pinch ~0.45 m ahead of the
# chassis at cube height), found by collision-checked IK against the MuJoCo
# model. A horizontal side grasp collides with the platform: the arm base is
# only 0.18 m off the floor.
_Q_NOMINAL = np.array([-0.254, 0.526, -1.38, -0.716, -1.571, 0.0])
# Collision-free joint-space waypoint between the env's reset pose and the
# nominal posture (the straight joint-space line drags the gripper through the
# platform edge). Found by sampling around the line with a contact check.
_Q_WAYPOINT = np.array([-0.384, -0.079, -0.617, -0.026, -0.809, 0.039])
_RAISE_HEIGHT = 0.20   # free-air lift before the tool is aligned
_DOWN = np.array([0.0, 0.0, -1.0])
_GRASP_Z_OFFSET = 0.0  # pinch target height relative to the cube centre


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class WholeBodyController:
    def __init__(self, env: PickPlaceEnvMuJoCo, ee_site='pinch',
                 base_weight=2.0, arm_weight=0.02, damping=1e-3,
                 q_nom=None, use_external=False):
        # use_external=False sends everything through the normalised 9-D action,
        # so recorded demos are exactly reproducible by a policy
        self.env = env
        self.use_external = use_external
        self.model, self.data = env.model, env.data
        self.ee_site = ee_site
        self.site_id = self.model.site('gripper_pinch' if ee_site == 'pinch'
                                       else 'attachment_site').id
        self.pinch_id = self.model.site('gripper_pinch').id
        self.flange_id = self.model.site('attachment_site').id
        self.arm_dofs = np.asarray(env._arm_qvel_adr)
        self.base_dofs = np.array([env._base_x_qvel, env._base_y_qvel, env._base_yaw_qvel])
        # yaw is stiffer than forward motion: base yaw and shoulder pan are redundant
        # for the EE position and would otherwise drift against each other
        self.W = np.diag([arm_weight] * 6 + [base_weight, 5.0 * base_weight])
        self.damping = damping
        self.q_nom = _Q_NOMINAL if q_nom is None else np.asarray(q_nom)
        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros((3, self.model.nv))
        self.reset()

    def reset(self):
        self.stage = 'raise'
        self._raise_tgt = None
        self._descend = False
        self.hold = 0

    # ------------------------------------------------------------------
    def _jacobian(self, site_id):
        mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, site_id)
        theta = self.env.base_pose[2]
        c, s = np.cos(theta), np.sin(theta)

        def cols(j):
            arm = j[:, self.arm_dofs]
            v = c * j[:, self.base_dofs[0]] + s * j[:, self.base_dofs[1]]
            w = j[:, self.base_dofs[2]]
            return np.column_stack([arm, v, w])

        return cols(self._jp), cols(self._jr)

    def _solve(self, pos_err, axis_des, gain_p=3.0, gain_o=3.0, w_orient=0.8,
               posture_gain=0.2, base_locked=False):
        jp, jr = self._jacobian(self.site_id)
        axis = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 2]
        # d(axis)/dt = -[axis]x * omega ; drive axis toward axis_des
        ja = -_skew(axis) @ jr
        ja[:, 6:] = 0.0  # base is no use for tool tilt; keep it for position only
        va = gain_o * (axis_des - axis)
        J = np.vstack([jp, w_orient * ja])
        e = np.concatenate([gain_p * pos_err, w_orient * va])
        W = self.W.copy()
        if base_locked:
            W[6:, 6:] *= 1e4
        A = J.T @ J + W + self.damping * np.eye(8)
        qd = np.linalg.solve(A, J.T @ e)
        # null-space posture pull on the arm joints only
        N = np.eye(8) - np.linalg.solve(A, J.T @ J)
        q = self.env.joint_positions[:6]
        qd_post = np.zeros(8)
        qd_post[:6] = posture_gain * (self.q_nom - q)
        return qd + N @ qd_post

    # ------------------------------------------------------------------
    def act(self):
        env, d = self.env, self.data
        phase = env.current_phase
        obj = env.real_object_pos
        flange = d.site_xpos[self.flange_id]
        pinch = d.site_xpos[self.pinch_id]
        ee = d.site_xpos[self.site_id]
        bx, by, bth = env.base_pose
        heading = np.array([np.cos(bth), np.sin(bth), 0.0])

        grip = -1.0
        if self.stage == 'raise' and phase in (1, 2):
            # the reset pose rests the gripper on the platform edge: lift straight
            # up in free air first (arm only, no tool-orientation task)
            if self._raise_tgt is None:
                self._raise_tgt = ee + np.array([0.0, 0.0, _RAISE_HEIGHT])
            if ee[2] > self._raise_tgt[2] - 0.05:
                self.stage = 'hover'
            else:
                qd = self._solve(self._raise_tgt - ee, _DOWN, w_orient=0.0,
                                 posture_gain=0.0, base_locked=True)
                return self._to_action(qd, grip, phase)
        if phase in (1, 2):
            axis_des = _DOWN
            tgt = obj.copy()
            tgt[2] += _GRASP_Z_OFFSET
            # the env's grasp gate looks at the flange; the physical grasp needs
            # the pinch point on the cube. ee_site selects which one we servo.
            err_now = np.linalg.norm(tgt - ee)
            # keep the descent vertical: servo XY first, then Z
            if self.stage == 'advance':
                xy_err = np.linalg.norm((tgt - ee)[:2])
                # latch the descent once XY is within finger clearance (hysteresis)
                self._descend = xy_err < 0.012 or (self._descend and xy_err < 0.03)
                if not self._descend:
                    tgt[2] = ee[2]
            if self.stage == 'hover':
                tgt = tgt + np.array([0.0, 0.0, _HOVER_HEIGHT])
                if np.linalg.norm(tgt - ee) < 0.04:
                    self.stage = 'advance'
            if self.stage == 'advance':
                if err_now < 0.02 and self._descend:
                    self.hold += 1
                    grip = 1.0
                elif self.hold > 0:
                    grip = 1.0
            pos_err = tgt - ee
        elif phase == 3:
            grip = 1.0
            axis_des = _DOWN
            tgt = ee.copy()
            tgt[2] = _LIFT_Z + 0.02  # env lift gate is on the controlled point's height
            pos_err = tgt - ee
        elif phase == 4:
            grip = 1.0
            axis_des = _DOWN
            tgt = ee.copy()
            tgt[:2] = env.target_pos[:2]
            tgt[2] = _LIFT_Z
            pos_err = tgt - ee
        else:  # phase 5, place
            axis_des = _DOWN
            # servo the flange onto the target (env success test) when ee_site is
            # flange, otherwise put the pinch point there
            tgt = env.target_pos.astype(float).copy()
            pos_err = tgt - ee
            grip = 1.0 if np.linalg.norm(pos_err) > 0.05 else -1.0

        fine = self.stage == 'advance' and phase in (1, 2)
        qd = self._solve(np.clip(pos_err, -0.3, 0.3), axis_des,
                         gain_p=6.0 if fine else 3.0, posture_gain=0.0 if fine else 0.2)
        if phase in (4, 5):
            # carrying the cube the arm alone runs out of reach (and torque): turn
            # the base toward the goal and drive it in, the arm tracks on top
            to_t = env.target_pos[:2] - np.array([bx, by])
            dist = float(np.linalg.norm(to_t))
            yaw_err = float(np.arctan2(to_t[1], to_t[0]) - bth)
            yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
            qd[7] = np.clip(1.5 * yaw_err, -0.8, 0.8) if dist > 0.25 else 0.0
            qd[6] = np.clip(1.0 * (dist - 0.30), 0.0, 0.25) if abs(yaw_err) < 0.5 else 0.0
        return self._to_action(qd, grip, phase)

    def _to_action(self, qd, grip, phase):
        # joint velocity = task-space solution + gravity feedforward, capped at
        # what the env can realise; sent through the 9-D action (default) or the
        # env.external_joint_vels hook.
        arm = qd[:6] + _GRAV_FF * self.data.qfrc_bias[self.arm_dofs] / (_ARM_KP * _ARM_DT)
        peak = np.max(np.abs(arm))
        if peak > _ARM_MAX_VEL:
            arm = arm * (_ARM_MAX_VEL / peak)
        a = np.zeros(9, dtype=np.float32)
        if self.use_external:
            self.env.external_joint_vels = arm
        else:
            a[:6] = np.clip(arm / _ACTION_TO_VEL, -1, 1)
        a[6] = grip
        a[7] = np.clip(qd[6] / 0.5, -1, 1)
        a[8] = np.clip(qd[7] / 1.0, -1, 1)
        return a


def run_episode(env, ctrl, seed, max_steps=None):
    env.reset(seed=seed)
    ctrl.reset()
    path = [env.base_pose[:2].copy()]
    steps = 0
    info = {}
    done = False
    total = 0.0
    while not done:
        a = ctrl.act()
        _, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
        path.append(env.base_pose[:2].copy())
        done = term or trunc or (max_steps is not None and steps >= max_steps)
    travel = float(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1).sum())
    obj = env.real_object_pos
    return dict(success=bool(info.get('is_success')), max_phase=int(info.get('max_phase', 0)),
                verified=int(info.get('verified_grasps', 0)), steps=steps, reward=total,
                base_travel=travel,
                obj_target_err=float(np.linalg.norm(obj[:2] - env.target_pos[:2])),
                obj_z=float(obj[2]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--ee-site', choices=['pinch', 'flange'], default='pinch')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    env = PickPlaceEnvMuJoCo(enable_assist=False, whole_body=True, curriculum_stage=0,
                             ee_site=args.ee_site)
    ctrl = WholeBodyController(env, ee_site=args.ee_site)
    for i in range(args.episodes):
        print(run_episode(env, ctrl, args.seed + i))


if __name__ == '__main__':
    main()
