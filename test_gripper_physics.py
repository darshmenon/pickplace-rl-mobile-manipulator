#!/usr/bin/env python3

import argparse
import importlib.util
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_SRC = os.path.join(REPO_ROOT, 'src', 'pickplace_rl_mobile')
if PYTHON_SRC not in sys.path:
    sys.path.insert(0, PYTHON_SRC)

try:
    from pickplace_rl_mobile.pickplace_env import PickPlaceEnv, _ARM_MOUNT_XYZ, ur3_fk
except ModuleNotFoundError:
    env_path = os.path.join(PYTHON_SRC, 'pickplace_rl_mobile', 'pickplace_env.py')
    spec = importlib.util.spec_from_file_location('pickplace_env', env_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    PickPlaceEnv = module.PickPlaceEnv
    _ARM_MOUNT_XYZ = module._ARM_MOUNT_XYZ
    ur3_fk = module.ur3_fk


def ee_local_from_joints(joint_angles: np.ndarray) -> np.ndarray:
    ee_in_arm_base = ur3_fk(joint_angles)
    ee_flipped = np.array([-ee_in_arm_base[0], -ee_in_arm_base[1], ee_in_arm_base[2]])
    return _ARM_MOUNT_XYZ + ee_flipped


def numerical_jacobian(joint_angles: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    base = ee_local_from_joints(joint_angles)
    jac = np.zeros((3, 6), dtype=np.float64)
    for idx in range(6):
        perturbed = joint_angles.copy()
        perturbed[idx] += eps
        jac[:, idx] = (ee_local_from_joints(perturbed) - base) / eps
    return jac


def world_to_local_delta(env: PickPlaceEnv, world_delta: np.ndarray) -> np.ndarray:
    _, _, btheta = env.base_pose
    rot_inv = np.array([
        [np.cos(btheta), np.sin(btheta), 0.0],
        [-np.sin(btheta), np.cos(btheta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rot_inv @ world_delta


def command_gripper(env: PickPlaceEnv, close: bool, steps: int) -> None:
    action = np.zeros(9, dtype=np.float32)
    action[6] = 1.0 if close else -1.0
    env.wait(steps=steps, action=action)


def move_ee_to_target(
    env: PickPlaceEnv,
    target_world: np.ndarray,
    max_steps: int = 180,
    position_tol: float = 0.015,
    close_gripper: bool = False,
) -> tuple[bool, float]:
    last_err_norm = float('inf')
    for _ in range(max_steps):
        ee_world = env.get_global_ee_pos()
        err_world = target_world - ee_world
        last_err_norm = float(np.linalg.norm(err_world))
        if last_err_norm < position_tol:
            return True, last_err_norm

        err_local = world_to_local_delta(env, err_world)
        jac = numerical_jacobian(env.joint_positions[:6])
        damping = 1e-3
        dq = jac.T @ np.linalg.solve(
            jac @ jac.T + damping * np.eye(3),
            np.clip(err_local, -0.04, 0.04),
        )

        action = np.zeros(9, dtype=np.float32)
        action[:6] = np.clip(dq / 0.25, -1.0, 1.0)
        action[6] = 1.0 if close_gripper else -1.0
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return False, last_err_norm

    return False, last_err_norm


def format_snapshot(snapshot: dict) -> str:
    keys = [
        'finger_joint',
        'left_inner_knuckle_joint',
        'left_inner_finger_joint',
        'right_outer_knuckle_joint',
        'right_inner_knuckle_joint',
        'right_inner_finger_joint',
    ]
    parts = []
    for key in keys:
        value = snapshot.get(key, np.nan)
        parts.append(f'{key}={value:.4f}' if not np.isnan(value) else f'{key}=nan')
    error_keys = [k for k in snapshot if k.endswith('_tracking_error')]
    for key in sorted(error_keys):
        value = snapshot[key]
        parts.append(f'{key}={value:.4f}' if not np.isnan(value) else f'{key}=nan')
    return ', '.join(parts)


def run_test(lift_threshold: float = 0.08) -> int:
    print('Running deterministic gripper pickup validation...')
    env = PickPlaceEnv()

    try:
        _, _ = env.reset()
        env.wait(steps=30)
        if env.real_object_pos is None:
            print('FAIL: no live object pose received from Gazebo.')
            return 1

        object_start = env.real_object_pos.copy()
        ee_start = env.get_global_ee_pos()
        print(f'Initial EE pose: {np.round(ee_start, 4)}')
        print(f'Initial object pose: {np.round(object_start, 4)}')

        command_gripper(env, close=False, steps=20)

        approach_target = object_start.copy()
        approach_target[2] = max(object_start[2], 0.055)
        reached, approach_err = move_ee_to_target(
            env,
            approach_target,
            max_steps=220,
            position_tol=0.018,
            close_gripper=False,
        )
        ee_after_approach = env.get_global_ee_pos()
        print(f'Approach target: {np.round(approach_target, 4)}')
        print(f'Approach result: reached={reached}, err={approach_err:.4f}, ee={np.round(ee_after_approach, 4)}')

        before_close = env.get_gripper_joint_snapshot()
        command_gripper(env, close=True, steps=90)
        after_close = env.get_gripper_joint_snapshot()
        print('Gripper before close:', format_snapshot(before_close))
        print('Gripper after close :', format_snapshot(after_close))

        lift_target = env.get_global_ee_pos().copy()
        lift_target[2] += 0.12
        reached_lift, lift_err = move_ee_to_target(
            env,
            lift_target,
            max_steps=180,
            position_tol=0.02,
            close_gripper=True,
        )
        command_gripper(env, close=True, steps=20)

        ee_final = env.get_global_ee_pos()
        obj_final = env.real_object_pos.copy() if env.real_object_pos is not None else np.array([np.nan, np.nan, np.nan])
        object_z_gain = obj_final[2] - object_start[2]

        print(f'Lift target: {np.round(lift_target, 4)}')
        print(f'Lift result: reached={reached_lift}, err={lift_err:.4f}, ee={np.round(ee_final, 4)}')
        print(f'Final object pose: {np.round(obj_final, 4)}')
        print(f'Object z gain: {object_z_gain:.4f} m')
        print(f'Pass threshold: object z > {lift_threshold:.3f} m')

        success = bool(obj_final[2] > lift_threshold)
        if success:
            print('PASS: object lifted above threshold.')
            return 0

        print('FAIL: object did not lift above threshold.')
        return 1
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description='Deterministic gripper pickup test')
    parser.add_argument('--lift-threshold', type=float, default=0.08, help='Required final object z height')
    args = parser.parse_args()
    return run_test(lift_threshold=args.lift_threshold)


if __name__ == '__main__':
    sys.exit(main())
