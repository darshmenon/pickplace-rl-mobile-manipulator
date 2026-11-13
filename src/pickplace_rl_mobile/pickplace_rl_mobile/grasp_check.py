"""Grasp and place verification from the real object pose, shared by the env, the WBC node and any BT node.

A "place complete" log line is not evidence: once the log said so while the cube had never moved. These
checks only trust the measured cube pose, never the controller's own phase counter.
"""

import numpy as np

LIFT_RISE = 0.04       # cube must rise this far above where it was grasped
HELD_XY = 0.03         # pads centred on the cube within this horizontal distance
HELD_Z = 0.025         # and level with it within this vertical distance
PLACE_XY = 0.10        # a placed cube lies within this of the target


def is_held(finger, cube_pos, pinch_pos, finger_min, finger_max, xy_tol=HELD_XY, z_tol=HELD_Z):
    """Fingers closed on something (not air) and the cube sits between the pads."""
    rel = np.asarray(cube_pos, dtype=float) - np.asarray(pinch_pos, dtype=float)
    return bool(finger_min < finger < finger_max
                and np.linalg.norm(rel[:2]) < xy_tol and abs(rel[2]) < z_tol)


def cube_lifted(cube_z, grasp_z, rise=LIFT_RISE):
    """True once the cube itself (not just the gripper) is `rise` above its grasp height."""
    return bool(cube_z >= grasp_z + rise)


def place_confirmed(cube_pos, target_pos, cube_start_pos, xy_tol=PLACE_XY, min_travel=0.10):
    """The cube is at the target AND really travelled there from where it started.

    The travel term rejects the incident where the cube was never lifted: it was already near the
    target (or never left its start), and the controller still reported a finished place.
    """
    cube, target, start = (np.asarray(v, dtype=float) for v in (cube_pos, target_pos, cube_start_pos))
    at_target = np.linalg.norm(cube[:2] - target[:2]) < xy_tol
    travelled = np.linalg.norm(cube[:2] - start[:2]) >= min_travel
    return bool(at_target and travelled)


class GraspTracker:
    """Per-episode record of the cube pose so the checks above need no controller state."""

    def __init__(self):
        self.start_pos = None
        self.grasp_z = None

    def reset(self, cube_pos):
        self.start_pos = np.asarray(cube_pos, dtype=float).copy()
        self.grasp_z = None

    def mark_grasp(self, cube_pos):
        self.grasp_z = float(cube_pos[2])

    def lifted(self, cube_pos):
        return self.grasp_z is not None and cube_lifted(cube_pos[2], self.grasp_z)

    def placed(self, cube_pos, target_pos):
        return self.start_pos is not None and place_confirmed(cube_pos, target_pos, self.start_pos)
