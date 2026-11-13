"""Velocity filter that sits between a policy and the robot (a simple barrier-style safety envelope).

Two rules, both applied to the commanded arm joint velocities qd:

1. Speed cap: while the base is moving, the end effector may not move faster than `ee_speed_cap`
   relative to the base, so the arm cannot whip while the chassis is turning or driving.
2. Keep-out boxes (for example the bin wall), in the base frame: when the end effector is within
   `margin` of a box and still moving into it, the velocity component along the box normal is removed,
   so it slides along the wall instead of through it. Same idea as a control barrier function h(x) >= 0
   with the constraint dh/dt >= -alpha * h, written for a signed distance h.

Pure numpy, so it unit-tests without ROS or MuJoCo. Forward kinematics is the UR3 DH chain that
safety_guard.py uses (base frame after the 180 degree yaw of base_link_inertia).
"""

from dataclasses import dataclass, field

import numpy as np

_DH = [
    (0.0, 0.1519, np.pi / 2),
    (-0.24365, 0.0, 0.0),
    (-0.21325, 0.0, 0.0),
    (0.0, 0.11235, np.pi / 2),
    (0.0, 0.08535, -np.pi / 2),
    (0.0, 0.0819, 0.0),
]
_ARM_MOUNT_Z = 0.10


def ee_fk(q):
    """End-effector position in the robot base frame for 6 arm joint angles."""
    T = np.eye(4)
    for (a, d, alpha), th in zip(_DH, q):
        ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(alpha), np.sin(alpha)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct],
                          [st, ct * ca, -ct * sa, a * st],
                          [0.0, sa, ca, d],
                          [0.0, 0.0, 0.0, 1.0]])
    p = T[:3, 3]
    return np.array([-p[0], -p[1], p[2] + _ARM_MOUNT_Z])


def ee_jacobian(q, eps=1e-6):
    """3x6 position Jacobian by central differences."""
    q = np.asarray(q, dtype=float)
    J = np.zeros((3, 6))
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        J[:, i] = (ee_fk(q + dq) - ee_fk(q - dq)) / (2 * eps)
    return J


@dataclass
class Box:
    """Axis-aligned keep-out box in the base frame (lo, hi corners)."""
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        self.lo, self.hi = np.asarray(self.lo, dtype=float), np.asarray(self.hi, dtype=float)

    def sdf_and_normal(self, p):
        """Signed distance (negative inside) and outward unit normal at p."""
        c, h = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
        local = p - c
        q = np.abs(local) - h
        outside = np.maximum(q, 0.0)
        dist = np.linalg.norm(outside) + min(np.max(q), 0.0)
        if np.linalg.norm(outside) > 1e-9:
            n = np.sign(local) * outside / np.linalg.norm(outside)
        else:  # inside: leave through the nearest face
            k = int(np.argmax(q))
            n = np.zeros(3)
            n[k] = 1.0 if local[k] >= 0 else -1.0
        return float(dist), n


@dataclass
class SafetyFilter:
    ee_speed_cap: float = 0.15       # m/s, only while the base moves
    base_moving_lin: float = 0.02    # m/s
    base_moving_ang: float = 0.05    # rad/s
    margin: float = 0.05             # m, start sliding this far from a keep-out box
    boxes: list = field(default_factory=list)
    stats: dict = field(default_factory=lambda: dict(calls=0, speed_clipped=0, wall_clipped=0))

    def filter(self, q, qd, base_lin=0.0, base_ang=0.0):
        """Returns the safe joint velocities (6,)."""
        qd = np.asarray(qd, dtype=float).copy()
        self.stats['calls'] += 1
        J = ee_jacobian(q)
        v = J @ qd
        base_moving = abs(base_lin) > self.base_moving_lin or abs(base_ang) > self.base_moving_ang
        speed = float(np.linalg.norm(v))
        if base_moving and speed > self.ee_speed_cap:
            qd *= self.ee_speed_cap / speed
            v = J @ qd
            self.stats['speed_clipped'] += 1
        p = ee_fk(q)
        v_safe = v.copy()
        for box in self.boxes:
            dist, n = box.sdf_and_normal(p)
            into = float(v_safe @ n)
            if dist < self.margin and into < 0.0:
                v_safe = v_safe - into * n
                self.stats['wall_clipped'] += 1
        if not np.allclose(v_safe, v):  # map the corrected Cartesian velocity back to joint space
            qd = qd + np.linalg.pinv(J) @ (v_safe - v)
        return qd
