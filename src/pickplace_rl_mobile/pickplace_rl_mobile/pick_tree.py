"""Navigate-then-pick behavior tree with explicit failure branches and per-stage failure counts.

    Sequence
      NavigateToPose  -> VerifyArrival -> Pick (WBC or RL) -> VerifyPlace -> ReturnToStart

Each stage is a callable returning (ok, reason). The tree stops at the first failure and records which
stage failed and why, so a batch of runs yields "failure per stage" instead of one success number.
No ROS imports: the Nav2 client, the WBC node and the checks in grasp_check.py are injected as stages,
which keeps the tree unit-testable and lets a sim run plug the real ones in.
"""

import collections
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

STAGES = ('navigate', 'verify_arrival', 'pick', 'verify_place', 'return')
Stage = Callable[[], Tuple[bool, str]]


@dataclass
class Result:
    ok: bool
    failed_stage: Optional[str] = None
    reason: str = ''
    completed: list = field(default_factory=list)


def arrival_check(get_pose, goal_xy, goal_yaw=None, xy_tol=0.05, yaw_tol=0.1):
    """Builds a VerifyArrival stage: base within tolerance of the pick pose, measured not assumed."""
    import math

    def stage():
        x, y, yaw = get_pose()
        err = math.hypot(x - goal_xy[0], y - goal_xy[1])
        if err > xy_tol:
            return False, f'arrival off by {err:.3f} m (tolerance {xy_tol} m)'
        if goal_yaw is not None:
            dyaw = abs((yaw - goal_yaw + math.pi) % (2 * math.pi) - math.pi)
            if dyaw > yaw_tol:
                return False, f'arrival yaw off by {dyaw:.3f} rad (tolerance {yaw_tol} rad)'
        return True, ''
    return stage


class PickTree:
    def __init__(self, **stages: Stage):
        missing = [s for s in STAGES if s not in stages]
        if missing:
            raise ValueError(f'missing stages: {missing}')
        self.stages = {s: stages[s] for s in STAGES}
        self.failures = collections.Counter()
        self.runs = 0
        self.successes = 0

    def run(self) -> Result:
        self.runs += 1
        done = []
        for name in STAGES:
            try:
                ok, reason = self.stages[name]()
            except Exception as e:  # a crashing stage is a failure of that stage, not of the tree
                ok, reason = False, f'{type(e).__name__}: {e}'
            if not ok:
                self.failures[name] += 1
                return Result(False, name, reason, done)
            done.append(name)
        self.successes += 1
        return Result(True, completed=done)

    def summary(self) -> str:
        lines = [f'{self.successes}/{self.runs} end-to-end successes']
        lines += [f'  failed at {s}: {self.failures[s]}' for s in STAGES if self.failures[s]]
        return '\n'.join(lines)
