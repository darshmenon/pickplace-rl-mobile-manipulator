import math

import pytest

from pickplace_rl_mobile.pick_tree import STAGES, PickTree, arrival_check


def ok():
    return True, ''


def make(**over):
    stages = {s: ok for s in STAGES}
    stages.update(over)
    return PickTree(**stages)


def test_all_stages_pass():
    t = make()
    r = t.run()
    assert r.ok and r.completed == list(STAGES) and t.summary().startswith('1/1')


def test_stops_at_first_failure_and_skips_later_stages():
    called = []
    t = make(verify_arrival=lambda: (False, 'too far'),
             pick=lambda: called.append('pick') or (True, ''))
    r = t.run()
    assert not r.ok and r.failed_stage == 'verify_arrival' and r.reason == 'too far'
    assert r.completed == ['navigate'] and not called


def test_failures_are_counted_per_stage():
    outcomes = iter([False, True, False])
    t = make(verify_place=lambda: (next(outcomes), 'cube not at target'))
    for _ in range(3):
        t.run()
    assert t.failures['verify_place'] == 2 and t.successes == 1 and t.runs == 3


def test_a_crashing_stage_is_a_failure_not_an_exception():
    def boom():
        raise RuntimeError('nav2 down')
    r = make(navigate=boom).run()
    assert r.failed_stage == 'navigate' and 'nav2 down' in r.reason


def test_missing_stage_rejected():
    with pytest.raises(ValueError):
        PickTree(navigate=ok)


def test_arrival_tolerances():
    stage = arrival_check(lambda: (0.02, 0.0, 0.0), (0.0, 0.0), 0.0)
    assert stage()[0]
    assert not arrival_check(lambda: (0.2, 0.0, 0.0), (0.0, 0.0))()[0]
    # yaw wraps at +-pi: 0.1 rad apart across the seam is inside a 0.3 rad tolerance
    assert arrival_check(lambda: (0.0, 0.0, math.pi - 0.05), (0.0, 0.0), -math.pi + 0.05, yaw_tol=0.3)()[0]
    assert not arrival_check(lambda: (0.0, 0.0, math.pi - 0.1), (0.0, 0.0), -math.pi + 0.5, yaw_tol=0.3)()[0]
