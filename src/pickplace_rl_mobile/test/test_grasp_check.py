from pickplace_rl_mobile.grasp_check import GraspTracker, cube_lifted, is_held, place_confirmed


def test_held_needs_closed_fingers_on_the_cube():
    assert is_held(0.2, [0, 0, 0.1], [0.01, 0, 0.1], 0.08, 0.15 + 0.1)
    assert not is_held(0.5, [0, 0, 0.1], [0, 0, 0.1], 0.08, 0.25)       # closed on air
    assert not is_held(0.2, [0, 0, 0.1], [0.1, 0, 0.1], 0.08, 0.25)     # cube off to the side


def test_lift_checks_the_cube_not_the_gripper():
    assert cube_lifted(0.30, 0.05)
    assert not cube_lifted(0.06, 0.05)  # gripper rose, cube stayed behind


def test_logged_place_complete_but_cube_never_moved():
    # the incident: controller says "place complete" while the cube is still at its start
    start = [0.4, 0.0, 0.05]
    assert not place_confirmed(start, [0.0, 0.5, 0.05], start)
    # cube starts next to the target and never moves: also not a place
    assert not place_confirmed([0.0, 0.5, 0.05], [0.0, 0.5, 0.05], [0.0, 0.52, 0.05])
    assert place_confirmed([0.0, 0.5, 0.05], [0.0, 0.5, 0.05], start)


def test_tracker_round_trip():
    t = GraspTracker()
    t.reset([0.4, 0.0, 0.05])
    assert not t.lifted([0.4, 0.0, 0.3])  # no grasp marked yet
    t.mark_grasp([0.4, 0.0, 0.05])
    assert t.lifted([0.4, 0.0, 0.3]) and not t.lifted([0.4, 0.0, 0.06])
    assert t.placed([0.0, 0.5, 0.05], [0.0, 0.5, 0.05])
