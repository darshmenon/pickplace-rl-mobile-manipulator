import numpy as np
import pytest

from pickplace_rl_mobile.safety_filter import Box, SafetyFilter, ee_fk, ee_jacobian

Q = np.array([0.0, -1.2, 1.5, -1.8, -1.57, 0.0])


def test_jacobian_matches_finite_motion():
    qd = np.array([0.1, -0.05, 0.08, 0.02, 0.0, 0.03])
    dt = 1e-4
    moved = (ee_fk(Q + qd * dt) - ee_fk(Q)) / dt
    assert moved == pytest.approx(ee_jacobian(Q) @ qd, abs=1e-4)


def test_speed_cap_only_applies_while_base_moves():
    f = SafetyFilter(ee_speed_cap=0.05)
    qd = np.full(6, 0.5)
    still = f.filter(Q, qd, 0.0, 0.0)
    assert still == pytest.approx(qd)
    moving = f.filter(Q, qd, 0.3, 0.0)
    assert np.linalg.norm(ee_jacobian(Q) @ moving) == pytest.approx(0.05, rel=1e-6)
    assert f.stats['speed_clipped'] == 1


def test_slides_along_a_wall_instead_of_through_it():
    p = ee_fk(Q)
    # wall: a slab whose near face is 1 cm in front of the end effector along +x
    wall = Box([p[0] + 0.01, p[1] - 1, p[2] - 1], [p[0] + 1, p[1] + 1, p[2] + 1])
    f = SafetyFilter(margin=0.05, boxes=[wall])
    J = ee_jacobian(Q)
    qd = np.linalg.pinv(J) @ np.array([0.2, 0.1, 0.0])   # heading into the wall and sideways
    safe = f.filter(Q, qd)
    v = J @ safe
    assert v[0] <= 1e-6                  # no velocity into the wall
    assert v[1] == pytest.approx(0.1, abs=1e-3)  # tangential motion kept
    assert f.stats['wall_clipped'] == 1


def test_moving_away_from_the_wall_is_untouched():
    p = ee_fk(Q)
    wall = Box([p[0] + 0.01, p[1] - 1, p[2] - 1], [p[0] + 1, p[1] + 1, p[2] + 1])
    f = SafetyFilter(boxes=[wall])
    qd = np.linalg.pinv(ee_jacobian(Q)) @ np.array([-0.2, 0.0, 0.0])
    assert f.filter(Q, qd) == pytest.approx(qd)
    assert f.stats['wall_clipped'] == 0


def test_box_sdf_signs():
    b = Box([0, 0, 0], [1, 1, 1])
    assert b.sdf_and_normal(np.array([2.0, 0.5, 0.5]))[0] == pytest.approx(1.0)
    d, n = b.sdf_and_normal(np.array([0.5, 0.5, 0.9]))
    assert d < 0 and n == pytest.approx([0, 0, 1])
