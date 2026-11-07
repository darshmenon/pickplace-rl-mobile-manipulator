import pytest

from pickplace_rl_mobile.drive_to import drive_speed


def test_stops_inside_tolerance():
    assert drive_speed(0.002, 0.3, 0.002, 0.02) == 0.0
    assert drive_speed(-0.01, 0.3, 0.002, 0.02) == 0.0  # overshoot never drives backwards


def test_cruise_speed_is_capped():
    assert drive_speed(0.8, 0.3, 0.002, 0.02) == 0.3


def test_creeps_near_target():
    # 1.5 * 5 mm is below the creep speed, so the creep speed wins
    assert drive_speed(0.005, 0.3, 0.002, 0.02) == 0.02


def test_proportional_in_between():
    assert drive_speed(0.1, 0.3, 0.002, 0.02) == pytest.approx(0.15)
