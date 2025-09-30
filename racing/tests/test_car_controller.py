import math

from racing.car_controller import CarController


def test_high_speed_steering_remains_effective():
    controller = CarController()
    controller.state.speed = 72.0
    controller.state.heading = 0.0
    before = controller.state.heading
    controller.step(0.0, 0.9, 0.5)
    after = controller.state.heading
    assert not math.isclose(before, after, abs_tol=1e-3)


def test_reverse_steering_changes_heading():
    controller = CarController()
    controller.state.speed = -8.0
    controller.state.heading = 0.0
    controller.step(0.0, 1.0, 0.5)
    assert not math.isclose(controller.state.heading, 0.0, abs_tol=1e-3)


def test_reverse_throttle_builds_negative_speed():
    controller = CarController()
    controller.state.speed = 0.0
    controller.step(-1.0, 0.0, 0.5)
    assert controller.state.speed < 0.0
