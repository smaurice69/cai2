"""Vehicle controller with speed-aware steering suitable for an oval circuit."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Tuple

from .track import Point


@dataclass
class CarState:
    """Current kinematic state of the car."""

    position: Point
    heading: float  # Radians, 0 along +X axis
    speed: float  # Signed speed in metres per second

    def copy(self) -> "CarState":
        return replace(self)


class CarController:
    """Simple car controller with stable steering behaviour at all speeds."""

    def __init__(
        self,
        *,
        wheelbase: float = 2.85,
        max_speed: float = 90.0,
        max_reverse_speed: float = 35.0,
        max_acceleration: float = 8.0,
        max_brake: float = 12.0,
        drag_coefficient: float = 0.015,
        rolling_resistance: float = 0.35,
        max_steering_angle: float = math.radians(28.0),
        min_high_speed_factor: float = 0.32,
        steering_speed_sensitivity: float = 0.035,
    ) -> None:
        self.wheelbase = wheelbase
        self.max_speed = max_speed
        self.max_reverse_speed = max_reverse_speed
        self.max_acceleration = max_acceleration
        self.max_brake = max_brake
        self.drag_coefficient = drag_coefficient
        self.rolling_resistance = rolling_resistance
        self.max_steering_angle = max_steering_angle
        self.min_high_speed_factor = min_high_speed_factor
        self.steering_speed_sensitivity = steering_speed_sensitivity
        self.state = CarState(position=Point(0.0, 0.0), heading=0.0, speed=0.0)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def reset(self, *, position: Tuple[float, float] = (0.0, 0.0), heading: float = 0.0) -> None:
        """Reset the car state."""

        self.state = CarState(position=Point(*position), heading=heading, speed=0.0)

    def step(self, throttle: float, steering: float, dt: float) -> CarState:
        """Advance the simulation by ``dt`` seconds."""

        throttle = self._clamp(throttle, -1.0, 1.0)
        steering = self._clamp(steering, -1.0, 1.0)

        # Longitudinal dynamics
        acceleration = self._longitudinal_acceleration(throttle)
        speed = self._apply_longitudinal_physics(self.state.speed, acceleration, dt)

        # Lateral dynamics
        heading = self._apply_steering(self.state.heading, speed, steering, dt)

        # Integrate position using updated speed and heading
        dx = math.cos(heading) * speed * dt
        dy = math.sin(heading) * speed * dt
        position = Point(self.state.position.x + dx, self.state.position.y + dy)

        self.state = CarState(position=position, heading=_wrap_angle(heading), speed=speed)
        return self.state

    def _longitudinal_acceleration(self, throttle: float) -> float:
        if throttle >= 0.0:
            return throttle * self.max_acceleration

        if self.state.speed > 0.0:
            # Smoothly brake when still moving forward.
            return throttle * self.max_brake

        # Already stopped or reversing: allow throttle to build negative speed.
        return throttle * self.max_acceleration

    def _apply_longitudinal_physics(self, speed: float, acceleration: float, dt: float) -> float:
        speed += acceleration * dt

        # Apply drag and rolling resistance proportional to current speed.
        drag = self.drag_coefficient * speed * abs(speed)
        resistance = self.rolling_resistance * math.copysign(1.0, speed) if speed != 0 else 0.0
        speed -= (drag + resistance) * dt

        # Clamp forward/backward speeds separately.
        if speed > 0.0:
            speed = min(speed, self.max_speed)
        else:
            speed = max(speed, -self.max_reverse_speed)

        # Avoid oscillating around zero with extremely small residual velocities.
        if abs(speed) < 1e-3:
            speed = 0.0
        return speed

    def _apply_steering(self, heading: float, speed: float, steering: float, dt: float) -> float:
        if speed == 0.0 or steering == 0.0:
            return heading

        steer_angle = self._steering_angle(speed, steering)
        turning_radius = self.wheelbase / math.tan(steer_angle)
        yaw_rate = speed / turning_radius
        return heading + yaw_rate * dt

    def _steering_angle(self, speed: float, steering: float) -> float:
        abs_speed = abs(speed)
        speed_factor = 1.0 / (1.0 + abs_speed * self.steering_speed_sensitivity)
        factor = self.min_high_speed_factor + (1.0 - self.min_high_speed_factor) * speed_factor
        # Preserve steering authority even at high speed.
        factor = max(self.min_high_speed_factor, factor)
        return steering * self.max_steering_angle * factor


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi] for numerical stability."""

    while angle <= -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle

