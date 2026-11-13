"""HighwayEnv bridge. Only this module reads simulator vehicle objects.

The callback runs at physics ticks, so sensing is independent of policy timing.
This small _simulate override follows HighwayEnv 1.12.1's stepping loop.
"""
from dataclasses import dataclass
import numpy as np
from highway_env.envs.highway_env import HighwayEnv


@dataclass(frozen=True)
class VehicleTruth:
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    length: float
    width: float


@dataclass(frozen=True)
class Scene:
    ego: VehicleTruth
    traffic: tuple[VehicleTruth, ...]


def snapshot(vehicle):
    return VehicleTruth(vehicle.position.copy(), vehicle.velocity.copy(),
                        float(vehicle.heading), float(vehicle.LENGTH), float(vehicle.WIDTH))


class HighwayAdapter(HighwayEnv):
    def __init__(self, config, render_mode=None):
        self.on_tick = None
        super().__init__(config=config, render_mode=render_mode)

    def scene(self):
        return Scene(snapshot(self.vehicle), tuple(snapshot(v) for v in self.road.vehicles
                                                   if v is not self.vehicle))

    def _simulate(self, action=None):
        frames = int(self.config["simulation_frequency"] / self.config["policy_frequency"])
        dt = 1 / self.config["simulation_frequency"]
        if action is not None:
            self.action_type.act(action)
        for frame in range(frames):
            self.road.act()
            self.road.step(dt)
            self.steps += 1
            if self.on_tick is not None:
                self.on_tick(self.steps * dt, dt, self.scene())
            if frame < frames - 1:
                self._automatic_rendering()
        self.enable_auto_render = False
