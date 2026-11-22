"""HighwayEnv bridge. Only this module reads simulator vehicle objects.

The callback runs at physics ticks, so sensing is independent of policy timing.
This small _simulate override follows HighwayEnv 1.12.1's stepping loop.
"""
from dataclasses import dataclass
import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.kinematics import Vehicle


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

    def render(self):
        frame = super().render()
        # HighwayEnv disables display with SDL's dummy driver, even when the
        # viewer only draws to an offscreen surface for an RGB array.
        if self.render_mode == "rgb_array" and self.viewer.offscreen and not self.viewer.enabled:
            self.viewer.enabled = True
            frame = super().render()
        return frame

    def _create_vehicles(self):
        super()._create_vehicles()
        self.slow_leaders = []
        scenario = self.config.get("slow_lead_scenario")
        if scenario is None:
            return
        # Controlled traffic replaces random traffic in this opt-in experiment.
        self.road.vehicles = list(self.controlled_vehicles)
        ego = self.vehicle
        ego_lane = ego.lane_index[2]
        for row in scenario:
            lane_id = (ego_lane + row["lane_offset"]) % self.config["lanes_count"]
            lane = self.road.network.get_lane(("0", "1", lane_id))
            longitudinal, _ = lane.local_coordinates(ego.position)
            vehicle = Vehicle(self.road, lane.position(longitudinal + row["gap_m"], 0),
                              lane.heading_at(longitudinal), row["speed_m_s"])
            self.road.vehicles.append(vehicle)
            self.slow_leaders.append(vehicle)

    def slow_lead_diagnostics(self):
        """Privileged evaluation only; never part of the policy observation.

        Count leaders fully behind ego, regardless of lane. This is longitudinal
        clearance, not proof that the agent performed a deliberate lane change.
        """
        return {"slow_leaders_passed": sum(
            self.vehicle.position[0] - v.position[0] > (self.vehicle.LENGTH + v.LENGTH) / 2
            for v in self.slow_leaders), "slow_leaders_total": len(self.slow_leaders)}

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
