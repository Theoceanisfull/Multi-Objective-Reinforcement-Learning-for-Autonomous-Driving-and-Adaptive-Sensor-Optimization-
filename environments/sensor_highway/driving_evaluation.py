"""Frozen driving-policy transfer tasks; traffic truth is diagnostic only.

Preserves DrivingEnv's sensor pipeline, controller telemetry, five actions and
warm-up. Traffic is seeded independently of sensor/weather random streams.
"""
import copy

from highway_env.vehicle.behavior import IDMVehicle

from .driving import DrivingBackend, DrivingEnv
from .overtaking import OvertakingAdapter


class EvaluationBackend(OvertakingAdapter):
    def _create_vehicles(self):
        task = self.config['driving_evaluation']['traffic']
        if task == 'repeated':
            super()._create_vehicles()
            return
        if task == 'control':
            DrivingBackend._create_vehicles(self)
            self.lead.experiment_id = 0
        else:
            lane_id = int(self.np_random.integers(3)) if task == 'varied_lead' else 1
            lane_index = ('0', '1', lane_id)
            lane = self.road.network.get_lane(lane_index)
            ego = self.action_type.vehicle_class(self.road, lane.position(100, 0),
                                                 heading=0, speed=25)
            self.controlled_vehicles = [ego]
            self.road.vehicles = [ego]
            gap = self.np_random.uniform(55, 110) if task == 'varied_lead' else 75.
            speed = float(self.np_random.uniform(18, 22))
            self._add_vehicle(0, lane_id, 100 + gap, speed)
            if task in ('blocked_left', 'blocked_right'):
                side = 0 if task == 'blocked_left' else 2
                self._add_vehicle(1, side, 100 + self.np_random.uniform(-3, 3), 25.)
        self._spawned = set()
        self._slots = []
        self.encountered = set()
        self.passed = set()
        self.spawn_count = len(self.road.vehicles) - 1
        self.following_s = 0.
        self.lane_changes = 0
        self._last_lane = int(self.vehicle.lane_index[2])

    def _add_vehicle(self, uid, lane_id, x, speed):
        lane_index = ('0', '1', lane_id)
        lane = self.road.network.get_lane(lane_index)
        vehicle = IDMVehicle(self.road, lane.position(x, 0), heading=0, speed=speed,
                             target_lane_index=lane_index, target_speed=speed,
                             enable_lane_change=False)
        vehicle.experiment_id = uid
        self.road.vehicles.append(vehicle)

    def _maintain_traffic(self, initial=False):
        if self.config['driving_evaluation']['traffic'] == 'repeated':
            super()._maintain_traffic(initial)


class DrivingEvaluationEnv(DrivingEnv):
    def __init__(self, config, render_mode=None):
        super().__init__(config, render_mode)
        self.backend.close()
        self.backend = EvaluationBackend(copy.deepcopy(self.config.highway), render_mode)

    def _info(self, driving_reward, energy, uncertainty):
        # DrivingEnv._info assumes the original single-lead backend.
        from .env import SensorHighwayEnv
        info = SensorHighwayEnv._info(self, driving_reward, energy, uncertainty)
        b = self.backend
        info.update(overtakes=len(b.passed), lane_changes=b.lane_changes,
                    traffic_spawned=b.spawn_count, active_traffic=len(b.road.vehicles)-1,
                    following_s=b.following_s)
        return info
