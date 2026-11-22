"""Opt-in persistent slow traffic and privileged diagnostics for 9 km studies.

Legacy SensorHighwayEnv is unchanged. Only this backend/oracle reads truth;
learned policies receive the existing sensor observations.
"""
import copy
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.road.road import Road, RoadNetwork

from .backend import HighwayAdapter
from .env import SensorHighwayEnv


class OvertakingAdapter(HighwayAdapter):
    def _create_road(self):
        self.road = Road(network=RoadNetwork.straight_road_network(
            self.config['lanes_count'], length=40000, speed_limit=30),
            np_random=self.np_random, record_history=False)

    def _create_vehicles(self):
        cfg = self.config['overtaking']
        lane = self.road.network.get_lane(('0', '1', 1))
        ego = self.action_type.vehicle_class(self.road, lane.position(100, 0),
                                             heading=lane.heading_at(100), speed=25)
        self.controlled_vehicles = [ego]
        self.road.vehicles = [ego]
        self._slots = []
        self._spawned = set()
        self.encountered = set()
        self.passed = set()
        self.spawn_count = 0
        self.following_s = 0.
        self.lane_changes = 0
        self._last_lane = 1
        # The immutable slot schedule is drawn once; policy/sensor randomness
        # never changes it. Off-window traffic follows nominal straight motion.
        for lane_id in range(3):
            offset = {0:170., 1:90., 2:250.}[lane_id]
            for k in range(-8, 64):
                self._slots.append((lane_id*1000+k+8, lane_id,
                    100 + offset + cfg['spacing_m']*k + self.np_random.uniform(-12,12),
                    self.np_random.uniform(cfg['speed_min_m_s'],cfg['speed_max_m_s'])))
        self._maintain_traffic(initial=True)

    def _maintain_traffic(self, initial=False):
        cfg = self.config['overtaking']
        ego = self.vehicle
        now = self.steps / self.config['simulation_frequency']
        self.road.vehicles = [v for v in self.road.vehicles
                              if v is ego or v.position[0] >= ego.position[0]-cfg['remove_behind_m']]
        for uid, lane_id, origin, speed in self._slots:
            if uid in self._spawned:
                continue
            x = origin + speed*now
            gap = x-ego.position[0]
            if gap < -cfg['remove_behind_m']:
                self._spawned.add(uid)
                continue
            if not (-cfg['remove_behind_m'] <= gap <= cfg['spawn_ahead_m']):
                continue
            if not initial and gap < cfg['entry_min_gap_m']:
                continue
            lane_index = ('0','1',lane_id)
            if any(v.lane_index == lane_index and abs(v.position[0]-x)<55
                   for v in self.road.vehicles):
                continue
            lane = self.road.network.get_lane(lane_index)
            vehicle = IDMVehicle(self.road,lane.position(x,0),heading=0,speed=speed,
                                 target_lane_index=lane_index,target_speed=speed,enable_lane_change=False)
            vehicle.experiment_id = uid
            self.road.vehicles.append(vehicle)
            self._spawned.add(uid)
            self.spawn_count += 1

    def _diagnostic_tick(self, dt):
        ego = self.vehicle
        if ego.crashed:
            return
        lane_id = ego.lane_index[2]
        if lane_id != self._last_lane:
            self.lane_changes += 1
            self._last_lane = lane_id
        for v in self.road.vehicles:
            if v is ego:
                continue
            uid = v.experiment_id
            gap = v.position[0]-ego.position[0]
            if 10 < gap < 150:
                self.encountered.add(uid)
            if uid in self.encountered and gap < -10 and not v.crashed:
                self.passed.add(uid)
        front, _ = self.road.neighbour_vehicles(ego, ego.lane_index)
        if front is not None:
            gap = front.position[0]-ego.position[0]-(front.LENGTH+ego.LENGTH)/2
            if 0 < gap < 60 and ego.speed <= front.speed+2:
                self.following_s += dt

    def _simulate(self, action=None):
        # Maintain traffic once per decision; entry is outside sensor radius.
        self._maintain_traffic()
        callback = self.on_tick
        def tick(t,dt,scene):
            self._diagnostic_tick(dt)
            if callback is not None:
                callback(t,dt,scene)
        self.on_tick = tick
        try:
            super()._simulate(action)
        finally:
            self.on_tick = callback


class OvertakingSensorEnv(SensorHighwayEnv):
    def __init__(self, config, render_mode=None):
        validate_overtaking(config)
        super().__init__(config, render_mode)
        self.backend.close()
        self.backend = OvertakingAdapter(copy.deepcopy(self.config.highway), render_mode)

    def _info(self, driving_reward, energy, uncertainty):
        info = super()._info(driving_reward,energy,uncertainty)
        info.update(overtakes=len(self.backend.passed),
                    traffic_spawned=self.backend.spawn_count,
                    active_traffic=len(self.backend.road.vehicles)-1,
                    following_s=self.backend.following_s,
                    lane_changes=self.backend.lane_changes)
        return info


def validate_overtaking(config):
    config.validate()
    cfg = config.highway.get('overtaking',{})
    if config.highway['lanes_count'] != 3:
        raise ValueError('Overtaking scenario requires three lanes')
    for name in ['spacing_m','speed_min_m_s','speed_max_m_s','remove_behind_m',
                 'spawn_ahead_m','entry_min_gap_m']:
        if not np.isfinite(cfg.get(name,np.nan)) or cfg[name] <= 0:
            raise ValueError(f'Invalid overtaking setting: {name}')
    if not (cfg['speed_min_m_s']<=cfg['speed_max_m_s']<25
            and cfg['spacing_m']>=120
            and cfg['spawn_ahead_m']>cfg['entry_min_gap_m']>config.grid.radius_m):
        raise ValueError('Require slow traffic, safe spacing, and entry outside sensor range')


def oracle_action(env, allow_passing=True):
    """Truth-based feasibility check ONLY. Never used to train learned agents."""
    ego = env.backend.vehicle
    current = ego.target_lane_index[2]
    actual_lane = env.backend.road.network.get_lane(ego.target_lane_index)
    _, lateral = actual_lane.local_coordinates(ego.position)
    gap_by_lane, rear_by_lane, front_by_lane = {}, {}, {}
    for lane in range(3):
        front = [v for v in env.backend.road.vehicles if v is not ego
                 and v.lane_index[2]==lane and v.position[0]>ego.position[0]]
        back = [v for v in env.backend.road.vehicles if v is not ego
                and v.lane_index[2]==lane and v.position[0]<=ego.position[0]]
        nearest = min(front,key=lambda v:v.position[0],default=None)
        gap_by_lane[lane] = nearest.position[0]-ego.position[0]-5 if nearest else 1000.
        rear_by_lane[lane] = ego.position[0]-max((v.position[0] for v in back),default=-10000)-5
        front_by_lane[lane] = nearest
    name = 'IDLE'
    gap = gap_by_lane[current]
    if allow_passing and abs(lateral)<.3 and gap<150:
        candidates = [lane for lane in [current-1,current+1] if 0<=lane<3
                      and gap_by_lane[lane]>max(70,gap+5) and rear_by_lane[lane]>35]
        if candidates:
            best = max(candidates,key=lambda lane:gap_by_lane[lane])
            name = 'LANE_LEFT' if best<current else 'LANE_RIGHT'
    if name == 'IDLE':
        front = front_by_lane[current]
        desired = 30.
        if front is not None and gap<max(35,ego.speed*2):
            desired = max(10.,front.speed-2 if gap<30 else front.speed)
        speeds = np.asarray(ego.target_speeds)
        target = speeds[max(0,np.searchsorted(speeds,desired,side='right')-1)]
        if ego.target_speed > target+.1:
            name = 'SLOWER'
        elif ego.target_speed < target-.1:
            name = 'FASTER'
    index = next(k for k,v in env.backend.action_type.actions.items() if v==name)
    return {'drive':index,'sensors':np.ones(len(env.config.sensors),np.int8)}
