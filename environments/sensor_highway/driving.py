"""Opt-in driving-only task: real sensor observations, one slower lead vehicle.

Navigation is assumed available through road localization. Other vehicles' true
positions/velocities are used for scenario generation and diagnostics only.
"""
import copy
import numpy as np
from gymnasium import spaces
from highway_env.vehicle.behavior import IDMVehicle

from .backend import HighwayAdapter
from .env import SensorHighwayEnv
from .config import ExperimentConfig


def driving_config():
    c = ExperimentConfig()
    c.highway.update(duration=45,lanes_count=3,vehicles_count=1,
                     high_speed_reward=0.,collision_reward=0.)
    c.highway['action']['target_speeds'] = [15,20,25,30]
    c.reward.route_distance_m = 1000.
    c.reward.progress_weight = .001
    c.reward.time_weight = .002
    c.reward.completion_bonus = 1.
    c.reward.early_completion_bonus = .1
    c.reward.failure_penalty = 2.
    c.reward.energy_weight = 0.
    c.reward.uncertainty_weight = 0.
    c.weather.preset = 'clear'
    c.validate()
    return c


class DrivingBackend(HighwayAdapter):
    def _create_vehicles(self):
        lane_index = ('0','1',1)
        lane = self.road.network.get_lane(lane_index)
        ego = self.action_type.vehicle_class(self.road,lane.position(100,0),heading=0,speed=25)
        self.controlled_vehicles=[ego]
        self.road.vehicles=[ego]
        speed=float(self.np_random.uniform(18,20))
        self.lead=IDMVehicle(self.road,lane.position(100+self.np_random.uniform(80,90),0),
                             heading=0,speed=speed,target_lane_index=lane_index,
                             target_speed=speed,enable_lane_change=False)
        self.road.vehicles.append(self.lead)
        self.overtaken=False
        self.lane_changes=0
        self._last_lane=1

    def _simulate(self,action=None):
        super()._simulate(action)
        if not self.vehicle.crashed and not self.lead.crashed:
            self.overtaken = self.overtaken or self.vehicle.position[0]>self.lead.position[0]+10
        lane=self.vehicle.lane_index[2]
        self.lane_changes+=int(lane!=self._last_lane)
        self._last_lane=lane


class DrivingEnv(SensorHighwayEnv):
    """Five driving actions; every sensor is requested on throughout the episode.

    Reset performs one forced 0.5 s IDLE warm-up with sensors on. Its reward,
    elapsed time, energy and distance are included in episode metrics, but it is
    not a learned transition. This avoids a blind first decision. Later sensor
    readings retain noise, latency and aging. No traffic truth enters observations.
    """
    def __init__(self,config=None,render_mode=None):
        super().__init__(config or driving_config(),render_mode)
        if self.config.highway['lanes_count']!=3:
            raise ValueError('Driving task requires three lanes')
        self.backend.close()
        self.backend=DrivingBackend(copy.deepcopy(self.config.highway),render_mode)
        self.action_space=spaces.Discrete(5)
        self.observation_space.spaces['controller']=spaces.Box(-1,1,(11,),np.float32)
        self.last_drive=1

    def _observation(self):
        obs,uncertainty=super()._observation()
        ego=self.backend.vehicle
        target_lane=int(ego.target_lane_index[2])
        # These are ego localization/controller measurements, not traffic truth.
        telemetry=[np.clip(ego.position[1]/8,-1,1),target_lane/2,ego.target_speed/30,
                   float(abs(ego.position[1]-4*target_lane)>.3),
                   float(target_lane>0),float(target_lane<2)]
        obs['controller']=np.asarray(telemetry+np.eye(5)[self.last_drive].tolist(),np.float32)
        return obs,uncertainty

    def _info(self,driving_reward,energy,uncertainty):
        info=super()._info(driving_reward,energy,uncertainty)
        info.update(overtakes=int(self.backend.overtaken),lane_changes=self.backend.lane_changes)
        return info

    def reset(self,*,seed=None,options=None):
        self.last_drive=1
        # The parent step validates joint actions; expose that space only while
        # delegating a sensor-on step. Externally this environment is Discrete(5).
        obs,info=super().reset(seed=seed,options=options)
        self.prelude_observation=obs
        self.prelude_info=info.copy()
        self.prelude_frame=self.render() if self.render_mode is not None else None
        obs,reward,terminated,truncated,info=self._joint_step(1)
        if terminated or truncated:
            raise ValueError('Task ends during the forced sensor warm-up')
        info['warmup_reward']=reward
        return obs,info

    def _joint_step(self,drive):
        joint=spaces.Dict({'drive':spaces.Discrete(5),'sensors':spaces.MultiBinary(len(self.config.sensors))})
        external=self.action_space
        self.action_space=joint
        try:
            return super().step({'drive':int(drive),'sensors':np.ones(len(self.config.sensors),np.int8)})
        finally:
            self.action_space=external

    def step(self,action):
        if not self.action_space.contains(action):raise ValueError('Expected driving action index 0..4')
        self.last_drive=int(action)
        return self._joint_step(action)
