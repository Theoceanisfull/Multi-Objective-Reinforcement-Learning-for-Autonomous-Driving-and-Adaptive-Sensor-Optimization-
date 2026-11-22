"""Mixed 2 km driving tasks with seeded traffic; no traffic truth in observations."""
import copy
import numpy as np

from .driving_evaluation import DrivingEvaluationEnv, EvaluationBackend


SCENARIOS = {
    'varied_lead': ('varied_lead', 420., 18., 22.),
    'blocked_left': ('blocked_left', 420., 18., 22.),
    'blocked_right': ('blocked_right', 420., 18., 22.),
    'repeated_sparse': ('repeated', 420., 18., 22.),
    'repeated_medium': ('repeated', 320., 18., 22.),
    'repeated_dense': ('repeated', 260., 18., 22.),
    'repeated_mixed_speed': ('repeated', 320., 16., 23.),
}


def scenario_config(source, scenario, weather='clear', route_m=2000., deadline_s=90.):
    traffic, spacing, low, high = SCENARIOS[scenario]
    case=dict(name=scenario, traffic=traffic, route_m=route_m, deadline_s=deadline_s)
    config=copy.deepcopy(source)
    config.reward.route_distance_m=float(route_m)
    config.highway['duration']=float(deadline_s)
    config.highway['driving_evaluation']=case
    config.highway['overtaking']=dict(spacing_m=spacing,speed_min_m_s=low,speed_max_m_s=high,
        remove_behind_m=100.,spawn_ahead_m=600.,entry_min_gap_m=300.)
    config.weather.preset=weather
    config.validate()
    return config


def episode_scenario(episode, episodes, seed):
    """Policy-independent schedule; every shuffled block covers its stage mix."""
    first=int(.2*episodes);second=int(.5*episodes)
    if episode < first:
        stage=0;offset=episode
        mix=['varied_lead','blocked_left','blocked_right','repeated_sparse']
    elif episode < second:
        stage=1;offset=episode-first
        mix=['varied_lead','blocked_left','blocked_right','repeated_sparse',
             'repeated_medium','repeated_medium','repeated_mixed_speed']
    else:
        stage=2;offset=episode-second
        mix=['varied_lead','blocked_left','blocked_right','repeated_sparse',
             'repeated_medium','repeated_medium','repeated_dense','repeated_dense',
             'repeated_dense','repeated_mixed_speed','repeated_mixed_speed']
    rng=np.random.default_rng(np.random.SeedSequence([seed,stage,offset//len(mix)]))
    scenario=str(rng.permutation(mix)[offset%len(mix)])
    wrng=np.random.default_rng(np.random.SeedSequence([seed,episode,771]))
    weather='clear' if stage==0 else str(wrng.choice(['clear','rain','fog'],p=[.8,.1,.1]))
    return scenario,weather,stage


class CurriculumBackend(EvaluationBackend):
    def _create_vehicles(self):
        self._curriculum_lane=int(self.np_random.integers(3))
        super()._create_vehicles()
        self._last_lane=int(self.vehicle.lane_index[2])

    def _maintain_traffic(self, initial=False):
        if initial and self.config['driving_evaluation']['traffic']=='repeated':
            # Set the ego lane before initial traffic placement and sensor warm-up.
            index=('0','1',self._curriculum_lane)
            lane=self.road.network.get_lane(index)
            ego=self.vehicle
            ego.position=lane.position(100,0)
            ego.lane_index=index;ego.target_lane_index=index;ego.lane=lane
        super()._maintain_traffic(initial)


class CurriculumEnv(DrivingEvaluationEnv):
    def __init__(self,config,render_mode=None):
        super().__init__(config,render_mode)
        self.backend.close()
        self.backend=CurriculumBackend(copy.deepcopy(config.highway),render_mode)
