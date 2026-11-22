"""Evaluation integrity: control parity, matched traffic, and repeated encounters."""
import unittest
import numpy as np

from algorithms.driving_agents import DrivingDQN
from environments.sensor_highway.config import ExperimentConfig
from environments.sensor_highway.driving import DrivingEnv
from environments.sensor_highway.driving_evaluation import DrivingEvaluationEnv
from experiments.driving_policy_helpers import rollout_driving
from experiments.evaluate_driving_generalization import CASES, DEFAULT_RUN, make_config, rollout


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.source = ExperimentConfig.load(DEFAULT_RUN/'config.json')

    def test_control_matches_original_sensor_observations_and_dynamics(self):
        config=make_config(self.source,CASES[0])
        with DrivingEnv(self.source) as a, DrivingEvaluationEnv(config) as b:
            oa,ia=a.reset(seed=300000);ob,ib=b.reset(seed=300000)
            for key in oa:np.testing.assert_array_equal(oa[key],ob[key])
            for action in [0,1,3,1,2,4]:
                oa,ra,ta,xa,ia=a.step(action)
                ob,rb,tb,xb,ib=b.step(action)
                for key in oa:np.testing.assert_array_equal(oa[key],ob[key])
                self.assertEqual((ra,ta,xa),(rb,tb,xb))
                self.assertEqual(ia['distance_m'],ib['distance_m'])

    def test_weather_does_not_change_initial_traffic(self):
        case=next(c for c in CASES if c['name']=='repeated_2250m')
        with DrivingEvaluationEnv(make_config(self.source,case,'clear')) as a, \
             DrivingEvaluationEnv(make_config(self.source,case,'fog')) as b:
            oa,_=a.reset(seed=300001);ob,_=b.reset(seed=300001)
            self.assertEqual(set(oa),{'grid','valid_cells','sensors','ego','task','controller'})
            for va,vb in zip(a.backend.road.vehicles,b.backend.road.vehicles):
                np.testing.assert_array_equal(va.position,vb.position)
                np.testing.assert_array_equal(va.velocity,vb.velocity)
            self.assertEqual(len(oa['controller']),11)

    def test_varied_lead_covers_starting_lanes_and_blocker_sides(self):
        case=next(c for c in CASES if c['traffic']=='varied_lead')
        lanes=set()
        with DrivingEvaluationEnv(make_config(self.source,case)) as env:
            for seed in range(300000,300012):
                env.reset(seed=seed)
                lanes.add(env.backend.vehicle.lane_index[2])
        self.assertEqual(lanes,{0,1,2})
        for name,side in [('blocked_left',0),('blocked_right',2)]:
            case=next(c for c in CASES if c['traffic']==name)
            with DrivingEvaluationEnv(make_config(self.source,case)) as env:
                env.reset(seed=300000)
                blocker=next(v for v in env.backend.road.vehicles if getattr(v,'experiment_id',None)==1)
                self.assertEqual(blocker.lane_index[2],side)
                self.assertLess(abs(blocker.position[0]-env.backend.vehicle.position[0]),5)

    def test_repeated_traffic_replenishes_without_close_spawning_and_counts_once(self):
        case=next(c for c in CASES if c['name']=='repeated_9000m')
        with DrivingEvaluationEnv(make_config(self.source,case)) as env:
            env.reset(seed=300001);b=env.backend
            ids={v.experiment_id for v in b.road.vehicles if v is not b.vehicle}
            b.vehicle.position[0]+=1500
            b._maintain_traffic()
            new=[v for v in b.road.vehicles if v is not b.vehicle and v.experiment_id not in ids]
            self.assertTrue(new)
            self.assertTrue(all(v.position[0]-b.vehicle.position[0]>=300 for v in new))
            car=new[0];b.vehicle.position[0]=car.position[0]-50;b._diagnostic_tick(.01)
            b.vehicle.position[0]=car.position[0]+20;b._diagnostic_tick(.01)
            count=len(b.passed);b._diagnostic_tick(.01)
            self.assertEqual(len(b.passed),count)
            self.assertIn(car.experiment_id,b.passed)

    def test_portable_checkpoint_frozen_and_control_metrics_match(self):
        agent=DrivingDQN.load(DEFAULT_RUN/'dqn/agent.pt')
        before=[p.detach().numpy().copy() for p in agent.online.parameters()]
        config=make_config(self.source,CASES[0])
        original,_=rollout_driving(self.source,agent,300000)
        actual,_=rollout(config,'dqn',agent,300000)
        for key in ['completed','crashed','elapsed_s','distance_m','overtakes']:
            self.assertEqual(original[key],actual[key])
        for a,b in zip(before,agent.online.parameters()):np.testing.assert_array_equal(a,b.detach().numpy())


if __name__=='__main__':unittest.main()
