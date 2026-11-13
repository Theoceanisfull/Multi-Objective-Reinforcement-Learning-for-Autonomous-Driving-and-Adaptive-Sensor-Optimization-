"""Finite-route reward ordering, termination, telemetry, and compatibility."""
import unittest
import numpy as np
from environments.sensor_highway import ExperimentConfig, SensorHighwayEnv


class RouteRewardTests(unittest.TestCase):
    def config(self):
        c = ExperimentConfig()
        c.highway.update(vehicles_count=0, duration=5, high_speed_reward=0)
        c.reward.route_distance_m = 60
        c.reward.energy_weight = c.reward.uncertainty_weight = 0
        return c

    def trip(self, speed):
        c = self.config()
        with SensorHighwayEnv(c) as env:
            obs, _ = env.reset(seed=12)
            np.testing.assert_array_equal(obs['task'], [1, 1])
            env.backend.vehicle.speed = speed
            env.backend.vehicle.target_speed = speed
            total = 0
            while True:
                obs, reward, end, cutoff, info = env.step({'drive': 1, 'sensors': np.zeros(18, np.int8)})
                self.assertAlmostEqual(reward, sum(info['reward_components'].values()))
                total += reward
                if end or cutoff:
                    self.assertTrue(end)
                    self.assertTrue(info['route_completed'])
                    self.assertEqual(obs['task'][0], 0)
                    self.assertTrue(env.observation_space.contains(obs))
                    return total, info

    def test_faster_safe_completion_earns_more(self):
        fast, f = self.trip(30)
        slow, s = self.trip(20)
        self.assertLess(f['time_s'], s['time_s'])
        self.assertGreater(fast, slow)

    def test_deadline_is_failure_and_crash_overrides_finish(self):
        for crash in [False, True]:
            c = self.config()
            c.highway['duration'] = .5
            c.reward.route_distance_m = 1 if crash else 1000
            with SensorHighwayEnv(c) as env:
                env.reset(seed=12)
                if crash:
                    env.backend.vehicle.crashed = True
                _, _, end, cutoff, info = env.step({'drive': 1, 'sensors': np.zeros(18, np.int8)})
                self.assertTrue(end or cutoff)
                self.assertFalse(info['route_completed'])
                self.assertEqual(info['reward_components']['completion_bonus'], 0)
                self.assertEqual(info['reward_components']['early_completion_bonus'], 0)
                self.assertEqual(info['reward_components']['failure_penalty'], -10)

    def test_overshoot_progress_is_capped(self):
        c = self.config()
        c.reward.route_distance_m = 1
        with SensorHighwayEnv(c) as env:
            env.reset(seed=12)
            _, _, end, _, info = env.step({'drive': 1, 'sensors': np.zeros(18, np.int8)})
            self.assertTrue(end)
            self.assertAlmostEqual(info['reward_components']['progress'], .02)

    def test_invalid_route_settings(self):
        for field in ['route_distance_m', 'time_weight', 'failure_penalty']:
            c = self.config()
            setattr(c.reward, field, -1)
            with self.assertRaises(ValueError):
                c.validate()
