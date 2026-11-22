"""Controlled slow traffic must be reproducible and physically passable."""
import unittest
import numpy as np
from environments.sensor_highway import ExperimentConfig
from environments.sensor_highway.backend import HighwayAdapter


def scenario_config():
    config = ExperimentConfig()
    config.highway.update(vehicles_count=0, duration=120, slow_lead_scenario=[
        {"lane_offset": i, "gap_m": float(60 + i * 120), "speed_m_s": 12.0}
        for i in range(3)])
    return config


class SlowLeaderTests(unittest.TestCase):
    def test_layout_motion_and_seed(self):
        config = scenario_config()
        config.validate()
        env = HighwayAdapter(config.highway)
        try:
            env.reset(seed=100000)
            positions = [v.position.copy() for v in env.road.vehicles]
            ego = env.vehicle
            self.assertEqual(len(env.road.vehicles), 4)
            self.assertEqual(len({v.lane_index for v in env.slow_leaders}), 3)
            self.assertEqual(env.slow_leaders[0].lane_index, ego.lane_index)
            np.testing.assert_allclose([v.position[0] - ego.position[0] for v in env.slow_leaders], [60, 180, 300])
            before = [v.position[0] for v in env.slow_leaders]
            env.step(1)
            np.testing.assert_allclose([v.position[0] - x for v, x in zip(env.slow_leaders, before)], 6)
            self.assertEqual(env.slow_lead_diagnostics()["slow_leaders_passed"], 0)
            env.reset(seed=100000)
            np.testing.assert_allclose([v.position for v in env.road.vehicles], positions)
            env.vehicle.position[0] = env.slow_leaders[-1].position[0] + 10
            self.assertEqual(env.slow_lead_diagnostics()["slow_leaders_passed"], 3)
        finally:
            env.close()

    def test_feasible_truth_controller(self):
        # Feasibility only: this reads simulator truth, unlike the learned agents.
        config = scenario_config()
        for seed in (42, 100000, 100001):
            env = HighwayAdapter(config.highway)
            try:
                env.reset(seed=seed)
                start_x = env.vehicle.position[0]
                for _ in range(240):
                    ego = env.vehicle
                    lane = ego.lane_index[2]
                    front = [v.position[0] - ego.position[0] for v in env.slow_leaders
                             if v.lane_index[2] == lane and v.position[0] > ego.position[0]]
                    action = 1
                    if front and min(front) < 70:
                        for candidate in (lane - 1, lane + 1):
                            if 0 <= candidate < 3 and all(
                                abs(v.position[0] - ego.position[0]) > 35
                                for v in env.slow_leaders if v.lane_index[2] == candidate):
                                action = 0 if candidate < lane else 2
                                break
                    _, _, terminated, truncated, _ = env.step(action)
                    self.assertFalse(ego.crashed)
                    if ego.position[0] - start_x >= 2250 or terminated or truncated:
                        break
                self.assertGreaterEqual(ego.position[0] - start_x, 2250)
                self.assertEqual(env.slow_lead_diagnostics()["slow_leaders_passed"], 3)
            finally:
                env.close()

    def test_opt_in_and_validation(self):
        env = HighwayAdapter(ExperimentConfig().highway)
        try:
            self.assertEqual(env.slow_leaders, [])
            self.assertEqual(len(env.road.vehicles), 13)
        finally:
            env.close()
        config = scenario_config()
        config.highway["slow_lead_scenario"][0]["gap_m"] = 0
        with self.assertRaises(ValueError):
            config.validate()
