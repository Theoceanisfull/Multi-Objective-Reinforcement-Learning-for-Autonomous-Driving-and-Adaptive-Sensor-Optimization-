"""Check the information boundary and sensing intervention in study helpers."""
import unittest
import numpy as np
from algorithms.sensor_agent import LinearExpectedSarsa, ObservationEncoder
from environments.sensor_highway import ExperimentConfig, SensorHighwayEnv
from experiments.sensor_study_helpers import modality_mask, measure_sensor, frozen_policy_rollout


class SensorStudyTests(unittest.TestCase):
    def test_modality_masks_and_removal(self):
        config = ExperimentConfig()
        self.assertEqual(modality_mask(config, 'off').sum(), 0)
        self.assertEqual(modality_mask(config, 'all_on').sum(), 18)
        for kind, count in [('rgb', 8), ('event', 8), ('radar', 1), ('lidar', 1)]:
            mask = modality_mask(config, kind)
            self.assertEqual(mask.sum(), count)
            np.testing.assert_array_equal(mask + modality_mask(config, 'without_' + kind), np.ones(18))
        with self.assertRaises(ValueError):
            modality_mask(config, 'unknown')

    def test_measurement_capabilities_and_motion(self):
        config = ExperimentConfig()
        for kind in ['rgb', 'event', 'radar', 'lidar']:
            sensor = next(s for s in config.sensors if s.kind == kind)
            result = measure_sensor(sensor, 10, -10, trials=100)
            self.assertGreater(result['detections'], 0)
            self.assertEqual(np.isnan(result['range_rmse_m']), kind in {'rgb', 'event'})
            self.assertEqual(np.isnan(result['velocity_rmse_m_s']), kind != 'radar')
        event = next(s for s in config.sensors if s.kind == 'event')
        self.assertEqual(measure_sensor(event, 10, 0, trials=100)['detections'], 0)

    def test_frozen_driver_mask_changes_energy_without_weight_update(self):
        config = ExperimentConfig()
        config.highway.update(duration=.5, vehicles_count=0)
        config.reward.route_distance_m = 1000
        encoder = ObservationEncoder(config)
        with SensorHighwayEnv(config) as env:
            obs, _ = env.reset(seed=42)
        agent = LinearExpectedSarsa(len(encoder(obs)), modes=('all_on',), feature_schema=encoder.feature_schema)
        # Strong IDLE preference keeps the driving intervention controlled.
        agent.weights[1, 0] = 1
        before = agent.weights.copy()
        all_on, h1 = frozen_policy_rollout(config, agent, 'all_on', 42)
        off, h2 = frozen_policy_rollout(config, agent, 'off', 42)
        np.testing.assert_array_equal(agent.weights, before)
        self.assertEqual(h1.drive.tolist(), h2.drive.tolist())
        self.assertAlmostEqual(all_on['distance_m'], off['distance_m'])
        self.assertGreater(all_on['energy_j'], off['energy_j'])
        self.assertTrue(np.isnan(off['completion_time_s']))
