"""Known geometry, causal bearing tracking, and delivery-timing checks."""
import unittest
import numpy as np
from environments.sensor_highway import SensorConfig
from experiments.cutin_study_helpers import CutInSettings, cutin_scene, bearing_rate, run_cutin


class CutInStudyTests(unittest.TestCase):
    def test_geometry_and_stationary_control(self):
        settings = CutInSettings()
        _, bearing, rate, y = cutin_scene(2., 20., 2., settings)
        self.assertAlmostEqual(y, 2.)
        self.assertAlmostEqual(bearing, np.arctan2(2., 20.))
        self.assertAlmostEqual(rate, np.rad2deg(-40 / 404))
        _, _, rate, y = cutin_scene(3., 20., 0., settings)
        self.assertEqual(y, 4.)
        self.assertEqual(rate, 0.)

    def test_bearing_rate_requires_history_and_recovers_linear_motion(self):
        settings = CutInSettings()
        self.assertTrue(np.isnan(bearing_rate([(0., 0.)], settings)))
        samples = [(t, np.deg2rad(10 - 5 * t)) for t in np.linspace(0, .4, 9)]
        self.assertAlmostEqual(bearing_rate(samples, settings), -5.)

    def test_no_target_event_in_stationary_control_and_latency_bound(self):
        settings = CutInSettings(duration_s=2.)
        event = SensorConfig('event', 'event', range_m=100, rate_hz=50, latency_s=.02)
        metrics, trace = run_cutin(event, 20, 0, 42, settings, capture=True)
        self.assertEqual(metrics['target_updates'], 0)
        self.assertFalse(metrics['control_false_alert'])
        self.assertTrue(trace.last_detection_age_s.isna().all())
        metrics, trace = run_cutin(event, 20, 4, 42, settings, capture=True)
        self.assertGreaterEqual(metrics['first_post_onset_detection_delay_s'], event.latency_s - 1e-8)
        self.assertFalse(trace.loc[trace.time_s < settings.onset_s, 'new_target_detection'].any())
        self.assertTrue((trace.last_detection_age_s.dropna() >= event.latency_s - 1e-8).all())
