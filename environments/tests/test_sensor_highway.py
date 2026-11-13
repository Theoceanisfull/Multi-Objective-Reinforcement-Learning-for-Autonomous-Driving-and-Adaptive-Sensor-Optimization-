"""Behavioral tests for information boundaries, timing, geometry, and costs."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from highway_env.envs.highway_env import HighwayEnv
from environments.sensor_highway import ExperimentConfig, SensorConfig, SensorHighwayEnv, WeatherConfig
from environments.sensor_highway.backend import VehicleTruth, Scene
from environments.sensor_highway.grid import GridBuilder
from environments.sensor_highway.sensors import Detection, Packet, SensorSuite
from environments.sensor_highway.weather import WeatherModel


def car(x, y=0, speed=25):
    return VehicleTruth(np.array([x, y], dtype=float), np.array([speed, 0.]), 0., 5., 2.)


def suite(sensor):
    rng = np.random.default_rng(7)
    return SensorSuite([sensor], WeatherModel(WeatherConfig(), rng), rng)


class SensorTests(unittest.TestCase):
    def test_bearing_only_camera_does_not_export_true_depth(self):
        sensor = SensorConfig("camera", "rgb", detection_probability=1,
                              false_positive_rate_hz=0, range_m=100000)
        a, b = suite(sensor), suite(sensor)
        p1 = a._capture(0, 0, Scene(car(0), (car(20),)))
        p2 = b._capture(0, 0, Scene(car(0), (car(80),)))
        self.assertEqual(len(p1.detections), 1)
        self.assertEqual(p1.detections, p2.detections)
        self.assertIsNone(p1.detections[0].range_m)
        self.assertIsNone(p1.detections[0].radial_velocity)

    def test_occluded_car_does_not_generate_second_return(self):
        sensor = SensorConfig("lidar", "lidar", fov_deg=360, range_m=100000,
                              detection_probability=1, false_positive_rate_hz=0)
        packet = suite(sensor)._capture(0, 0, Scene(car(0), (car(10), car(30))))
        self.assertEqual(len(packet.detections), 1)

    def test_delay_warmup_energy_and_cancel(self):
        sensor = SensorConfig("radar", "radar", rate_hz=10, warmup_s=.1,
                              latency_s=.05, power_w=10, standby_w=1)
        sensors = suite(sensor)
        scene = Scene(car(0), (car(20),))
        sensors.set_active([1], 0)
        for t in np.arange(.01, .15, .01):
            sensors.tick(float(t), .01, scene)
        self.assertFalse(sensors.latest)
        sensors.tick(.15, .01, scene)
        self.assertAlmostEqual(sensors.latest[0].capture_time, .1)
        self.assertAlmostEqual(sensors.total_energy_j, 1.5)
        sensors.tick(.2, .05, scene)
        self.assertTrue(sensors.pending)
        sensors.set_active([0], .2)
        self.assertFalse(sensors.pending)
        sensors.tick(.3, .1, scene)
        self.assertAlmostEqual(sensors.total_energy_j, 2.1)
        self.assertAlmostEqual(sensors.latest[0].capture_time, .1)

    def test_weather_direction_and_relative_robustness(self):
        model = WeatherModel(WeatherConfig("rain", 1), np.random.default_rng(1))
        radar, rgb = model.effect("radar", 0), model.effect("rgb", 0)
        self.assertLessEqual(radar.detection_factor, 1)
        self.assertGreater(radar.detection_factor, rgb.detection_factor)
        glare = WeatherModel(WeatherConfig("glare", 1), np.random.default_rng(1))
        self.assertLess(glare.effect("rgb", 0).detection_factor,
                        glare.effect("event", 0).detection_factor)
        self.assertEqual(glare.effect("rgb", np.pi).detection_factor, 1)
        fog = WeatherModel(WeatherConfig("fog", 1), np.random.default_rng(1))
        self.assertEqual(fog.effect("event", 0), fog.effect("rgb", 0))

    def test_weather_coefficients_can_be_overridden(self):
        config = WeatherConfig("rain", 1, effect_overrides={"rain": {"radar": [.8, .7, 2, 3]}})
        effect = WeatherModel(config, np.random.default_rng(1)).effect("radar", 0)
        self.assertEqual(effect.detection_factor, .8)
        self.assertEqual(effect.noise_factor, 2)

    def test_east_camera_sees_right_not_left(self):
        sensor = SensorConfig("east", "rgb", heading_deg=90, fov_deg=45,
                              range_m=100000, detection_probability=1, false_positive_rate_hz=0)
        packet = suite(sensor)._capture(0, 0, Scene(car(0), (car(0, -20), car(0, 20))))
        self.assertEqual(len(packet.detections), 1)
        self.assertAlmostEqual(packet.detections[0].bearing, np.pi / 2, delta=.1)


class GridTests(unittest.TestCase):
    def setUp(self):
        self.builder = GridBuilder(ExperimentConfig().grid)
        self.origin = np.zeros(2)

    def build(self, packets, time=0):
        return self.builder.build(packets, self.origin, 0, time, 25)

    def test_unknown_map_and_expiration(self):
        grid, uncertainty = self.build([])
        self.assertTrue(np.all(grid[0][self.builder.valid] == .5))
        self.assertEqual(uncertainty, 1)
        detection = Detection(0, .01, 30, 1, None, 100)
        packet = Packet(0, 0, 0, self.origin, (detection,))
        old_grid, _ = self.build([packet], time=4)
        np.testing.assert_array_equal(old_grid, grid)

    def test_fusion_increases_evidence_at_agreement_and_age_weakens_it(self):
        radar = Packet(0, 0, 0, self.origin, (Detection(0, .1, 30, 1, -5, 100),))
        camera = Packet(1, 0, 0, self.origin, (Detection(0, .02, None, 1, None, 100),))
        single, _ = self.build([radar])
        fused, _ = self.build([radar, camera])
        row, col = np.unravel_index(np.argmin((self.builder.forward - 30)**2
                                            + self.builder.right**2), self.builder.forward.shape)
        self.assertGreater(fused[0, row, col], single[0, row, col])
        aged, _ = self.build([radar], time=1)
        self.assertLess(aged[0, row, col], single[0, row, col])
        self.assertGreater(single[4, row, col], 0)
        self.assertLess(single[3, row, col], 0)

    def test_highway_right_lane_is_displayed_to_the_right(self):
        # At heading zero a HighwayEnv right-hand lane has positive world y.
        bearing = float(np.arctan2(8, 24))
        packet = Packet(0, 0, 0, self.origin,
                        (Detection(bearing, .01, np.hypot(24, 8), 1, None, 100),))
        grid, _ = self.build([packet])
        peak = np.unravel_index(grid[0].argmax(), grid[0].shape)
        self.assertAlmostEqual(self.builder.right[peak], 8, delta=2)
        self.assertAlmostEqual(self.builder.forward[peak], 24, delta=2)

    def test_world_measurements_reproject_when_ego_rotates(self):
        # HighwayEnv headings turn clockwise: world +x becomes left/W
        # after the ego turns toward world +y.
        packet = Packet(0, 0, 0, self.origin, (Detection(0, .01, 30, 1, None, 100),))
        grid, _ = self.builder.build([packet], self.origin, np.pi/2, 0, 25)
        peak = np.unravel_index(grid[0].argmax(), grid[0].shape)
        self.assertAlmostEqual(self.builder.right[peak], -30, delta=2)
        self.assertAlmostEqual(self.builder.forward[peak], 0, delta=2)


class EnvironmentTests(unittest.TestCase):
    def test_gym_interface_reproducibility(self):
        with SensorHighwayEnv() as env:
            check_env(env, skip_render_check=True)

    def test_disabled_sensors_unknown_and_standby_energy(self):
        with SensorHighwayEnv() as env:
            env.reset(seed=42)
            obs, reward, _, _, info = env.step({"drive": 1, "sensors": np.zeros(18, np.int8)})
            self.assertTrue(np.all(obs["grid"][0][obs["valid_cells"] > 0] == .5))
            self.assertAlmostEqual(info["energy_j"], .5 * sum(s.standby_w for s in env.config.sensors))
            self.assertAlmostEqual(reward, sum(info["reward_components"].values()))

    def test_adapter_preserves_stock_motion(self):
        config = ExperimentConfig()
        with SensorHighwayEnv(config) as wrapped, HighwayEnv(config=config.highway) as stock:
            wrapped.reset(seed=123)
            stock.reset(seed=123)
            for action in [1, 3, 2, 4]:
                _, _, end, cutoff, _ = wrapped.step({"drive": action, "sensors": np.zeros(18, np.int8)})
                _, _, stock_end, stock_cutoff, _ = stock.step(action)
                np.testing.assert_allclose(wrapped.backend.vehicle.position, stock.vehicle.position)
                self.assertEqual((end, cutoff), (stock_end, stock_cutoff))
                if end or cutoff:
                    break

    def test_json_roundtrip_and_invalid_config(self):
        config = ExperimentConfig()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            config.save(path)
            self.assertEqual(config, ExperimentConfig.load(path))
        config.highway["simulation_frequency"] = 11
        with self.assertRaises(ValueError):
            config.validate()


if __name__ == "__main__":
    unittest.main()
