"""Check learning targets and observation-only action selection."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from algorithms.sensor_agent import LinearExpectedSarsa, ObservationEncoder, AgentSettings
from environments.sensor_highway import ExperimentConfig, SensorHighwayEnv
from experiments.agent_helpers import train_agent, run_agent


class AgentTests(unittest.TestCase):
    def test_expected_target_and_terminal_target(self):
        agent = LinearExpectedSarsa(2, modes=['all_on'])
        agent.weights[0, 1] = 4
        state, next_state = np.array([1., 0]), np.array([0., 1.])
        # epsilon=.5: expectation=.5*4 + .5*(4/5)=2.4
        agent.update(state, 1, 1, next_state, False, .5, .1, .9)
        self.assertAlmostEqual(agent.weights[1, 0], .316)
        agent.update(state, 2, -5, next_state, True, .5, .1, .9)
        self.assertAlmostEqual(agent.weights[2, 0], -.5)

    def test_features_do_not_treat_unknown_as_free(self):
        config = ExperimentConfig()
        with SensorHighwayEnv(config) as env:
            obs, _ = env.reset(seed=42)
            encoded = ObservationEncoder(config)(obs)
            # Bias/telemetry precede 15 regions x 6 features; unknown has no evidence.
            np.testing.assert_array_equal(encoded[4:94], np.zeros(90))
            self.assertTrue(np.isfinite(encoded).all())
            agent = LinearExpectedSarsa(len(encoded))
            for index in range(len(agent.actions)):
                self.assertTrue(env.action_space.contains(agent.decode(index, env)))

    def test_route_deadline_target_features_and_model_roundtrip(self):
        from unittest.mock import patch
        from algorithms.sensor_agent import ROUTE_FEATURE_SCHEMA
        config = ExperimentConfig()
        config.highway.update(duration=.5, vehicles_count=0)
        config.reward.route_distance_m = 1000
        terminal_flags = []
        original = LinearExpectedSarsa.update

        def record(agent, features, action, reward, next_features, terminated, *args):
            terminal_flags.append(terminated)
            return original(agent, features, action, reward, next_features, terminated, *args)

        with patch.object(LinearExpectedSarsa, 'update', record):
            agent, history = train_agent(config, AgentSettings(episodes=1), seed=8)
        self.assertEqual(terminal_flags, [True])
        self.assertEqual(agent.feature_schema, ROUTE_FEATURE_SCHEMA)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'route.npz'
            agent.save(path)
            restored = LinearExpectedSarsa.load(path)
            self.assertEqual(restored.feature_schema, ROUTE_FEATURE_SCHEMA)
            np.testing.assert_array_equal(restored.weights, agent.weights)
            metrics, _ = run_agent(config, restored, seed=100000)
        self.assertFalse(metrics['completed'])
        self.assertTrue(np.isnan(metrics['completion_time_s']))

    def test_train_replay_evaluation_and_save(self):
        config = ExperimentConfig()
        config.highway['duration'] = 1
        settings = AgentSettings(episodes=2)
        a, h1 = train_agent(config, settings, seed=8)
        b, h2 = train_agent(config, settings, seed=8)
        np.testing.assert_array_equal(a.weights, b.weights)
        np.testing.assert_array_equal(h1['return'], h2['return'])
        before = a.weights.copy()
        metrics, _ = run_agent(config, a, seed=100000)
        np.testing.assert_array_equal(a.weights, before)
        self.assertGreater(metrics['elapsed_s'], 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'agent.npz'
            a.save(path)
            restored = LinearExpectedSarsa.load(path)
            np.testing.assert_array_equal(restored.weights, a.weights)
            self.assertEqual(restored.actions, a.actions)


if __name__ == '__main__':
    unittest.main()
