"""Expected-policy targets, plasticity boundaries, serialization, and task integration."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from algorithms.snn import SNNSettings, SpikingExpectedSarsa
from environments.sensor_highway import ExperimentConfig
from experiments.agent_helpers import run_agent
from experiments.snn_helpers import train_snn


class SNNTests(unittest.TestCase):
    def agent(self, rule='surrogate'):
        return SpikingExpectedSarsa(3, SNNSettings(hidden_size=8, simulation_steps=8),
                                    modes=('all_on',), seed=7, learning_rule=rule)

    def test_expected_target_and_terminal(self):
        agent = self.agent()
        with torch.no_grad():
            agent.network.readout.bias.copy_(torch.arange(5.))
        state = np.array([1., .5, -.4])
        for epsilon in (0., .3, 1.):
            target = agent.td_target(2., state, False, epsilon)
            self.assertAlmostEqual(float(target), 2 + .95 * ((1-epsilon)*4 + epsilon*2), places=5)
            self.assertFalse(target.requires_grad)
        # Terminal targets must not even evaluate a potentially invalid next state.
        self.assertEqual(float(agent.td_target(-10., None, True, .5)), -10.)

    def test_learning_rules_and_signed_values(self):
        state = np.array([1., .5, -.4])
        for rule in SpikingExpectedSarsa.MODES:
            agent = self.agent(rule)
            with torch.no_grad():
                agent.network.input.weight.fill_(.15)
            hidden = agent.network.input.weight.detach().clone()
            for _ in range(4):
                agent.update(state, 0, -2., state, True, .2)
            self.assertLess(agent.values(state)[0], 0)
            self.assertEqual(agent.q_updates, 4)
            self.assertTrue(all(np.isfinite(v) for v in agent.last_metrics.values()))
            changed = not torch.equal(hidden, agent.network.input.weight)
            self.assertEqual(changed, rule != 'reservoir')
            if rule == 'td_stdp':
                self.assertGreater(agent.last_metrics['mean_abs_eligibility'], 0.)
                self.assertLessEqual(float(agent.network.input.weight.abs().max()), agent.settings.weight_bound)
            # Only the selected output row receives the direct readout update.
            torch.testing.assert_close(agent.network.readout.weight[1:], torch.zeros_like(agent.network.readout.weight[1:]))

    def test_target_is_detached_from_readout_gradient(self):
        agent = self.agent('reservoir')
        state = np.array([1., .5, -.4])
        with torch.no_grad():
            agent.network.readout.bias.copy_(torch.arange(5.))
        agent.update(state, 0, 1., state, False, .5)
        expected = torch.tensor([-3.85, 0., 0., 0., 0.])  # delta = 1+.95*3-0
        torch.testing.assert_close(agent.network.readout.bias.grad, expected)

    def test_observation_window_reset_and_roundtrip(self):
        state, other = np.array([1., .5, -.4]), np.array([1., 0., 1.])
        for rule in SpikingExpectedSarsa.MODES:
            agent = self.agent(rule)
            agent.update(state, 1, 2., other, False, .3)
            before = agent.values(state)
            agent.values(other)
            np.testing.assert_array_equal(agent.values(state), before)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'snn.pt'
                agent.save(path)
                restored = SpikingExpectedSarsa.load(path)
                self.assertEqual(restored.learning_rule, rule)
                self.assertEqual(restored.actions, agent.actions)
                np.testing.assert_array_equal(restored.values(state), before)

    def test_route_cutoff_and_frozen_evaluation(self):
        for route, expected in [(2250., True), (None, False)]:
            config = ExperimentConfig()
            config.highway.update(duration=.5, vehicles_count=0)
            config.reward.route_distance_m = route
            flags = []
            original = SpikingExpectedSarsa.update

            def record(agent, features, action, reward, next_features, terminal, epsilon):
                flags.append(terminal)
                return original(agent, features, action, reward, next_features, terminal, epsilon)

            with patch.object(SpikingExpectedSarsa, 'update', record):
                a, h1 = train_snn(config, SNNSettings(episodes=1, hidden_size=8), seed=8)
            self.assertEqual(flags, [expected])
            b, h2 = train_snn(config, SNNSettings(episodes=1, hidden_size=8), seed=8)
            np.testing.assert_array_equal(h1['return'], h2['return'])
            for pa, pb in zip(a.network.parameters(), b.network.parameters()):
                torch.testing.assert_close(pa, pb, rtol=0, atol=0)
            before = {k: v.clone() for k, v in a.network.state_dict().items()}
            updates = a.q_updates
            run_agent(config, a, seed=100000)
            self.assertEqual(a.q_updates, updates)
            for k, v in a.network.state_dict().items():
                torch.testing.assert_close(before[k], v, rtol=0, atol=0)

    def test_invalid_inputs(self):
        for changes in ({'simulation_steps': 0}, {'gamma': float('nan')}, {'learning_rate': float('inf')}):
            with self.assertRaises(ValueError):
                SNNSettings(**changes).validate()
        with self.assertRaises(ValueError):
            self.agent().values([1., float('nan'), 0.])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
