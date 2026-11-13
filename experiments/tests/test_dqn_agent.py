"""Check DQN targets, real/model separation, and environment integration."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.dqn_agent import DQNAgent, DynaDQNAgent, DQNSettings
from environments.sensor_highway import ExperimentConfig
from experiments.agent_helpers import run_agent
from experiments.dqn_helpers import train_dqn


class DQNTests(unittest.TestCase):
    def settings(self, **changes):
        return DQNSettings(episodes=2, batch_size=2, learning_starts=2,
                           model_learning_starts=2, hidden_size=8, **changes)

    def test_targets_use_frozen_network_and_terminal_mask(self):
        agent = DQNAgent(2, self.settings(), modes=("all_on",))
        with torch.no_grad():
            for p in agent.target.parameters():
                p.zero_()
            agent.target[-1].bias[0] = 4
            agent.online[-1].bias.fill_(100)
        target = agent.td_targets(torch.tensor([1., -5., 1.]), torch.zeros(3, 2),
                                  torch.tensor([0., 1., .25]))
        torch.testing.assert_close(target, torch.tensor([1 + .99 * 4, -5, 1 + .99 * .75 * 4]))
        self.assertFalse(target.requires_grad)

    def test_planning_updates_and_real_replay_separation(self):
        agent = DynaDQNAgent(2, self.settings(target_update_steps=3), modes=("all_on",))
        state = np.array([1., .2])
        before_model = [p.clone() for p in agent.model.parameters()]
        self.assertEqual(agent.observe(state, 0, 2., state, False), {})
        metrics = agent.observe(state, 1, -1., state, True)
        self.assertEqual(agent.q_updates, 3)  # One real update plus two imagined updates.
        self.assertEqual(len(agent.replay), 2)  # Imagined steps are never stored as real data.
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_model, agent.model.parameters())))
        for a, b in zip(agent.online.parameters(), agent.target.parameters()):
            torch.testing.assert_close(a, b)
        imagined = agent.imagine_batch(agent.replay.sample(2, agent.rng))
        self.assertFalse(imagined[3].requires_grad)
        torch.testing.assert_close(imagined[3][:, 0], torch.ones(2))
        self.assertTrue(((imagined[4] >= 0) & (imagined[4] <= 1)).all())

    def test_route_and_legacy_cutoff_targets(self):
        for route, expected in [(None, False), (1000, True)]:
            config = ExperimentConfig()
            config.highway.update(duration=.5, vehicles_count=0)
            config.reward.route_distance_m = route
            flags = []
            original = DQNAgent.observe

            def record(agent, features, action, reward, next_features, terminal):
                flags.append(terminal)
                return original(agent, features, action, reward, next_features, terminal)

            with patch.object(DQNAgent, "observe", record):
                train_dqn(config, DQNSettings(episodes=1), seed=8)
            self.assertEqual(flags, [expected])

    def test_repeatable_training_policy_save_and_frozen_evaluation(self):
        config = ExperimentConfig()
        config.highway.update(duration=1, vehicles_count=0)
        config.reward.route_distance_m = 1000
        for dyna in [False, True]:
            a, h1 = train_dqn(config, self.settings(), seed=8, dyna=dyna)
            b, h2 = train_dqn(config, self.settings(), seed=8, dyna=dyna)
            for pa, pb in zip(a.online.parameters(), b.online.parameters()):
                torch.testing.assert_close(pa, pb, rtol=0, atol=0)
            np.testing.assert_array_equal(h1['return'], h2['return'])
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'agent.pt'
                a.save(path)
                restored = DQNAgent.load(path)
                self.assertIs(type(a), type(restored))
                self.assertEqual(a.actions, restored.actions)
                state = np.ones(a.feature_count)
                np.testing.assert_array_equal(a.values(state), restored.values(state))
                before = [p.clone() for p in restored.online.parameters()]
                metrics, _ = run_agent(config, restored, seed=100000)
                for pa, pb in zip(before, restored.online.parameters()):
                    torch.testing.assert_close(pa, pb, rtol=0, atol=0)
                self.assertEqual(len(restored.replay), 0)
                self.assertFalse(metrics['completed'])


if __name__ == '__main__':
    unittest.main()
