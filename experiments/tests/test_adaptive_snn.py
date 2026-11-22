"""Homeostasis, representation freezing, and adaptive checkpoint tests."""
from dataclasses import asdict
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.snn import AdaptiveExpectedSarsa, AdaptiveSNNSettings, SpikingExpectedSarsa, SNNSettings
from environments.sensor_highway import ExperimentConfig, SensorHighwayEnv
from algorithms.sensor_agent import ObservationEncoder
from experiments import adaptive_snn_study as study
from experiments.adaptive_snn_study import train


class AdaptiveTests(unittest.TestCase):
    def agent(self, variant='adaptive_reservoir'):
        return AdaptiveExpectedSarsa(16, AdaptiveSNNSettings(hidden_size=8), seed=7, variant=variant)

    def test_homeostasis_reduces_repeated_firing_without_state_leak(self):
        agent = self.agent()
        x = torch.ones(16)
        with torch.no_grad():
            agent.network.input.weight.fill_(.03)
            adaptive, _, theta = agent.network.encode(x, adapt=True)
            fixed, _, _ = agent.network.encode(x, adapt=False)
        self.assertTrue((adaptive < fixed).any())
        self.assertTrue((theta > 0).all())
        torch.testing.assert_close(agent.network.threshold_offset, torch.zeros(8))
        initial = agent.values(x)
        agent.td_target(1, torch.zeros(16), False, .5)
        np.testing.assert_array_equal(agent.values(x), initial)

    def test_stdp_changes_only_pretrained_weights_and_bounds_hold(self):
        for variant in ['prototype_control', 'stdp_pretrained']:
            agent = self.agent(variant)
            before = agent.network.input.weight.clone()
            for i in range(20):
                agent.pretrain_step(np.random.default_rng(i).uniform(-1,1,16))
            changed = not torch.equal(before, agent.network.input.weight)
            self.assertEqual(changed, variant == 'stdp_pretrained')
            weights = agent.network.input.weight
            self.assertTrue((weights >= 0).all())
            self.assertTrue((weights <= agent.settings.prototype_weight_max+1e-6).all())
            torch.testing.assert_close(weights.sum(1), torch.full((8,),4.), atol=1e-5, rtol=0)
            self.assertGreater(float(agent.network.threshold_offset.mean()), 0.)
            torch.testing.assert_close(agent.network.readout.weight, torch.zeros_like(agent.network.readout.weight))

    def test_readout_training_and_inference_freeze_representation(self):
        for variant in AdaptiveExpectedSarsa.VARIANTS:
            agent = self.agent(variant)
            x = np.ones(16)
            if variant != 'adaptive_reservoir':
                agent.pretrain_step(x)
            before = {k:v.clone() for k,v in agent.network.state_dict().items() if not k.startswith('readout.')}
            agent.update(x, 0, -2., x, True, .5)
            self.assertLess(agent.values(x)[0],0.)
            for k,v in before.items():
                torch.testing.assert_close(v, agent.network.state_dict()[k], rtol=0, atol=0)
            self.assertEqual(float(agent.td_target(-10,None,True,.5)),-10)
            self.assertFalse(agent.td_target(1,x,False,.5).requires_grad)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'agent.pt'
                agent.save(path)
                restored = AdaptiveExpectedSarsa.load(path)
                np.testing.assert_array_equal(agent.values(x),restored.values(x))
                self.assertEqual(agent.pretraining_updates, restored.pretraining_updates)

    def test_legacy_initialization_is_paired(self):
        adaptive = self.agent()
        old = SpikingExpectedSarsa(16,SNNSettings(hidden_size=8),seed=7,learning_rule='reservoir')
        torch.testing.assert_close(adaptive.network.input.weight, old.network.input.weight, rtol=0,atol=0)
        torch.testing.assert_close(adaptive.network.input.bias, old.network.input.bias, rtol=0,atol=0)

    def test_task_training_and_control_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp)
            original = ExperimentConfig()
            original.highway['duration'] = 120
            original.reward.route_distance_m = 2250
            original.save(control/'config.json')
            settings = SNNSettings(episodes=60)
            paths = ('algorithms/snn.py','experiments/agent_helpers.py')
            metadata = {'snn_settings': asdict(settings), 'protocol_hash': 'test', 'versions': {},
                        'sources_sha256': {p: hashlib.sha256((study.ROOT/p).read_bytes()).hexdigest()
                                           for p in paths}}
            manifest = control/'metadata.json'
            manifest.write_text(json.dumps(metadata))
            with patch.object(study,'CONTROL',control):
                config, _, _, _ = study.protocol()
                metadata['sources_sha256']['experiments/agent_helpers.py'] = 'old implementation'
                manifest.write_text(json.dumps(metadata))
                with self.assertRaises(AssertionError):
                    study.protocol()
        self.assertEqual(config.highway['duration'],120)
        self.assertEqual(config.reward.route_distance_m,2250)
        config.highway.update(duration=.5,vehicles_count=0)
        encoder = ObservationEncoder(config)
        with SensorHighwayEnv(config) as env:
            obs,_ = env.reset(seed=8)
        cfg = AdaptiveSNNSettings(episodes=1,hidden_size=8)
        agent = AdaptiveExpectedSarsa(len(encoder(obs)),cfg,feature_schema=encoder.feature_schema)
        history = train(config,cfg,agent,8)
        self.assertEqual(len(history),1)
        self.assertEqual(history.steps.iloc[0],1)

    def test_invalid_adaptation_parameters(self):
        for settings in [AdaptiveSNNSettings(threshold_decay=1),AdaptiveSNNSettings(threshold_increment=float('nan'))]:
            with self.assertRaises(ValueError):
                settings.validate()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
