"""Ensure shortened training has its own exploration schedule and provenance."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from experiments import run_9000m_study as study
from experiments.run_9000m_preliminary import prepare_preliminary, EPISODES


class PreliminaryTests(unittest.TestCase):
    def test_separate_budget_schedule_and_reused_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            previous=study.prepare(Path(temp)/'full')
            old=json.loads((previous/'protocol.json').read_text())
            study.atomic_json(previous/'study_status.json', {'phase':'stopped'})
            study.atomic_json(previous/'validation.json', {
                'passed': True,
                'config_sha256': hashlib.sha256((previous/'config.json').read_bytes()).hexdigest(),
                'backend_sha256': old['source_hashes']['environments/sensor_highway/overtaking.py'],
            })
            (previous/'scenario_validation.csv').write_text('controller,completed\noracle,True\n')
            pilot=study.prepare(previous/'pilot',episodes=3,seeds=[70042],pilot=True)
            study.atomic_json(pilot/'study_status.json', {'phase':'complete'})
            output=prepare_preliminary(Path(temp)/'preliminary',previous)
            protocol=study.read_protocol(output)
            self.assertEqual(protocol['episodes'],500)
            self.assertEqual(500*len(protocol['policies'])*len(protocol['training_seeds']),9000)
            self.assertEqual(protocol['config'],old['config'])
            self.assertIn('experiments/run_9000m_preliminary.py',protocol['source_hashes'])
            self.assertAlmostEqual(1-.9*(EPISODES-1)/(EPISODES-1),.1)
            for policy in study.POLICIES:
                agent=study.make_agent(policy,112,'sensor_grid_regions_route_v2',42,protocol)
                if hasattr(agent,'settings'):self.assertEqual(agent.settings.episodes,500)
            self.assertEqual(json.loads((output/'pilot/study_status.json').read_text())['phase'],'complete')
            self.assertFalse(list(output.glob('seed*/*/resume.pkl')))
            prepare_preliminary(output,previous)  # Idempotent for the same protocol.
            self.assertEqual(json.loads((previous/'protocol.json').read_text()),old)

    def test_fresh_checkout_can_prepare_without_previous_study(self):
        with tempfile.TemporaryDirectory() as temp:
            output=prepare_preliminary(Path(temp)/'preliminary')
            self.assertEqual(study.read_protocol(output)['episodes'],EPISODES)
            self.assertFalse((output/'pilot').exists())

    def test_cannot_overwrite_original_study(self):
        with self.assertRaises(ValueError):prepare_preliminary(study.DEFAULT_OUTPUT)


if __name__=='__main__':unittest.main()
