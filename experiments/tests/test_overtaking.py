"""Traffic persistence, truth separation, counters, and exact training resume."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from environments.sensor_highway.overtaking import OvertakingSensorEnv
from experiments import run_9000m_study as study


class OvertakingTests(unittest.TestCase):
    def config(self):
        c=study.make_config()
        c.weather.preset='clear'
        return c

    def test_seeded_traffic_and_no_truth_observation_fields(self):
        c=self.config()
        with OvertakingSensorEnv(c) as a, OvertakingSensorEnv(c) as b:
            oa,_=a.reset(seed=321);ob,_=b.reset(seed=321)
            self.assertEqual(set(oa),{'grid','valid_cells','sensors','ego','task'})
            self.assertTrue((oa['sensors'][:,0]==0).all())
            for i in range(6):
                aa={'drive':1,'sensors':np.zeros(18,np.int8)}
                bb={'drive':1,'sensors':np.ones(18,np.int8)}
                oa,_,_,_,ia=a.step(aa);ob,_,_,_,ib=b.step(bb)
                np.testing.assert_array_equal(a.backend.vehicle.position,b.backend.vehicle.position)
                self.assertEqual(ia['overtakes'],ib['overtakes'])
                np.testing.assert_array_equal(np.array([v.position for v in a.backend.road.vehicles]),
                                              np.array([v.position for v in b.backend.road.vehicles]))

    def test_replenishment_uses_safe_entry_and_distinct_ids(self):
        with OvertakingSensorEnv(self.config()) as env:
            env.reset(seed=4);backend=env.backend
            ids={v.experiment_id for v in backend.road.vehicles if v is not backend.vehicle}
            old_count=backend.spawn_count
            backend.vehicle.position[0]+=2000
            backend._maintain_traffic()
            self.assertGreater(backend.spawn_count,old_count)
            for v in backend.road.vehicles:
                if v is backend.vehicle:continue
                self.assertGreaterEqual(v.position[0]-backend.vehicle.position[0],-100)
                if v.experiment_id not in ids:
                    self.assertGreaterEqual(v.position[0]-backend.vehicle.position[0],300)
            active=[v.experiment_id for v in backend.road.vehicles if v is not backend.vehicle]
            self.assertEqual(len(active),len(set(active)))
            self.assertLess(len(active),20)

    def test_overtakes_count_once_and_ignore_unencountered_spawns(self):
        with OvertakingSensorEnv(self.config()) as env:
            env.reset(seed=4);b=env.backend
            front=min((v for v in b.road.vehicles if v is not b.vehicle and v.position[0]>b.vehicle.position[0]),
                      key=lambda v:v.position[0])
            b._diagnostic_tick(.01)
            self.assertIn(front.experiment_id,b.encountered)
            b.vehicle.position[0]=front.position[0]+15
            b._diagnostic_tick(.01);count=len(b.passed)
            b._diagnostic_tick(.01)
            self.assertEqual(len(b.passed),count)
            self.assertIn(front.experiment_id,b.passed)

    def test_slow_strategy_cannot_meet_deadline(self):
        c=self.config()
        self.assertEqual(c.reward.route_distance_m,9000)
        self.assertEqual(c.highway['duration'],360)
        self.assertGreater(9000/20,360)
        self.assertLess(9000/30,360)

    def test_terminal_reward_makes_early_failure_unattractive(self):
        c=self.config();c.highway['duration']=.5
        with OvertakingSensorEnv(c) as env:
            env.reset(seed=4)
            _,r,done,trunc,info=env.step({'drive':1,'sensors':np.zeros(18,np.int8)})
            self.assertTrue(trunc);self.assertFalse(done)
            self.assertEqual(info['reward_components']['failure_penalty'],-200)
            self.assertLess(r,-190)

    def test_stop_request_checkpoints_then_resumes(self):
        c=self.config();c.highway['duration']=.5
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            with patch.object(study,'make_config',return_value=c):
                study.prepare(output,episodes=2,seeds=[7],pilot=True,checkpoint_every=25)
            original=study.checkpoint
            def request_stop(folder,state,policy):
                original(folder,state,policy)
                if state['episode']==0:(output/'STOP').touch()
            with patch.object(study,'checkpoint',request_stop):
                self.assertIsNone(study.train_one(output,'sarsa',7))
            folder=output/'seed7'/'sarsa'
            self.assertEqual(json.loads((folder/'status.json').read_text())['phase'],'stopped')
            self.assertTrue((folder/'resume.pkl').exists())
            (output/'STOP').unlink()
            study.train_one(output,'sarsa',7)
            self.assertEqual(len(pd.read_csv(folder/'training.csv')),2)

    def test_partial_summary_tolerates_initial_empty_checkpoint_history(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            study.prepare(output,episodes=3,seeds=[7],pilot=True)
            folder=output/'seed7'/'sarsa';folder.mkdir(parents=True)
            (folder/'training.csv').write_text('\n')
            self.assertIsNone(study.summarize(output))

    def test_exact_resume_and_configuration_guard(self):
        c=self.config();c.highway['duration']=.5
        torch.set_num_threads(1)
        for policy in ['sarsa','dqn','dyna_dqn','adaptive_reservoir','prototype_control','stdp_pretrained']:
            with tempfile.TemporaryDirectory() as temp:
                a=Path(temp)/'interrupted';b=Path(temp)/'reference'
                with patch.object(study,'make_config',return_value=c):
                    study.prepare(a,episodes=3,seeds=[7],pilot=True,checkpoint_every=1)
                    study.prepare(b,episodes=3,seeds=[7],pilot=True,checkpoint_every=1)
                original_factory=study.make_agent
                def small_agent(*args,**kwargs):
                    agent=original_factory(*args,**kwargs)
                    if isinstance(agent,study.DQNAgent):
                        agent.settings.batch_size=2
                        agent.settings.learning_starts=2
                        agent.settings.model_learning_starts=2
                        agent.settings.target_update_steps=3
                    return agent
                original=study.checkpoint
                def interrupt(folder,state,name):
                    original(folder,state,name)
                    if state['episode']==2:raise InterruptedError('simulated interruption')
                with patch.object(study,'make_agent',small_agent):
                    with patch.object(study,'checkpoint',interrupt):
                        with self.assertRaises(InterruptedError):study.train_one(a,policy,7)
                    study.train_one(a,policy,7)
                    study.train_one(b,policy,7)
                pa=study.policy_state(study.load_policy(policy,a/'seed7'/policy))
                pb=study.policy_state(study.load_policy(policy,b/'seed7'/policy))
                for k in pa:
                    np.testing.assert_array_equal(np.asarray(pa[k]),np.asarray(pb[k]))
                if policy in ['dqn','dyna_dqn']:
                    import pickle
                    with (a/'seed7'/policy/'resume.pkl').open('rb') as f: aa=pickle.load(f)['agent']
                    with (b/'seed7'/policy/'resume.pkl').open('rb') as f: bb=pickle.load(f)['agent']
                    self.assertGreater(aa.q_updates,0)
                    self.assertEqual(aa.q_updates,bb.q_updates)
                    self.assertEqual(len(aa.replay),3)
                    for va,vb in zip(aa.target.parameters(),bb.target.parameters()):
                        torch.testing.assert_close(va,vb,rtol=0,atol=0)
                    for key,values in aa.optimizer.state_dict()['state'].items():
                        for name,value in values.items():
                            torch.testing.assert_close(value,bb.optimizer.state_dict()['state'][key][name],rtol=0,atol=0)
                    if policy=='dyna_dqn':
                        for va,vb in zip(aa.model.parameters(),bb.model.parameters()):
                            torch.testing.assert_close(va,vb,rtol=0,atol=0)
                ha=pd.read_csv(a/'seed7'/policy/'training.csv')
                hb=pd.read_csv(b/'seed7'/policy/'training.csv')
                np.testing.assert_array_equal(ha['return'],hb['return'])
                (a/'seed7'/policy/'evaluation.csv').unlink()
                study.train_one(a,policy,7)  # Resume after training finished but evaluation was interrupted.
                p=a/'config.json';data=json.loads(p.read_text());data['highway']['duration']=1;p.write_text(json.dumps(data))
                with self.assertRaises(ValueError):study.read_protocol(a)


if __name__=='__main__':unittest.main()
