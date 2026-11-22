"""Controller masks, real/model separation, seeded traffic and planning behavior."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pickle

import numpy as np
import torch

from algorithms.curriculum_driving import CurriculumDQN,CurriculumDynaDQN,CurriculumSettings,action_mask,project_next
from environments.sensor_highway.driving import driving_config
from environments.sensor_highway.driving_curriculum import CurriculumEnv,SCENARIOS,scenario_config,episode_scenario


def features():
    x=np.zeros(123,np.float32);x[0]=1.
    x[-11:]=[.5,.5,25/30,0,1,1,0,1,0,0,0]
    return x


class CurriculumTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def settings(self,**kw):
        defaults=dict(episodes=3,hidden_size=8,model_hidden_size=8,batch_size=2,replay_capacity=128,
            learning_starts=2,model_learning_starts=4,min_model_validation=2,validation_interval=1000)
        return CurriculumSettings(**{**defaults,**kw})

    def test_masks_apply_to_exploration_and_bellman_selection(self):
        a=CurriculumDQN(123,self.settings());x=features()
        x[-10]=0.;x[-7]=0.;x[-6]=1. # Leftmost target lane.
        with torch.no_grad():
            for net in [a.online,a.target]:
                for p in net.parameters():p.zero_()
            a.online[-1].bias.copy_(torch.tensor([100.,0.,5.,0.,0.]))
            a.target[-1].bias.copy_(torch.tensor([1000.,0.,2.,0.,0.]))
        rng=np.random.default_rng(7)
        self.assertEqual(a.select(x,0.,rng),2)
        self.assertTrue(all(a.select(x,1.,rng)!=0 for _ in range(100)))
        target=a.td_targets(torch.tensor([1.,-2.]),torch.tensor(np.stack([x,x])),torch.tensor([0.,1.]))
        torch.testing.assert_close(target,torch.tensor([1+.99*2,-2.]))
        x[-8]=1.
        mask=action_mask(torch.tensor(x)[None])[0]
        self.assertFalse(mask[0]);self.assertFalse(mask[2]);self.assertTrue(mask[1])

    def test_rollout_controller_projection_and_terminal_anchor(self):
        x=torch.tensor(features())[None]
        nxt=project_next(x,torch.tensor([2]),x.clone())
        self.assertEqual(float(nxt[0,-10]),1.)
        self.assertEqual(float(nxt[0,-8]),1.)
        self.assertEqual(float(nxt[0,-6]),0.)
        torch.testing.assert_close(nxt[0,-5:],torch.tensor([0.,0.,1.,0.,0.]))
        a=CurriculumDynaDQN(123,self.settings())
        bad=(x,torch.tensor([1]),torch.tensor([0.]),x,torch.tensor([1.]))
        with self.assertRaises(ValueError):a.imagine_batch(bad)

    def test_terminal_data_only_real_and_holdout_episode_excluded_from_model_fit(self):
        a=CurriculumDynaDQN(123,self.settings());x=features()
        a.begin_episode(0)
        a.observe(x,1,.01,x,False);a.observe(x,1,-2.,x,True)
        self.assertEqual(len(a.replay),2);self.assertEqual(len(a.model_replay),1)
        a.begin_episode(9)
        a.observe(x,1,.01,x,False);a.observe(x,1,-2.,x,True)
        self.assertEqual(len(a.replay),4);self.assertEqual(len(a.model_replay),1)
        self.assertEqual(len(a.validation),1)
        self.assertTrue(all(item[-1]==0 for item in a.model_replay.items))

    def test_planning_uses_learned_predictions_without_polluting_real_replay(self):
        a=CurriculumDynaDQN(123,self.settings());x=features()
        for _ in range(6):a.observe(x,1,.01,x,False)
        a.validation.extend([(x,1,.01,x,0.)]*3)
        # Isolate acceptance/rollout paths; the real pilot tests fitted predictions.
        def exact(states,actions):return states.clone(),torch.full((len(states),),.01)
        a.predictions=exact
        a.validate_model();self.assertTrue(a.planning_allowed)
        # Disable optimizer-only model fitting because the mock has no gradients.
        a.learn_model=lambda batch:0.
        old=a.q_updates;a.observe(x,1,.01,x,False)
        self.assertEqual(a.planning_updates,1);self.assertEqual(a.q_updates,old+2)
        self.assertEqual(len(a.replay),7)
        a.model_diagnostics={};a.predictions=lambda s,ac:(torch.ones_like(s),torch.ones(len(s)))
        a.validate_model();self.assertFalse(a.planning_allowed)

    def test_dyna_disabled_matches_real_dqn_updates(self):
        s=self.settings(planning_steps=0)
        a=CurriculumDQN(123,s,7);b=CurriculumDynaDQN(123,s,7);x=features()
        for i in range(20):
            for agent in [a,b]:agent.observe(x,1,.01 if i%5 else -2.,x,i%5==0)
        for va,vb in zip(a.online.parameters(),b.online.parameters()):torch.testing.assert_close(va,vb,rtol=0,atol=0)
        self.assertEqual(b.planning_updates,0)

    def test_portable_roundtrip(self):
        a=CurriculumDynaDQN(123,self.settings());x=features()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'agent.pt';a.save(p);b=CurriculumDQN.load(p)
            np.testing.assert_array_equal(a.values(x),b.values(x))

    def test_schedule_and_traffic_are_paired_and_cover_all_lanes(self):
        schedule=[episode_scenario(i,100,42) for i in range(100)]
        self.assertEqual(schedule,[episode_scenario(i,100,42) for i in range(100)])
        self.assertEqual({s[0] for s in schedule},set(SCENARIOS))
        self.assertTrue(all(s[1]=='clear' for s in schedule[:20]))
        c=scenario_config(driving_config(),'repeated_dense');lanes=set()
        with CurriculumEnv(c) as a,CurriculumEnv(c) as b:
            for seed in range(20):
                oa,ia=a.reset(seed=seed);ob,ib=b.reset(seed=seed)
                self.assertTrue(a.observation_space.contains(oa))
                self.assertEqual(len(oa['controller']),11)
                lanes.add(a.backend.vehicle.lane_index[2])
                for va,vb in zip(a.backend.road.vehicles,b.backend.road.vehicles):
                    np.testing.assert_array_equal(va.position,vb.position)
                self.assertEqual(ia['lane_changes'],0)
        self.assertEqual(lanes,{0,1,2})

    def test_clean_stop_and_exact_resume_matches_uninterrupted_training(self):
        from experiments import train_driving_curriculum as study
        settings=CurriculumSettings(episodes=2,planning_steps=0,batch_size=2,
            learning_starts=2,model_learning_starts=4,replay_capacity=16)
        for policy in ['dqn','dyna_dqn']:
            with tempfile.TemporaryDirectory() as temp:
                a=study.create_study(settings,Path(temp)/'a',deadline_s=1.,eval_seeds=0,final_seeds=0,replays=False)
                b=study.create_study(settings,Path(temp)/'b',deadline_s=1.,eval_seeds=0,final_seeds=0,replays=False)
                class StopAfterEpisode(CurriculumEnv):
                    def step(self,action):
                        result=super().step(action)
                        if result[2] or result[3]:(a/'STOP').touch()
                        return result
                with patch.object(study,'CurriculumEnv',StopAfterEpisode):study.train_one(a,policy)
                import json
                self.assertEqual(json.loads((a/policy/'status.json').read_text())['phase'],'stopped')
                (a/'STOP').unlink()
                study.train_one(a,policy);study.train_one(b,policy)
                with (a/policy/'resume.pkl').open('rb') as f:aa=pickle.load(f)
                with (b/policy/'resume.pkl').open('rb') as f:bb=pickle.load(f)
                self.assertEqual(aa['episode'],2)
                self.assertEqual(aa['agent'].q_updates,bb['agent'].q_updates)
                for va,vb in zip(aa['agent'].online.parameters(),bb['agent'].online.parameters()):
                    torch.testing.assert_close(va,vb,rtol=0,atol=0)
                if policy=='dyna_dqn':
                    for va,vb in zip(aa['agent'].model.parameters(),bb['agent'].model.parameters()):
                        torch.testing.assert_close(va,vb,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
