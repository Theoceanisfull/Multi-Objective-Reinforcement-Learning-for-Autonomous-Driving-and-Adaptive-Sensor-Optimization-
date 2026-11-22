"""Driving observations, Double targets, Dyna ablations and frozen evaluation."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from environments.sensor_highway.driving import DrivingEnv,driving_config
from environments.sensor_highway.overtaking import oracle_action
from algorithms.driving_agents import (DrivingSettings,DrivingEncoder,DrivingExpectedSarsa,
                                      DrivingDQN,DrivingDynaDQN)
from experiments.driving_policy_helpers import fit_driving,rollout_driving,check_q


class DrivingTests(unittest.TestCase):
    def settings(self,**changes):
        defaults=dict(episodes=2,hidden_size=8,batch_size=2,learning_starts=2,
                      replay_capacity=64,model_learning_starts=4,target_update_steps=3,
                      validation_interval=2,min_validation_terminals=1)
        return DrivingSettings(**{**defaults,**changes})

    def test_five_actions_all_on_ready_and_controller_telemetry(self):
        c=driving_config();encoder=DrivingEncoder(c)
        with DrivingEnv(c) as env:
            obs,info=env.reset(seed=4)
            self.assertEqual(env.action_space.n,5)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue((obs['sensors'][:,0]==1).all())
            self.assertTrue((obs['sensors'][:,1]==1).all())
            self.assertEqual(info['time_s'],.5)
            self.assertIn('warmup_reward',info)
            self.assertEqual(len(encoder(obs)),123)
            self.assertEqual(set(obs),{'grid','valid_cells','sensors','ego','task','controller'})
            self.assertGreater(obs['grid'][1].max(),0)
            obs,_,_,_,_=env.step(3)
            self.assertAlmostEqual(obs['controller'][2],1.)
            self.assertEqual(obs['controller'][-2],1.)
            self.assertTrue((obs['sensors'][:,0]==1).all())
            with self.assertRaises(ValueError):env.step(5)

    def test_double_target_uses_online_selection_and_target_evaluation(self):
        agent=DrivingDQN(3,self.settings())
        with torch.no_grad():
            for net in [agent.online,agent.target]:
                for p in net.parameters():p.zero_()
            agent.online[-1].bias.copy_(torch.tensor([0.,5.,1.,2.,3.]))
            agent.target[-1].bias.copy_(torch.tensor([10.,2.,0.,0.,0.]))
        target=agent.td_targets(torch.tensor([1.,-2.]),torch.zeros(2,3),torch.tensor([0.,1.]))
        torch.testing.assert_close(target,torch.tensor([1+.99*2,-2.]))
        self.assertFalse(target.requires_grad)
        agent.options.double_q=False
        target=agent.td_targets(torch.tensor([1.]),torch.zeros(1,3),torch.tensor([0.]))
        torch.testing.assert_close(target,torch.tensor([1+.99*10]))

    def test_dyna_no_planning_has_identical_real_q_updates_and_independent_model_rng(self):
        settings=self.settings(planning_steps=0)
        a=DrivingDQN(3,settings,seed=7);b=DrivingDynaDQN(3,settings,seed=7)
        state=np.array([1.,.3,.5],np.float32)
        for i in range(30):
            reward=-2. if i%7==0 else .01
            a.observe(state,i%5,reward,state,i%7==0)
            b.observe(state,i%5,reward,state,i%7==0)
        self.assertEqual(a.q_updates,b.q_updates)
        for va,vb in zip(a.online.parameters(),b.online.parameters()):
            torch.testing.assert_close(va,vb,rtol=0,atol=0)
        self.assertEqual(b.planning_updates,0)
        self.assertEqual(len(b.validation),3)
        self.assertEqual(len(b.model_replay),27)
        self.assertEqual(len(b.replay),30)

    def test_planning_requires_heldout_terminal_coverage_and_accuracy(self):
        b=DrivingDynaDQN(3,self.settings(),seed=7)
        state=np.array([1.,0.,0.],np.float32)
        for _ in range(30):b.observe(state,0,0.,state,False)
        b.validate_model();self.assertFalse(b.planning_allowed)
        self.assertEqual(b.planning_updates,0)
        # Perfect mock model makes the guard pass only after both outcome classes
        # are represented; it does not write into real replay during planning.
        b.validation.clear()
        for i in range(25):
            x=np.array([1.,float(i==0),0.],np.float32)
            b.validation.append((x,0,0.,x,float(i==0)))
        def perfect(states,actions):
            return torch.zeros_like(states),torch.zeros(len(states)),torch.where(states[:,1]>.5,100.,-100.)
        b.model_predictions=perfect
        b.validate_model();self.assertTrue(b.planning_allowed)

    def test_planning_batches_do_not_enter_real_replay(self):
        b=DrivingDynaDQN(3,self.settings(validation_interval=1000),seed=7)
        state=np.array([1.,.3,.5],np.float32)
        for _ in range(5):b.observe(state,0,.01,state,False)
        b.planning_allowed=True  # Isolate the imagined-update path after gate tests above.
        old_updates=b.q_updates
        b.observe(state,0,.01,state,False)
        self.assertEqual(len(b.replay),6)
        self.assertEqual(b.planning_updates,1)
        self.assertEqual(b.q_updates,old_updates+2)
        self.assertTrue(all(np.isfinite(x) for x in b.values(state)))

    def test_roundtrip_and_frozen_evaluation_and_q_guard(self):
        c=driving_config();c.highway['duration']=1.
        for policy in ['sarsa','dqn','dyna_dqn']:
            with tempfile.TemporaryDirectory() as temp:
                a,h=fit_driving(c,self.settings(),policy,7,Path(temp))
                self.assertEqual(len(h),2)
                self.assertTrue((h.max_abs_q<20).all())
                self.assertTrue((h.sensor_power_w>0).all())
                self.assertTrue((h.deadline_failure).all())
                metrics,roll=rollout_driving(c,a,80000)
                self.assertEqual(metrics['elapsed_s'],1.)
                self.assertEqual(len(roll.history),2) # Includes the forced warm-up.
        a=DrivingExpectedSarsa(3);a.weights[0,0]=21.
        with self.assertRaises(FloatingPointError):check_q(a,np.array([1.,0.,0.]),20.)

    def test_invalid_settings(self):
        for changes in [{'learning_rate':float('nan')},{'q_abort_limit':float('inf')},{'gamma':1.}]:
            with self.assertRaises(ValueError):DrivingSettings(**changes).validate()


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
