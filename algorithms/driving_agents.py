"""Driving-only adapters; preserve the earlier SARSA/DQN implementations.

DQN uses an optional Double-DQN target. Dyna separates real/model replay random
streams and gates planning on held-out model error and terminal coverage.
"""
from collections import deque
from dataclasses import asdict,dataclass
import math

import numpy as np
import torch
from torch import nn

from algorithms.sensor_agent import ObservationEncoder,LinearExpectedSarsa
from algorithms.dqn_agent import DQNAgent,DynaDQNAgent,DQNSettings,ReplayBuffer,make_network

DRIVING_SCHEMA='driving_sensor_grid_controller_v1'


class DrivingEncoder(ObservationEncoder):
    def __init__(self,config):
        super().__init__(config)
        self.feature_schema=DRIVING_SCHEMA

    def __call__(self,obs):
        return np.concatenate([super().__call__(obs),obs['controller']]).astype(np.float32)


@dataclass
class DrivingSettings:
    episodes:int=500
    gamma:float=.99
    alpha:float=.12
    epsilon_start:float=1.
    epsilon_end:float=.1
    learning_rate:float=.0001
    hidden_size:int=64
    batch_size:int=32
    replay_capacity:int=10000
    learning_starts:int=128
    target_update_steps:int=100
    planning_steps:int=1
    model_learning_starts:int=1000
    double_q:bool=True
    validation_interval:int=50
    min_validation_terminals:int=5
    feature_mae_limit:float=.05
    reward_mae_limit:float=.1
    terminal_mae_limit:float=.15
    q_abort_limit:float=20.

    def base(self):
        self.validate()
        return DQNSettings(**{k:asdict(self)[k] for k in DQNSettings.__dataclass_fields__})

    def validate(self):
        if type(self.episodes) is not int or self.episodes<=0:raise ValueError('episodes must be positive')
        if not (0<self.alpha<=1 and math.isfinite(self.q_abort_limit) and self.q_abort_limit>0):
            raise ValueError('Invalid alpha or Q guard')
        for name in ['validation_interval','min_validation_terminals']:
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:raise ValueError(name)
        for name in ['feature_mae_limit','reward_mae_limit','terminal_mae_limit']:
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError(name)
        DQNSettings(**{k:asdict(self)[k] for k in DQNSettings.__dataclass_fields__}).validate()


class DrivingExpectedSarsa(LinearExpectedSarsa):
    def __init__(self,feature_count):
        super().__init__(feature_count,modes=('all_on',),feature_schema=DRIVING_SCHEMA)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            if str(data['feature_schema'])!=DRIVING_SCHEMA:raise ValueError('Incompatible driving features')
            agent=cls(data['weights'].shape[1]);agent.weights[:]=data['weights']
            if not np.isfinite(agent.weights).all():raise ValueError('Invalid weights')
        return agent


class DrivingDQN(DQNAgent):
    def __init__(self,feature_count,settings=None,seed=42):
        self.options=DrivingSettings(**asdict(settings or DrivingSettings()))
        super().__init__(feature_count,self.options.base(),modes=('all_on',),
                         feature_schema=DRIVING_SCHEMA,seed=seed)

    def td_targets(self,rewards,next_states,terminals):
        with torch.no_grad():
            if self.options.double_q:
                actions=self.online(next_states).argmax(dim=1)
                next_q=self.target(next_states).gather(1,actions[:,None]).squeeze(1)
            else:next_q=self.target(next_states).max(dim=1).values
            return rewards+self.settings.gamma*(1-terminals)*next_q

    def save(self,path):
        torch.save({'schema':DRIVING_SCHEMA,'kind':type(self).__name__,
                    'feature_count':self.feature_count,'options':asdict(self.options),
                    'online':self.online.state_dict()},path)

    @classmethod
    def load(cls,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['schema']!=DRIVING_SCHEMA:raise ValueError('Incompatible driving features')
        classes={'DrivingDQN':DrivingDQN,'DrivingDynaDQN':DrivingDynaDQN}
        agent=classes[data['kind']](data['feature_count'],DrivingSettings(**data['options']))
        agent.online.load_state_dict(data['online']);agent.target.load_state_dict(data['online'])
        if not all(torch.isfinite(p).all() for p in agent.online.parameters()):raise ValueError('Invalid weights')
        return agent


class DrivingDynaDQN(DrivingDQN):
    model_predictions=DynaDQNAgent.model_predictions
    learn_model=DynaDQNAgent.learn_model
    imagine_batch=DynaDQNAgent.imagine_batch

    def __init__(self,feature_count,settings=None,seed=42):
        super().__init__(feature_count,settings,seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1)
            self.model=make_network(feature_count+len(self.actions),feature_count+2,self.settings.hidden_size)
        self.model_optimizer=torch.optim.Adam(self.model.parameters(),lr=self.settings.learning_rate)
        self.model_rng=np.random.default_rng(seed+1)
        self.model_replay=ReplayBuffer(self.settings.replay_capacity)
        self.validation=deque(maxlen=512)
        self.real_steps=0;self.planning_updates=0;self.planning_allowed=False
        self.model_diagnostics={}

    def validate_model(self):
        if not self.validation:return {}
        states,actions,rewards,next_states,terminals=zip(*self.validation)
        states=torch.tensor(np.asarray(states),dtype=torch.float32)
        actions=torch.tensor(actions,dtype=torch.long)
        rewards=torch.tensor(rewards,dtype=torch.float32)
        next_states=torch.tensor(np.asarray(next_states),dtype=torch.float32)
        terminals=torch.tensor(terminals,dtype=torch.float32)
        with torch.no_grad():
            changes,r,logits=self.model_predictions(states,actions)
            feature_error=float((changes-(next_states-states)).abs().mean())
            reward_error=float((r-rewards).abs().mean())
            pos=terminals>.5;neg=~pos
            terminal_error=max(float((logits.sigmoid()[mask]-terminals[mask]).abs().mean())
                               for mask in [pos,neg] if mask.any())
        self.model_diagnostics={'model_feature_mae':feature_error,'model_reward_mae':reward_error,
                                'model_terminal_mae':terminal_error,'validation_terminals':int(pos.sum())}
        self.planning_allowed=bool(len(self.model_replay)>=self.settings.model_learning_starts
            and int(pos.sum())>=self.options.min_validation_terminals and int(neg.sum())>=20
            and feature_error<=self.options.feature_mae_limit
            and reward_error<=self.options.reward_mae_limit
            and terminal_error<=self.options.terminal_mae_limit)
        return self.model_diagnostics.copy()

    def observe(self,features,action,reward,next_features,terminal):
        metrics=super().observe(features,action,reward,next_features,terminal)
        self.real_steps+=1
        item=(np.array(features,dtype=np.float32,copy=True),int(action),float(reward),
              np.array(next_features,dtype=np.float32,copy=True),float(terminal))
        if self.real_steps%10==0:self.validation.append(item)
        else:self.model_replay.add(*item)
        if len(self.model_replay)>=self.settings.batch_size:
            metrics['model_loss']=self.learn_model(self.model_replay.sample(self.settings.batch_size,self.model_rng))
        if self.real_steps%self.options.validation_interval==0:metrics.update(self.validate_model())
        if metrics and self.planning_allowed and self.settings.planning_steps:
            losses=[]
            for _ in range(self.settings.planning_steps):
                batch=self.model_replay.sample(self.settings.batch_size,self.model_rng)
                imagined=list(self.imagine_batch(batch))
                imagined[2]=imagined[2].clamp(-2.1,2.1)
                losses.append(self.learn_batch(tuple(imagined)))
                self.planning_updates+=1
            metrics['planning_loss']=float(np.mean(losses))
        metrics['planning_enabled']=float(self.planning_allowed and self.settings.planning_steps>0)
        return metrics
