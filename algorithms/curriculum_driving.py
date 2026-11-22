"""Masked Double DQN and replay-anchored, one-step nonterminal Dyna planning.

Imagined states/rewards come from a learned model. Only real nonterminal
state/action pairs seed imagination, so their one-step ending flag is known.
Terminal experience stays in real Q replay; no learned ending probability gates
or hallucinates these short rollouts. This is not a multi-step generative model.
"""
from collections import deque
from dataclasses import asdict,dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .driving_agents import DrivingDQN,DrivingSettings,DRIVING_SCHEMA
from .dqn_agent import DQNAgent,ReplayBuffer


@dataclass
class CurriculumSettings(DrivingSettings):
    episodes:int=2000
    gamma:float=.99
    epsilon_start:float=.6
    epsilon_end:float=.05
    replay_capacity:int=50000
    learning_starts:int=256
    planning_steps:int=1
    model_learning_starts:int=1000
    model_learning_rate:float=.001
    model_hidden_size:int=128
    planning_loss_weight:float=.25
    reward_mae_limit:float=.02
    occupancy_mae_limit:float=.12
    min_model_validation:int=64
    validation_interval:int=100

    def validate(self):
        super().validate()
        if not (0<self.model_learning_rate<1 and 0<self.planning_loss_weight<=1):
            raise ValueError('Invalid model learning rate or planning weight')
        for name in ['model_hidden_size','min_model_validation']:
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:raise ValueError(name)
        if not (0<self.occupancy_mae_limit<1):raise ValueError('occupancy_mae_limit')


def action_mask(states):
    """Controller telemetry only; used for behavior and all Bellman maximizations."""
    c=states[:,-11:]
    mask=torch.ones((len(states),5),dtype=torch.bool,device=states.device)
    settled=c[:,3]<.5
    mask[:,0]=(c[:,4]>.5)&settled
    mask[:,2]=(c[:,5]>.5)&settled
    mask[:,3]=c[:,2]<(1.-1e-4)
    mask[:,4]=c[:,2]>(.5+1e-4)
    return mask


def project_next(states,actions,prediction):
    """Bound sensor features and apply known command/controller transitions."""
    nxt=prediction.clamp(0,1).clone()
    signed=[2,3]+[9+6*i for i in range(15)]
    nxt[:,signed]=prediction[:,signed].clamp(-1,1)
    nxt[:,0]=1.
    c=states[:,-11:];out=nxt[:,-11:]
    lane=(c[:,1]*2).round()
    lane=(lane-(actions==0).float()+(actions==2).float()).clamp(0,2)
    speed=(c[:,2]+((actions==3).float()-(actions==4).float())/6).clamp(.5,1.)
    out[:,1]=lane/2
    out[:,2]=speed
    out[:,3]=((out[:,0]-lane/2).abs()>.0375).float()
    out[:,4]=(lane>0).float();out[:,5]=(lane<2).float()
    out[:,6:]=F.one_hot(actions,num_classes=5).float()
    return nxt


class CurriculumDQN(DrivingDQN):
    def __init__(self,feature_count,settings=None,seed=42):
        self.curriculum=CurriculumSettings(**asdict(settings or CurriculumSettings()))
        self.curriculum.validate()
        base=DrivingSettings(**{k:asdict(self.curriculum)[k] for k in DrivingSettings.__dataclass_fields__})
        super().__init__(feature_count,base,seed)
        if feature_count!=123:raise ValueError('Curriculum controller projection requires 123 features')

    def select(self,features,epsilon,rng):
        valid=action_mask(torch.as_tensor(features,dtype=torch.float32)[None])[0].numpy()
        choices=np.flatnonzero(valid)
        if rng.random()<epsilon:return int(rng.choice(choices))
        values=self.values(features);best=values[valid].max()
        return int(rng.choice(np.flatnonzero(valid & np.isclose(values,best,rtol=0,atol=1e-12))))

    def td_targets(self,rewards,next_states,terminals):
        with torch.no_grad():
            choices=self.online(next_states).masked_fill(~action_mask(next_states),-torch.inf).argmax(1)
            values=self.target(next_states).gather(1,choices[:,None]).squeeze(1)
            return rewards+self.settings.gamma*(1-terminals)*values

    def learn_batch(self,batch,weight=1.):
        states,actions,rewards,next_states,terminals=batch
        q=self.online(states).gather(1,actions[:,None]).squeeze(1)
        target=self.td_targets(rewards,next_states,terminals)
        loss=F.smooth_l1_loss(q,target)
        self.optimizer.zero_grad();(weight*loss).backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),10.)
        self.optimizer.step();self.q_updates+=1
        if self.q_updates%self.settings.target_update_steps==0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def save(self,path):
        torch.save(dict(schema=DRIVING_SCHEMA,kind=type(self).__name__,feature_count=self.feature_count,
                        settings=asdict(self.curriculum),online=self.online.state_dict()),path)

    @classmethod
    def load(cls,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['schema']!=DRIVING_SCHEMA:raise ValueError('Incompatible driving schema')
        kind={'CurriculumDQN':CurriculumDQN,'CurriculumDynaDQN':CurriculumDynaDQN}[data['kind']]
        a=kind(data['feature_count'],CurriculumSettings(**data['settings']))
        a.online.load_state_dict(data['online']);a.target.load_state_dict(data['online'])
        if not all(torch.isfinite(p).all() for p in a.online.parameters()):raise ValueError('Invalid weights')
        return a


class CurriculumDynaDQN(CurriculumDQN):
    def __init__(self,feature_count,settings=None,seed=42):
        super().__init__(feature_count,settings,seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1)
            h=self.curriculum.model_hidden_size
            self.model=nn.Sequential(nn.Linear(feature_count+5,h),nn.ReLU(),
                                     nn.Linear(h,h),nn.ReLU(),nn.Linear(h,feature_count+1))
        self.model_optimizer=torch.optim.Adam(self.model.parameters(),lr=self.curriculum.model_learning_rate)
        self.model_rng=np.random.default_rng(seed+1)
        self.model_replay=ReplayBuffer(self.settings.replay_capacity)
        self.validation=deque(maxlen=4096)
        self.model_validation_episode=False
        self.planning_updates=0;self.real_steps=0;self.planning_allowed=False
        self.model_diagnostics={}

    def begin_episode(self,episode):
        # Entire episodes held out from model fitting; Q still learns real data.
        self.model_validation_episode=(episode%10==9)

    def predictions(self,states,actions):
        flags=F.one_hot(actions,num_classes=5).float()
        raw=self.model(torch.cat([states,flags],dim=1))
        return project_next(states,actions,states+raw[:,:self.feature_count]),raw[:,-1]

    def learn_model(self,batch):
        states,actions,rewards,next_states,_=batch
        predicted,r=self.predictions(states,actions)
        loss=F.smooth_l1_loss(predicted,next_states)+F.smooth_l1_loss(r,rewards)
        self.model_optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(),10.)
        self.model_optimizer.step()
        return float(loss.detach())

    def validate_model(self):
        if len(self.validation)<self.curriculum.min_model_validation:
            self.planning_allowed=False
            return {'model_validation_count':len(self.validation)}
        states,actions,rewards,next_states,_=zip(*self.validation)
        states=torch.tensor(np.asarray(states),dtype=torch.float32)
        actions=torch.tensor(actions,dtype=torch.long)
        rewards=torch.tensor(rewards,dtype=torch.float32)
        next_states=torch.tensor(np.asarray(next_states),dtype=torch.float32)
        with torch.no_grad():
            predicted,r=self.predictions(states,actions)
            error=(predicted-next_states).abs()
            feature=float(error.mean());reward=float((r-rewards).abs().mean())
            occupancy=float(error[:,[4+6*i for i in range(15)]].mean())
        self.model_diagnostics=dict(model_feature_mae=feature,model_reward_mae=reward,
            model_occupancy_mae=occupancy,model_validation_count=len(self.validation))
        self.planning_allowed=bool(len(self.model_replay)>=self.settings.model_learning_starts
            and feature<=self.curriculum.feature_mae_limit
            and reward<=self.curriculum.reward_mae_limit
            and occupancy<=self.curriculum.occupancy_mae_limit)
        return self.model_diagnostics.copy()

    def imagine_batch(self,batch):
        states,actions,_,_,terminals=batch
        if terminals.any():raise ValueError('One-step seeds must be real nonterminal transitions')
        with torch.no_grad():
            next_states,rewards=self.predictions(states,actions)
            # Nonterminal shaping: <=15m per 0.5s, time cost 0.001; no finish bonus.
            rewards=rewards.clamp(-.02,.02)
        return states,actions,rewards,next_states,torch.zeros_like(terminals)

    def observe(self,features,action,reward,next_features,terminal):
        # Bypass legacy Dyna behavior: real replay update is exactly the DQN path.
        metrics=DQNAgent.observe(self,features,action,reward,next_features,terminal)
        self.real_steps+=1
        if not terminal:
            if self.model_validation_episode:
                self.validation.append((features.copy(),int(action),float(reward),next_features.copy(),0.))
            else:self.model_replay.add(features,action,reward,next_features,False)
        if len(self.model_replay)>=self.settings.batch_size:
            metrics['model_loss']=self.learn_model(self.model_replay.sample(self.settings.batch_size,self.model_rng))
        if self.real_steps%self.curriculum.validation_interval==0:
            metrics.update(self.validate_model())
        if self.planning_allowed and self.settings.planning_steps:
            losses=[]
            for _ in range(self.settings.planning_steps):
                imagined=self.imagine_batch(self.model_replay.sample(self.settings.batch_size,self.model_rng))
                losses.append(self.learn_batch(imagined,self.curriculum.planning_loss_weight))
                self.planning_updates+=1
            metrics['planning_loss']=float(np.mean(losses))
        metrics['planning_enabled']=float(self.planning_allowed and self.settings.planning_steps>0)
        return metrics
