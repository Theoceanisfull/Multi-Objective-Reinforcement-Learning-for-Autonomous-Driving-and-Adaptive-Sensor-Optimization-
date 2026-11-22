"""Resumable paired 2 km DQN / Dyna-DQN curriculum, with observable planning.

Run --launch for two independent local workers. Notebook calls the same API.
A STOP file in the study directory stops both after their current episode.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from algorithms.curriculum_driving import CurriculumDQN,CurriculumDynaDQN,CurriculumSettings,action_mask
from algorithms.driving_agents import DrivingDQN,DrivingEncoder
from environments.sensor_highway.config import ExperimentConfig
from environments.sensor_highway.driving_curriculum import CurriculumEnv,SCENARIOS,episode_scenario,scenario_config
from experiments.driving_policy_helpers import ROOT,check_q
from experiments.evaluate_driving_generalization import DEFAULT_RUN,rollout,export_replay

AGENTS={'dqn':CurriculumDQN,'dyna_dqn':CurriculumDynaDQN}
SOURCE_FILES=['algorithms/curriculum_driving.py','algorithms/driving_agents.py','algorithms/dqn_agent.py',
    'algorithms/sensor_agent.py','environments/sensor_highway/driving_curriculum.py',
    'environments/sensor_highway/driving_evaluation.py','environments/sensor_highway/driving.py',
    'environments/sensor_highway/overtaking.py','environments/sensor_highway/env.py',
    'environments/sensor_highway/backend.py','environments/sensor_highway/sensors.py',
    'environments/sensor_highway/grid.py','environments/sensor_highway/weather.py',
    'environments/sensor_highway/config.py',
    'experiments/train_driving_curriculum.py','experiments/evaluate_driving_generalization.py']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def write_csv(path,frame):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(tmp,index=False);tmp.replace(path)


def create_study(settings=None,output=None,source=DEFAULT_RUN,seed=42,train_seed_start=2000000,
                 route_m=2000.,deadline_s=90.,eval_every=250,eval_seeds=3,final_seeds=10,replays=True):
    settings=settings or CurriculumSettings();settings.validate()
    if route_m<=0 or deadline_s<=.5:raise ValueError('Invalid route/deadline')
    if type(seed) is not int or type(train_seed_start) is not int:raise ValueError('Integer seeds required')
    if eval_every<=0 or eval_seeds<0 or final_seeds<0:raise ValueError('Invalid evaluation budget')
    source=Path(source).resolve()
    if settings.hidden_size!=64:raise ValueError('Warm-start checkpoint uses 64 hidden units')
    reserved=set(range(400000,400000+eval_seeds))|set(range(500000,500000+final_seeds))
    training=set(range(train_seed_start,train_seed_start+settings.episodes))
    old=json.loads((source/'metadata.json').read_text())
    old_training=set(range(old['seed'],old['seed']+old['settings']['episodes']))
    if training&(reserved|old_training):raise ValueError('Fresh training and evaluation seeds must be disjoint')
    if output is None:
        stamp=datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S_%f')
        output=ROOT/'runs/local/driving_curriculum_2000m'/stamp
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    config=ExperimentConfig.load(source/'config.json');config.save(output/'base_config.json')
    meta=dict(settings=asdict(settings),source_run=str(source),seed=seed,train_seed_start=train_seed_start,
        route_m=float(route_m),deadline_s=float(deadline_s),eval_every=eval_every,eval_seeds=eval_seeds,
        final_seeds=final_seeds,replays=replays,policies=list(AGENTS),scenarios=list(SCENARIOS),
        total_additional_episodes=settings.episodes*2,common_warm_start='dqn/agent.pt',
        warm_start_sha256=digest(source/'dqn/agent.pt'),
        source_sha256={p:digest(ROOT/p) for p in SOURCE_FILES},python=sys.version,
        packages={name:importlib.metadata.version(name) for name in ['torch','numpy','pandas','gymnasium','highway-env']},
        notes=['Both policies start from the same saved 1000-episode DQN Q weights; optimizers/replay are fresh.',
            'All sensors on; energy/uncertainty rewards remain zero. Gamma is 0.99.',
            'Model rollouts are one-step, seeded only by real nonterminal transitions with the same observed action.',
            'A learned model predicts sensor/ego features and nonterminal shaping reward. Controller commands are analytic.',
            'Terminal events and bonuses/penalties are learned from real Q replay only; no learned termination model is used.',
            'Every tenth entire episode is reserved from model fitting, not from real Q learning.',
            'One imagined update per real step after warm-up and validation; imagined loss weight 0.25.',
            'Action masks apply to behavior, exploration and real/imagined Bellman targets.',
            'Independent policy workers; paired scenario schedule and traffic seeds, one initialization seed.',
            'Development seeds start 400000; final held-out seeds start 500000; old 300000 evaluation seeds are not reused.'])
    write_json(output/'protocol.json',meta)
    for path in SOURCE_FILES:
        archived=output/'sources'/path
        archived.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,archived)
    for name in AGENTS:
        (output/name).mkdir()
        write_json(output/name/'status.json',dict(phase='pending',episode=0,episodes=settings.episodes))
    print(f'Study: {output}\nAdditional episodes: {settings.episodes} per model, {settings.episodes*2} total',flush=True)
    return output


def read_protocol(output):
    output=Path(output);meta=json.loads((output/'protocol.json').read_text())
    for path,expected in meta['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError(f'Source changed since study creation: {path}')
    if digest(Path(meta['source_run'])/meta['common_warm_start'])!=meta['warm_start_sha256']:
        raise ValueError('Source warm-start checkpoint changed')
    return meta


def initialize_agent(policy,meta):
    settings=CurriculumSettings(**meta['settings'])
    agent=AGENTS[policy](123,settings,meta['seed'])
    old=DrivingDQN.load(Path(meta['source_run'])/meta['common_warm_start'])
    agent.online.load_state_dict(old.online.state_dict());agent.target.load_state_dict(old.online.state_dict())
    for a,b in zip(agent.online.parameters(),old.online.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    return agent


def checkpoint(folder,agent,rng,history,episode,meta):
    # Full trusted local training state, written atomically for crash-safe resume.
    path=folder/'resume.pkl';tmp=path.with_suffix('.pkl.tmp')
    with tmp.open('wb') as f:
        pickle.dump(dict(agent=agent,rng=rng,history=history,episode=episode,
            torch_rng=torch.get_rng_state(),protocol_sha256=digest(folder.parent/'protocol.json')),f)
    tmp.replace(path)
    portable=folder/'agent.pt.tmp';agent.save(portable);portable.replace(folder/'agent.pt')
    if history:write_csv(folder/'training.csv',pd.DataFrame(history))


def evaluate(output,policy,agent,episode,final=False):
    meta=read_protocol(output);folder=Path(output)/policy
    source=ExperimentConfig.load(Path(output)/'base_config.json')
    count=meta['final_seeds'] if final else meta['eval_seeds']
    start=500000 if final else 400000
    if count==0:return pd.DataFrame()
    rows=[]
    batches=[(name,'clear') for name in SCENARIOS]
    if final:batches += [('repeated_medium',w) for w in ['rain','fog']]
    for name,weather in batches:
        config=scenario_config(source,name,weather,meta['route_m'],meta['deadline_s'])
        for seed in range(start,start+count):
            if (Path(output)/'STOP').exists() or (folder/'STOP').exists():
                raise InterruptedError('Stop requested during evaluation; checkpoint already saved')
            metrics,_=rollout(config,policy,agent,seed,env_class=CurriculumEnv)
            rows.append(dict(episode=episode,scenario=name,weather=weather,**metrics))
    frame=pd.DataFrame(rows)
    path=folder/('final_evaluation.csv' if final else 'development_evaluation.csv')
    if not final and path.exists():
        old=pd.read_csv(path);old=old[old.episode!=episode]
        frame=pd.concat([old,frame],ignore_index=True)
    write_csv(path,frame)
    summary=frame.groupby(['episode','scenario','weather'],sort=False).agg(
        trips=('completed','size'),completions=('completed','sum'),crashes=('crashed','sum'),
        deadline_failures=('deadline_failure','sum'),mean_finish_s=('completion_time_s','mean'),
        mean_overtakes=('overtakes','mean'),max_overtakes=('overtakes','max'),
        mean_lane_changes=('lane_changes','mean'),max_abs_q=('max_abs_q','max')).reset_index()
    write_csv(folder/('final_summary.csv' if final else 'development_summary.csv'),summary)
    if final and meta['replays']:
        candidates=frame[frame.weather=='clear']
        row=candidates.sort_values(['completed','distance_m'],ascending=[False,False]).iloc[0]
        config=scenario_config(source,row.scenario,row.weather,meta['route_m'],meta['deadline_s'])
        export_replay(config,policy,agent,int(row.seed),folder/'replay',row,env_class=CurriculumEnv)
    return frame


def train_one(output,policy):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output=Path(output).resolve();meta=read_protocol(output);folder=output/policy
    settings=CurriculumSettings(**meta['settings']);settings.validate()
    resume=folder/'resume.pkl'
    if resume.exists():
        with resume.open('rb') as f:state=pickle.load(f)
        if state['protocol_sha256']!=digest(output/'protocol.json'):raise ValueError('Resume protocol changed')
        agent=state['agent'];rng=state['rng'];history=state['history'];start=state['episode']
        torch.set_rng_state(state['torch_rng'])
    else:
        agent=initialize_agent(policy,meta);rng=np.random.default_rng(meta['seed']);history=[];start=0
        checkpoint(folder,agent,rng,history,0,meta)
    source=ExperimentConfig.load(output/'base_config.json')
    wall_start=time.perf_counter();wall_base=history[-1]['training_wall_s'] if history else 0.
    for episode in range(start,settings.episodes):
        if (output/'STOP').exists() or (folder/'STOP').exists():
            checkpoint(folder,agent,rng,history,episode,meta)
            write_json(folder/'status.json',dict(phase='stopped',episode=episode,episodes=settings.episodes,
                planning_updates=getattr(agent,'planning_updates',0)));return
        name,weather,stage=episode_scenario(episode,settings.episodes,meta['seed'])
        config=scenario_config(source,name,weather,meta['route_m'],meta['deadline_s'])
        epsilon=settings.epsilon_start+(settings.epsilon_end-settings.epsilon_start)*episode/max(1,settings.episodes-1)
        if hasattr(agent,'begin_episode'):agent.begin_episode(episode)
        diagnostics=[];max_q=0.;steps=0
        with CurriculumEnv(config) as env:
            obs,info=env.reset(seed=meta['train_seed_start']+episode)
            encoder=DrivingEncoder(config);features=encoder(obs);total=info['warmup_reward']
            initial_lane=int(env.backend.vehicle.lane_index[2])
            while True:
                max_q=max(max_q,check_q(agent,features,settings.q_abort_limit))
                action=agent.select(features,epsilon,rng)
                if not bool(action_mask(torch.tensor(features)[None])[0,action]):raise AssertionError('Invalid controller action')
                obs,reward,terminated,truncated,info=env.step(action)
                nxt=encoder(obs);total+=reward;steps+=1
                diagnostics.append(agent.observe(features,action,reward,nxt,terminated or truncated))
                features=nxt
                if terminated or truncated:break
            max_q=max(max_q,check_q(agent,features,settings.q_abort_limit))
        planning=getattr(agent,'planning_updates',0)
        row=dict(episode=episode+1,scenario=name,weather=weather,stage=stage,
            traffic_seed=meta['train_seed_start']+episode,initial_lane=initial_lane,epsilon=epsilon,
            return_=total,steps=steps,completed=info['route_completed'],crashed=info['crashed'],
            deadline_failure=bool(truncated and not info['route_completed']),elapsed_s=info['time_s'],
            distance_m=info['distance_m'],overtakes=info['overtakes'],lane_changes=info['lane_changes'],
            sensor_power_w=info['total_energy_j']/info['time_s'],max_abs_q=max_q,
            q_updates=agent.q_updates,real_q_updates=agent.q_updates-planning,
            planning_updates=planning,imagined_transitions=planning*settings.batch_size,
            training_wall_s=wall_base+time.perf_counter()-wall_start,
            **{k:float(np.mean([d[k] for d in diagnostics if k in d]))
               for k in set().union(*diagnostics)})
        row.update(getattr(agent,'model_diagnostics',{}))
        history.append(row)
        write_csv(folder/'training.csv',pd.DataFrame(history))
        if (episode+1)%25==0 or episode+1==settings.episodes:
            checkpoint(folder,agent,rng,history,episode+1,meta)
            print(f'{policy} {episode+1}/{settings.episodes}: {name}, complete={info["route_completed"]}, '
                  f'max|Q|={max_q:.3f}, planning={planning}, wall={row["training_wall_s"]:.1f}s',flush=True)
        write_json(folder/'status.json',dict(phase='training',episode=episode+1,episodes=settings.episodes,
            scenario=name,weather=weather,stage=stage,planning_updates=planning,
            planning_allowed=getattr(agent,'planning_allowed',False),max_abs_q=max_q,
            training_wall_s=row['training_wall_s'],model_diagnostics=getattr(agent,'model_diagnostics',{}),pid=os.getpid()))
        if (episode+1)%meta['eval_every']==0 and episode+1<settings.episodes:
            write_json(folder/'status.json',dict(phase='development_evaluation',episode=episode+1,
                episodes=settings.episodes,planning_updates=planning,pid=os.getpid()))
            evaluate(output,policy,agent,episode+1)
    if policy=='dyna_dqn' and settings.planning_steps and agent.planning_updates==0:
        raise RuntimeError('No planning updates occurred; inspect held-out model diagnostics before claiming Dyna training')
    write_json(folder/'status.json',dict(phase='final_evaluation',episode=settings.episodes,
        episodes=settings.episodes,planning_updates=getattr(agent,'planning_updates',0),pid=os.getpid()))
    evaluate(output,policy,agent,settings.episodes,final=True)
    write_json(folder/'status.json',dict(phase='complete',episode=settings.episodes,episodes=settings.episodes,
        planning_updates=getattr(agent,'planning_updates',0),real_q_updates=agent.q_updates-getattr(agent,'planning_updates',0),
        training_wall_s=wall_base+time.perf_counter()-wall_start,pid=os.getpid()))
    print(f'{policy}: complete; final planning updates={getattr(agent,"planning_updates",0)}',flush=True)


def launch_workers(output):
    output=Path(output).resolve();meta=read_protocol(output);workers={}
    for policy in AGENTS:
        folder=output/policy
        status=json.loads((folder/'status.json').read_text())
        if status['phase']=='complete':continue
        pid_path=folder/'worker.pid'
        if pid_path.exists():
            try:os.kill(int(pid_path.read_text()),0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'{policy} worker already running')
        with (folder/'worker.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u','-m','experiments.train_driving_curriculum',
                '--output',str(output),'--worker',policy],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
        pid_path.write_text(str(process.pid)+'\n');workers[policy]=process.pid
    write_json(output/'workers.json',dict(pids=workers,total_additional_episodes=meta['total_additional_episodes']))
    print(json.dumps(workers),flush=True)
    return workers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--episodes',type=int,default=2000)
    p.add_argument('--source',type=Path,default=DEFAULT_RUN)
    p.add_argument('--launch',action='store_true')
    p.add_argument('--worker',choices=list(AGENTS))
    p.add_argument('--pilot',action='store_true')
    args=p.parse_args()
    if args.worker:
        if args.output is None:p.error('--worker requires --output')
        try:train_one(args.output,args.worker)
        except InterruptedError:
            path=args.output/args.worker/'status.json'
            status=json.loads(path.read_text());status.update(phase='stopped',pid=os.getpid())
            write_json(path,status)
        except Exception as error:
            write_json(args.output/args.worker/'status.json',dict(phase='failed',error=str(error),pid=os.getpid()))
            traceback.print_exc();raise
        return
    output=args.output
    if output is None or not (output/'protocol.json').exists():
        settings=CurriculumSettings(episodes=args.episodes)
        output=create_study(settings,output,args.source,eval_seeds=0 if args.pilot else 3,
                           final_seeds=0 if args.pilot else 10,replays=not args.pilot)
    if args.launch:launch_workers(output)
    else:
        for policy in AGENTS:train_one(output,policy)


if __name__=='__main__':main()
