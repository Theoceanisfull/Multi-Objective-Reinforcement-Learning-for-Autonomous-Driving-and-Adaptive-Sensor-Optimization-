"""Bounded, explicit driving-only training and frozen evaluation for the notebook."""
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch

from algorithms.driving_agents import (DrivingEncoder,DrivingExpectedSarsa,DrivingDQN,
                                      DrivingDynaDQN,DrivingSettings)
from environments.sensor_highway.driving import DrivingEnv
from experiments.notebook_helpers import Rollout

AGENTS={'sarsa':DrivingExpectedSarsa,'dqn':DrivingDQN,'dyna_dqn':DrivingDynaDQN}
LABELS={'sarsa':'Expected SARSA (linear)','dqn':'DQN (Double target)','dyna_dqn':'Dyna-DQN (gated planning)'}
ROOT=Path(__file__).resolve().parents[1]


def new_output(config,settings,profile,seed):
    stamp=datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S_%f')
    output=ROOT/'runs/local/driving_policy'/f'{profile}_{stamp}'
    output.mkdir(parents=True)
    config.save(output/'config.json')
    files=['environments/sensor_highway/driving.py','algorithms/driving_agents.py',
           'experiments/driving_policy_helpers.py','algorithms/sensor_agent.py','algorithms/dqn_agent.py']
    metadata={'profile':profile,'settings':asdict(settings),'seed':seed,'config':asdict(config),
              'sources_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
              'note':'Always-on, real sensor observations. Pilot results are integration checks, not convergence.'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return output


def check_q(agent,features,limit):
    values=agent.values(features)
    if not np.isfinite(values).all() or np.abs(values).max()>limit:
        raise FloatingPointError(f'Q-value stability guard: {values}; inspect before extending training')
    return float(np.abs(values).max())


def fit_driving(config,settings,policy,seed,folder):
    settings.validate();torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if policy not in AGENTS:raise ValueError(policy)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training.csv').exists():raise FileExistsError('Use a new output directory for fresh training')
    rng=np.random.default_rng(seed);encoder=DrivingEncoder(config)
    history=[];start=time.perf_counter()
    with DrivingEnv(config) as env:
        obs,_=env.reset(seed=seed)
        count=len(encoder(obs))
        agent=AGENTS[policy](count) if policy=='sarsa' else AGENTS[policy](count,settings,seed)
        for episode in range(settings.episodes):
            epsilon=settings.epsilon_start+(settings.epsilon_end-settings.epsilon_start)*episode/max(1,settings.episodes-1)
            obs,info=env.reset(seed=seed+episode);features=encoder(obs)
            total=float(info['warmup_reward']);diagnostics=[];q_magnitudes=[];steps=0
            while True:
                q_magnitudes.append(check_q(agent,features,settings.q_abort_limit))
                index=agent.select(features,epsilon,rng)
                obs,reward,terminated,truncated,info=env.step(index)
                next_features=encoder(obs)
                terminal=terminated or truncated
                if policy=='sarsa':
                    error=agent.update(features,index,reward,next_features,terminal,
                                       epsilon,settings.alpha,settings.gamma)
                    metrics={'abs_td':abs(error)}
                else:metrics=agent.observe(features,index,reward,next_features,terminal)
                diagnostics.append(metrics);total+=reward;steps+=1;features=next_features
                if terminated or truncated:break
            q_magnitudes.append(check_q(agent,features,settings.q_abort_limit))
            history.append({'episode':episode+1,'seed':seed+episode,'epsilon':epsilon,
                'return':total,'steps':steps,'completed':info['route_completed'],'crashed':info['crashed'],
                'deadline_failure':bool(truncated and not info['route_completed']),
                'elapsed_s':info['time_s'],'distance_m':info['distance_m'],'overtakes':info['overtakes'],
                'sensor_power_w':info['total_energy_j']/info['time_s'],
                'max_abs_q':max(q_magnitudes),'training_wall_s':time.perf_counter()-start,
                'q_updates':getattr(agent,'q_updates',steps),
                'planning_updates':getattr(agent,'planning_updates',0),
                **{k:float(np.mean([v[k] for v in diagnostics if k in v]))
                   for k in set().union(*diagnostics)}})
            if (episode+1)%25==0 or episode+1==settings.episodes:
                frame=pd.DataFrame(history);frame.to_csv(folder/'training.csv',index=False)
                agent.save(folder/('agent.npz' if policy=='sarsa' else 'agent.pt'))
                with (folder/'resume.pkl').open('wb') as f:
                    pickle.dump({'agent':agent,'rng':rng,'history':history,'settings':settings,
                                 'episode':episode+1,'torch_rng':torch.get_rng_state()},f)
                print(f'{LABELS[policy]}: {episode+1}/{settings.episodes}, return={total:.3f}, '
                      f'max|Q|={max(q_magnitudes):.3f}, wall={time.perf_counter()-start:.1f}s',flush=True)
    # Portable policy round trip; full training state is also saved locally.
    restored=(DrivingExpectedSarsa.load(folder/'agent.npz') if policy=='sarsa'
              else DrivingDQN.load(folder/'agent.pt'))
    np.testing.assert_array_equal(agent.values(features),restored.values(features))
    return agent,pd.DataFrame(history)


def rollout_driving(config,agent,seed,capture=False):
    encoder=DrivingEncoder(config);rng=np.random.default_rng(seed+1000000)
    records=[];frames=[];snapshots=[]
    before=([agent.weights.copy()] if isinstance(agent,DrivingExpectedSarsa)
            else [p.detach().clone() for p in agent.online.parameters()])
    with DrivingEnv(config,render_mode='rgb_array' if capture else None) as env:
        obs,info=env.reset(seed=seed);total=info['warmup_reward'];steps=0
        records.append({'time_s':info['time_s'],'drive':'IDLE (warmup)','sensing':'all_on',
            'reward':info['warmup_reward'],'speed_m_s':float(obs['ego'][0]),
            'distance_m':info['distance_m'],'total_energy_j':info['total_energy_j'],
            'overtakes':info['overtakes'],'lane_changes':info['lane_changes'],
            'crashed':False,'completed':False,'warmup':True})
        if capture:
            snapshots.extend([env.prelude_observation,obs])
            frames.extend([env.prelude_frame,env.render()])
        max_q=0.
        while True:
            features=encoder(obs)
            max_q=max(max_q,check_q(agent,features,getattr(getattr(agent,'options',None),'q_abort_limit',20.)))
            index=agent.select(features,0.,rng)
            obs,reward,terminated,truncated,info=env.step(index);total+=reward;steps+=1
            records.append({'time_s':info['time_s'],'drive':agent.actions[index][0],
                'sensing':'all_on','reward':reward,'speed_m_s':float(obs['ego'][0]),
                'distance_m':info['distance_m'],'total_energy_j':info['total_energy_j'],
                'overtakes':info['overtakes'],'lane_changes':info['lane_changes'],
                'crashed':info['crashed'],'completed':info['route_completed'],'warmup':False})
            if capture:snapshots.append(obs);frames.append(env.render())
            if terminated or truncated:break
        metrics={'seed':seed,'return':total,'steps':steps,'completed':info['route_completed'],
            'crashed':info['crashed'],'deadline_failure':bool(truncated and not info['route_completed']),
            'elapsed_s':info['time_s'],'completion_time_s':info['time_s'] if info['route_completed'] else np.nan,
            'distance_m':info['distance_m'],'overtakes':info['overtakes'],'lane_changes':info['lane_changes'],
            'sensor_power_w':info['total_energy_j']/info['time_s'],'max_abs_q':max_q,
            'energy_j':info['total_energy_j']}
        roll=Rollout(pd.DataFrame(records),snapshots,frames,env.sensor_names,config.grid.radius_m)
    after=([agent.weights] if isinstance(agent,DrivingExpectedSarsa) else list(agent.online.parameters()))
    for a,b in zip(before,after):
        if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
        else:torch.testing.assert_close(a,b,rtol=0,atol=0)
    return metrics,roll


def save_driving_replay(config,agent,seed,folder,label):
    from experiments.replay_agents import replay_frames,HTML
    folder=Path(folder);folder.mkdir(exist_ok=True)
    metrics,roll=rollout_driving(config,agent,seed,capture=True)
    data,gifs=replay_frames(roll,metrics,label)
    html=HTML.replace('Matched agent replays','Driving-only replay').replace(
        'Frozen policies from the 1,000-episode run. Same starting scenario; actions produce different traffic interactions.',
        'Driving-only policy with every sensor on. Episode timing includes a forced 0.5 s warm-up. Pilot models are not converged policies.')
    replay={'seed':seed,'weather':config.weather.preset,'route_m':config.reward.route_distance_m,
        'deadline_s':config.highway['duration'],'step_s':.5,'sensor_names':roll.sensor_names,
        'agents':[{'id':'policy','label':label,'metrics':metrics,'frames':data}]}
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        return None if isinstance(value,float) and not np.isfinite(value) else value
    (folder/'replay.html').write_text(html.replace('__DATA__',json.dumps(clean(replay),allow_nan=False)))
    durations=[500]*len(gifs);durations[-1]=2000
    gifs[0].save(folder/'policy.gif',save_all=True,append_images=gifs[1:],duration=durations,loop=0)
    roll.history.to_csv(folder/'policy_history.csv',index=False)
    (folder/'metrics.json').write_text(json.dumps(clean(metrics),indent=2)+'\n')
    return metrics


def compare_driving(config,settings,output,seed=42,test_seeds=range(100000,100005),replays=True):
    test_seeds=list(test_seeds)
    if set(range(seed,seed+settings.episodes))&set(test_seeds):raise ValueError('Training/test seeds overlap')
    output=Path(output);histories=[];evaluations=[];agents={}
    for policy in AGENTS:
        folder=output/policy
        agent,history=fit_driving(config,settings,policy,seed,folder)
        agents[policy]=agent;histories.append(history.assign(policy=LABELS[policy]))
        rows=[{'policy':LABELS[policy],**rollout_driving(config,agent,s)[0]} for s in test_seeds]
        result=pd.DataFrame(rows);result.to_csv(folder/'evaluation.csv',index=False);evaluations.append(result)
        if replays:
            replay=save_driving_replay(config,agent,test_seeds[0],folder/'replay',LABELS[policy])
            for k in ['return','distance_m','elapsed_s','completed','crashed']:
                np.testing.assert_allclose(replay[k],result.iloc[0][k],rtol=1e-10,atol=1e-8)
    training=pd.concat(histories,ignore_index=True);evaluation=pd.concat(evaluations,ignore_index=True)
    training.to_csv(output/'training.csv',index=False);evaluation.to_csv(output/'evaluation.csv',index=False)
    summary=evaluation.groupby('policy',sort=False).agg(trips=('completed','size'),
        completion=('completed','mean'),crash=('crashed','mean'),deadline_failure=('deadline_failure','mean'),
        successful_time_s=('completion_time_s','mean'),mean_distance_m=('distance_m','mean'),
        mean_overtakes=('overtakes','mean'),max_abs_q=('max_abs_q','max'))
    summary.to_csv(output/'summary.csv')
    return agents,training,evaluation,summary
