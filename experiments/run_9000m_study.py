"""Checkpointed six-agent 9 km overtaking study; run from final_project.

python -m experiments.run_9000m_study --prepare --validate --pilot
python -m experiments.run_9000m_study --launch
python -m experiments.run_9000m_study --status
Local resume.pkl files contain trusted Python training state (not portable policies).
"""
import argparse
import copy
import fcntl
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from algorithms.sensor_agent import ObservationEncoder, LinearExpectedSarsa, AgentSettings
from algorithms.dqn_agent import DQNAgent, DynaDQNAgent, DQNSettings
from algorithms.snn import AdaptiveExpectedSarsa, AdaptiveSNNSettings
from environments.sensor_highway import ExperimentConfig
from environments.sensor_highway.overtaking import OvertakingSensorEnv, oracle_action
from experiments.notebook_helpers import Rollout

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'runs/local/overtaking_9000m_5000ep'
POLICIES = {'sarsa':'Expected SARSA', 'dqn':'DQN', 'dyna_dqn':'Dyna-DQN',
            'adaptive_reservoir':'Adaptive reservoir',
            'prototype_control':'Prototype control (no STDP)',
            'stdp_pretrained':'STDP-pretrained prototypes'}
TRAIN_SEEDS = [42,10042,20042]  # Disjoint 5,000-episode training blocks.
TEST_SEEDS = list(range(100000,100020))
WEATHERS = ['clear','rain','fog']
SOURCE_PATHS = ['experiments/run_9000m_study.py','environments/sensor_highway/overtaking.py',
                'algorithms/sensor_agent.py','algorithms/dqn_agent.py','algorithms/snn.py',
                'experiments/notebook_helpers.py'] + [f'environments/sensor_highway/{p}.py'
                for p in ['backend','config','env','grid','sensors','weather']]


def timestamp():
    return datetime.now(ZoneInfo('America/New_York')).isoformat()


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def atomic_csv(path, frame):
    path = Path(path)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    frame.to_csv(temp,index=False)
    os.replace(temp,path)


def make_config():
    c = ExperimentConfig()
    c.highway.update(duration=360,lanes_count=3,high_speed_reward=0.,collision_reward=0.)
    c.highway['action']['target_speeds'] = [10,15,20,25,30]
    c.highway['overtaking'] = {'spacing_m':500.,'speed_min_m_s':18.,'speed_max_m_s':22.,
        'spawn_ahead_m':600.,'entry_min_gap_m':300.,'remove_behind_m':100.}
    c.weather.preset = 'random'
    c.reward.route_distance_m = 9000.
    c.reward.completion_bonus = 100.
    c.reward.early_completion_bonus = 25.
    c.reward.failure_penalty = 200.
    c.validate()
    return c


def prepare(output=DEFAULT_OUTPUT, episodes=5000, seeds=TRAIN_SEEDS, pilot=False, checkpoint_every=25):
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    protocol = {'episodes':episodes,'training_seeds':list(seeds),'policies':POLICIES,
        'test_seeds':[80000,80001] if pilot else TEST_SEEDS,
        'weather':['clear'] if pilot else WEATHERS,
        'gamma':.999,'epsilon_start':1.,'epsilon_end':.1,
        'pretraining_episodes':3 if pilot else 20,'pretraining_passes':3,
        'pretraining_seed_offset':50000,'checkpoint_every_episodes':checkpoint_every,
        'pilot':pilot,'config':asdict(make_config()),
        'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}}
    import importlib.metadata
    protocol['package_versions'] = {p:importlib.metadata.version(p) for p in
                                    ['torch','numpy','pandas','gymnasium','highway-env']}
    h = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    path = output/'protocol.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_hash'] != h:
            raise ValueError('Protocol/source changed: use a new output directory to protect results')
    else:
        atomic_json(path,{**protocol,'protocol_hash':h,'created_at':timestamp()})
        make_config().save(output/'config.json')
    return output


def read_protocol(output):
    protocol = json.loads((Path(output)/'protocol.json').read_text())
    for path,h in protocol['source_hashes'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=h:
            raise ValueError(f'Implementation changed since preparation: {path}')
    if asdict(ExperimentConfig.load(Path(output)/'config.json')) != protocol['config']:
        raise ValueError('Saved configuration differs from the study protocol')
    return protocol


def make_agent(policy, features, schema, seed, protocol):
    common = dict(episodes=protocol['episodes'],gamma=protocol['gamma'],
                  epsilon_start=protocol['epsilon_start'],epsilon_end=protocol['epsilon_end'])
    if policy=='sarsa':
        return LinearExpectedSarsa(features,feature_schema=schema)
    if policy in ['dqn','dyna_dqn']:
        cls = DynaDQNAgent if policy=='dyna_dqn' else DQNAgent
        return cls(features,DQNSettings(**common),feature_schema=schema,seed=seed)
    return AdaptiveExpectedSarsa(features,AdaptiveSNNSettings(**common),
                                 feature_schema=schema,seed=seed,variant=policy)


def collect_corpus(config, seed, protocol, folder):
    path = folder/'features.npy'
    if path.exists():
        return np.load(path)
    encoder = ObservationEncoder(config)
    rng = np.random.default_rng(seed+protocol['pretraining_seed_offset'])
    observations, episodes = [],[]
    with OvertakingSensorEnv(config) as env:
        obs,_ = env.reset(seed=seed)
        agent = LinearExpectedSarsa(len(encoder(obs)),feature_schema=encoder.feature_schema)
        for episode in range(protocol['pretraining_episodes']):
            traffic_seed = seed+protocol['pretraining_seed_offset']+episode
            assert traffic_seed not in protocol['test_seeds']
            obs,_ = env.reset(seed=traffic_seed)
            steps = 0
            while True:
                features = encoder(obs)
                observations.append(features)
                action = agent.select(features,1.,rng)
                obs,_,done,truncated,info = env.step(agent.decode(action,env))
                steps += 1
                if done or truncated:break
            episodes.append({'episode':episode+1,'seed':traffic_seed,'steps':steps,
                             'elapsed_s':info['time_s']})
    folder.mkdir(parents=True,exist_ok=True)
    atomic_csv(folder/'episodes.csv',pd.DataFrame(episodes))
    temp=folder/f'features.{os.getpid()}.tmp.npy'
    np.save(temp,np.asarray(observations,dtype=np.float32))
    os.replace(temp,path)
    return np.load(path)


def representation_state(agent):
    if not isinstance(agent,AdaptiveExpectedSarsa):return None
    return {k:v.clone() for k,v in agent.network.state_dict().items() if not k.startswith('readout.')}


def assert_representation(agent,before):
    if before is not None:
        for k,v in before.items():
            torch.testing.assert_close(v,agent.network.state_dict()[k],rtol=0,atol=0)


def policy_state(agent):
    if isinstance(agent,LinearExpectedSarsa):return {'weights':agent.weights.copy()}
    net = agent.network if isinstance(agent,AdaptiveExpectedSarsa) else agent.online
    return {k:v.clone() for k,v in net.state_dict().items()}


def assert_policy(agent,before):
    after=policy_state(agent)
    for k,v in before.items():
        if isinstance(v,np.ndarray):np.testing.assert_array_equal(v,after[k])
        else:torch.testing.assert_close(v,after[k],rtol=0,atol=0)


def save_policy(agent,policy,folder):
    path = folder/('agent.npz' if policy=='sarsa' else 'agent.pt')
    temp = path.with_name('policy.tmp'+path.suffix)
    agent.save(temp)
    os.replace(temp,path)
    if policy=='dyna_dqn':
        torch.save(agent.model.state_dict(),folder/'prediction_model.pt')


def load_policy(policy,folder):
    if policy=='sarsa':return LinearExpectedSarsa.load(folder/'agent.npz')
    if policy in ['dqn','dyna_dqn']:return DQNAgent.load(folder/'agent.pt')
    return AdaptiveExpectedSarsa.load(folder/'agent.pt')


def checkpoint(folder,state,policy):
    temp=folder/'resume.tmp'
    with temp.open('wb') as f:
        pickle.dump({**state,'torch_rng':torch.get_rng_state()},f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp,folder/'resume.pkl')
    if state['history']:
        atomic_csv(folder/'training.csv',pd.DataFrame(state['history']))
    save_policy(state['agent'],policy,folder)


def train_one(output,policy,seed):
    folder=Path(output)/f'seed{seed}'/policy
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _train_one(output,policy,seed)
        except Exception as error:
            atomic_json(folder/'status.json',{'phase':'failed','policy':policy,'seed':seed,
                'error':repr(error),'updated_at':timestamp()})
            raise


def _train_one(output,policy,seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output=Path(output);protocol=read_protocol(output)
    config=ExperimentConfig.load(output/'config.json')
    folder=output/f'seed{seed}'/policy
    folder.mkdir(parents=True,exist_ok=True)
    status=folder/'status.json'
    if (output/'STOP').exists():
        previous=json.loads(status.read_text()) if status.exists() else {}
        if previous.get('phase')!='complete':
            atomic_json(status,{**previous,'phase':'stopped','policy':policy,'seed':seed,'updated_at':timestamp()})
        return None
    if (folder/'evaluation.csv').exists():
        if not (folder/'replay_complete.json').exists():
            export_matched_replays(config,load_policy(policy,folder),folder,protocol)
        return str(folder)
    encoder=ObservationEncoder(config)
    started=time.perf_counter()
    if (folder/'resume.pkl').exists():
        # Trusted local checkpoint created by this runner; includes optimizer,
        # replay/model state, exploration RNG, Torch RNG and completed episodes.
        with (folder/'resume.pkl').open('rb') as f:state=pickle.load(f)
        if state['protocol_hash']!=protocol['protocol_hash']:raise ValueError('Checkpoint protocol differs')
        torch.set_rng_state(state['torch_rng'])
    else:
        with OvertakingSensorEnv(config) as env:
            obs,_=env.reset(seed=seed)
        agent=make_agent(policy,len(encoder(obs)),encoder.feature_schema,seed,protocol)
        if policy in ['prototype_control','stdp_pretrained']:
            atomic_json(status,{'phase':'pretraining','policy':policy,'seed':seed,'updated_at':timestamp()})
            corpus_dir=folder.parent/'corpus'
            corpus_dir.mkdir(exist_ok=True)
            # Paired prototype variants reuse one training-only corpus per seed.
            with (corpus_dir/'collection.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                corpus=collect_corpus(config,seed,protocol,corpus_dir)
            diagnostics=[]
            for index in range(protocol['pretraining_passes']):
                order=np.random.default_rng(seed+30000+index).permutation(len(corpus))
                values=[agent.pretrain_step(corpus[i]) for i in order]
                diagnostics.append({'pass':index+1,'observations':len(corpus),
                    **{k:float(np.mean([v[k] for v in values])) for k in values[0]}})
            atomic_csv(folder/'pretraining.csv',pd.DataFrame(diagnostics))
        state={'agent':agent,'rng':np.random.default_rng(seed),'history':[],
               'episode':0,'protocol_hash':protocol['protocol_hash']}
        checkpoint(folder,state,policy)
    agent=state['agent'];rng=state['rng'];frozen=representation_state(agent)
    with OvertakingSensorEnv(config) as env:
        probe,_=env.reset(seed=seed)
        features=encoder(probe)
        for episode in range(state['episode'],protocol['episodes']):
            epsilon=1.-.9*episode/max(1,protocol['episodes']-1)
            obs,_=env.reset(seed=seed+episode);features=encoder(obs)
            total_return=0.;steps=0;metrics=[]
            while True:
                index=agent.select(features,epsilon,rng)
                obs,reward,done,truncated,info=env.step(agent.decode(index,env))
                next_features=encoder(obs);terminal=done or truncated
                if policy=='sarsa':
                    td=agent.update(features,index,reward,next_features,terminal,epsilon,.12,protocol['gamma'])
                    loss={'abs_td':abs(td)}
                elif policy in ['dqn','dyna_dqn']:
                    loss=agent.observe(features,index,reward,next_features,terminal)
                else:
                    td=agent.update(features,index,reward,next_features,terminal,epsilon)
                    loss={'abs_td':abs(td),**agent.last_metrics}
                metrics.append(loss);total_return+=reward;steps+=1;features=next_features
                if done or truncated:break
            record={'episode':episode+1,'seed':seed+episode,'epsilon':epsilon,
                    'return':total_return,'steps':steps,'crashed':info['crashed'],
                    'completed':info['route_completed'],'deadline_failure':bool(truncated and not info['route_completed']),
                    'elapsed_s':info['time_s'],'distance_m':info['distance_m'],
                    'energy_j':info['total_energy_j'],'overtakes':info['overtakes'],
                    'following_s':info['following_s'],'lane_changes':info['lane_changes'],
                    'traffic_spawned':info['traffic_spawned'],
                    **{k:float(np.mean([m[k] for m in metrics if k in m])) for k in set().union(*metrics)}}
            state['history'].append(record);state['episode']=episode+1
            atomic_json(status,{'phase':'training','policy':policy,'seed':seed,'episode':episode+1,
                        'episodes':protocol['episodes'],'last_return':total_return,
                        'updated_at':timestamp(),'pid':os.getpid()})
            if (episode+1)%protocol['checkpoint_every_episodes']==0 or episode+1==protocol['episodes'] or (output/'STOP').exists():
                assert_representation(agent,frozen)
                checkpoint(folder,state,policy)
                print(f'{policy} seed {seed}: {episode+1}/{protocol["episodes"]}, '
                      f'return={total_return:.2f}, wall={time.perf_counter()-started:.1f}s',flush=True)
            if (output/'STOP').exists():
                atomic_json(status,{'phase':'stopped','policy':policy,'seed':seed,'episode':state['episode'],
                    'episodes':protocol['episodes'],'updated_at':timestamp()})
                return None
    atomic_json(status,{'phase':'evaluation','policy':policy,'seed':seed,'episode':state['episode'],
                       'episodes':protocol['episodes'],'updated_at':timestamp(),'pid':os.getpid()})
    # Verify portable inference policy and frozen evaluation, not just resume.pkl.
    restored=load_policy(policy,folder)
    np.testing.assert_array_equal(agent.values(features),restored.values(features))
    before=policy_state(restored);results=[]
    progress_path=folder/'evaluation_progress.csv'
    if progress_path.exists():results=pd.read_csv(progress_path).to_dict('records')
    completed={(r['weather'],r['seed']) for r in results}
    for weather in protocol['weather']:
        trial=copy.deepcopy(config);trial.weather.preset=weather
        for test_seed in protocol['test_seeds']:
            if (weather,test_seed) in completed:continue
            if (output/'STOP').exists():
                atomic_json(status,{'phase':'stopped','policy':policy,'seed':seed,'episode':state['episode'],
                    'episodes':protocol['episodes'],'updated_at':timestamp()})
                return None
            result,_=rollout(trial,restored,test_seed)
            results.append({'policy':POLICIES[policy],'policy_id':policy,'training_seed':seed,
                            'weather':weather,**result})
            atomic_csv(progress_path,pd.DataFrame(results))
    assert_policy(restored,before)
    atomic_csv(folder/'evaluation.csv',pd.DataFrame(results))
    atomic_json(status,{'phase':'complete','policy':policy,'seed':seed,'episode':state['episode'],
                       'episodes':protocol['episodes'],'test_trips':len(results),'updated_at':timestamp()})
    if (output/'STOP').exists():return None
    export_matched_replays(config,restored,folder,protocol)
    return str(folder)


def rollout(config,agent,seed,capture=False,controller=None):
    encoder=ObservationEncoder(config);rng=np.random.default_rng(seed+1000000)
    snapshots,frames,records=[],[],[];counts={};drives={}
    with OvertakingSensorEnv(config,render_mode='rgb_array' if capture else None) as env:
        obs,info=env.reset(seed=seed)
        if capture:snapshots.append(obs);frames.append(env.render())
        total=0.
        while True:
            if controller is None:
                index=agent.select(encoder(obs),0.,rng)
                action=agent.decode(index,env);drive,mode=agent.actions[index]
            elif controller=='slow':
                action={'drive':next(k for k,v in env.backend.action_type.actions.items() if v=='SLOWER'),
                        'sensors':np.ones(len(config.sensors),np.int8)}
                drive,mode='SLOWER','all_on'
            else:
                action=oracle_action(env,allow_passing=controller=='oracle')
                drive,mode=env.backend.action_type.actions[action['drive']],'all_on'
            obs,reward,done,truncated,info=env.step(action);total+=reward
            counts[mode]=counts.get(mode,0)+1;drives[drive]=drives.get(drive,0)+1
            records.append({'time_s':info['time_s'],'drive':drive,'sensing':mode,'reward':reward,
                'speed_m_s':float(obs['ego'][0]),'distance_m':info['distance_m'],
                'total_energy_j':info['total_energy_j'],'overtakes':info['overtakes'],
                'crashed':info['crashed'],'completed':info['route_completed'],
                'following_s':info['following_s'],'lane_changes':info['lane_changes']})
            if capture:snapshots.append(obs);frames.append(env.render())
            if done or truncated:break
        metrics={'seed':seed,'return':total,'crashed':info['crashed'],'completed':info['route_completed'],
            'deadline_failure':bool(truncated and not info['route_completed']),
            'completion_time_s':info['time_s'] if info['route_completed'] else np.nan,
            'elapsed_s':info['time_s'],'distance_m':info['distance_m'],'steps':len(records),
            'energy_j':info['total_energy_j'],'mean_power_w':info['total_energy_j']/info['time_s'],
            'overtakes':info['overtakes'],'lane_changes':info['lane_changes'],
            'following_s':info['following_s'],'traffic_spawned':info['traffic_spawned'],
            **{f'mode_fraction_{k}':v/len(records) for k,v in counts.items()},
            **{f'drive_fraction_{k}':v/len(records) for k,v in drives.items()}}
        roll=Rollout(pd.DataFrame(records),snapshots,frames,env.sensor_names,config.grid.radius_m)
    return metrics,roll


def export_matched_replays(config,agent,folder,protocol):
    from experiments.replay_agents import replay_frames,HTML
    target=folder/'replays';target.mkdir(exist_ok=True)
    evals=pd.read_csv(folder/'evaluation.csv')
    selected=[evals.iloc[0]]  # Same seed/weather for all policies.
    failures=evals[~evals.completed]
    if len(failures) and (failures.iloc[0]['seed'],failures.iloc[0].weather)!=(selected[0]['seed'],selected[0].weather):
        selected.append(failures.iloc[0])
    for row in selected:
        output=target/f'{row.weather}_seed{int(row["seed"])}';output.mkdir(exist_ok=True)
        trial=copy.deepcopy(config);trial.weather.preset=row.weather
        result,roll=rollout(trial,agent,int(row['seed']),capture=True)
        for key in ['return','distance_m','energy_j','overtakes','completed','crashed']:
            np.testing.assert_allclose(result[key],row[key],rtol=1e-10,atol=1e-8)
        clean={k:None if isinstance(v,float) and not np.isfinite(v) else v for k,v in result.items()}
        data,gifs=replay_frames(roll,clean,str(row.policy))
        name='policy';step=1/config.highway['policy_frequency']
        durations=[round(step*1000)]*len(gifs);durations[-1]=2000
        gifs[0].save(output/'policy.gif',save_all=True,append_images=gifs[1:],duration=durations,loop=0)
        atomic_csv(output/'policy_history.csv',roll.history)
        replay={'seed':int(row['seed']),'weather':row.weather,'step_s':step,'route_m':9000,
                'deadline_s':360,'sensor_names':roll.sensor_names,
                'agents':[{'id':name,'label':str(row.policy),'metrics':clean,'frames':data}]}
        template=HTML.replace('Frozen policies from the 1,000-episode run.',
                              f'Frozen policy from the {protocol["episodes"]}-episode overtaking study.')
        (output/'replay.html').write_text(template.replace('__DATA__',json.dumps(replay,allow_nan=False)))
        atomic_json(output/'metrics.json',clean)
    atomic_json(folder/'replay_complete.json',{'updated_at':timestamp(),'scenarios':len(selected)})


def validate_scenario(output):
    config=ExperimentConfig.load(Path(output)/'config.json');config.weather.preset='clear'
    records=[]
    for seed in [900001,900002,900003,900004,900005]:
        for controller in ['oracle','follow','slow']:
            result,_=rollout(config,None,seed,controller=controller)
            records.append({'controller':controller,**result})
            print('Validation',seed,controller,result['completed'],result['crashed'],
                  round(result['distance_m']),result['overtakes'],flush=True)
    frame=pd.DataFrame(records);atomic_csv(Path(output)/'scenario_validation.csv',frame)
    oracle=frame[frame.controller=='oracle']
    passed=(oracle.completed.mean()>=.8 and not oracle.crashed.any()
            and (oracle.loc[oracle.completed,'overtakes']>=3).all()
            and not frame[frame.controller.isin(['follow','slow'])].completed.any())
    atomic_json(Path(output)/'validation.json',{'passed':bool(passed),'oracle_completion':float(oracle.completed.mean()),
            'oracle_crashes':int(oracle.crashed.sum()),
            'config_sha256':hashlib.sha256((Path(output)/'config.json').read_bytes()).hexdigest(),
            'backend_sha256':hashlib.sha256((ROOT/'environments/sensor_highway/overtaking.py').read_bytes()).hexdigest(),
            'updated_at':timestamp()})
    if not passed:raise RuntimeError('Scenario feasibility checks failed; full training remains unlaunched')
    return frame


def summarize(output):
    output=Path(output);protocol=read_protocol(output);frames=[];histories=[]
    for seed in protocol['training_seeds']:
        for policy in POLICIES:
            folder=output/f'seed{seed}'/policy
            if (folder/'evaluation.csv').exists():frames.append(pd.read_csv(folder/'evaluation.csv'))
            if (folder/'training.csv').exists():
                try:
                    frame=pd.read_csv(folder/'training.csv')
                except pd.errors.EmptyDataError:
                    continue
                if len(frame):
                    histories.append(frame.assign(policy=POLICIES[policy],training_seed=seed))
    if histories:atomic_csv(output/'training.csv',pd.concat(histories,ignore_index=True))
    if not frames:return None
    data=pd.concat(frames,ignore_index=True);atomic_csv(output/'evaluation.csv',data)
    per_seed=data.groupby(['policy','training_seed']).agg(trips=('completed','size'),
        completion=('completed','mean'),crash=('crashed','mean'),deadline_failure=('deadline_failure','mean'),
        successful_time_s=('completion_time_s','mean'),overtakes=('overtakes','mean'),
        power_w=('mean_power_w','mean'),following_s=('following_s','mean'),mean_return=('return','mean')).reset_index()
    atomic_csv(output/'per_training_seed.csv',per_seed)
    summary=per_seed.groupby('policy').agg(training_seeds=('training_seed','size'),
        completion=('completion','mean'),completion_sd=('completion','std'),crash=('crash','mean'),
        deadline_failure=('deadline_failure','mean'),overtakes=('overtakes','mean'),
        power_w=('power_w','mean'),following_s=('following_s','mean'),mean_return=('mean_return','mean'))
    summary['successful_time_s']=data.groupby('policy').completion_time_s.mean()
    atomic_csv(output/'summary.csv',summary.reset_index())
    return summary


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    if not (output/'summary.csv').exists():return
    summary=pd.read_csv(output/'summary.csv').set_index('policy')
    per_seed=pd.read_csv(output/'per_training_seed.csv')
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    labels=list(summary.index);x=np.arange(len(labels))
    for i,seed in enumerate(sorted(per_seed.training_seed.unique())):
        points=per_seed[per_seed.training_seed==seed].set_index('policy').reindex(labels)
        axes[0].scatter(x+(i-1)*.06,points.completion,label=f'Seed {seed}')
    axes[0].plot(x,summary.completion,'k_',markersize=18)
    axes[0].set(title='Held-out completion',ylabel='Completion fraction',ylim=(-.05,1.05))
    axes[0].legend(fontsize=8)
    axes[1].bar(x,summary.successful_time_s)
    axes[1].set(title='Successful completion time',ylabel='Seconds')
    axes[2].bar(x,summary.power_w)
    axes[2].set(title='Modeled sensor power (all trips)',ylabel='Watts')
    for ax in axes:ax.set_xticks(x,labels,rotation=30,ha='right',fontsize=8)
    fig.suptitle('9,000 m / 360 s overtaking study')
    fig.savefig(output/'comparison.png',dpi=160)
    plt.close(fig)


def coordinate(output,workers=3):
    output=Path(output);protocol=read_protocol(output)
    if not protocol['pilot']:
        if not (output/'validation.json').exists() or not json.loads((output/'validation.json').read_text())['passed']:
            raise RuntimeError('Run --validate first')
        evidence=json.loads((output/'validation.json').read_text())
        if evidence.get('config_sha256')!=hashlib.sha256((output/'config.json').read_bytes()).hexdigest() or evidence.get('backend_sha256')!=protocol['source_hashes']['environments/sensor_highway/overtaking.py']:
            raise RuntimeError('Validation does not match this traffic/configuration; rerun --validate')
        pilot=output/'pilot'/'study_status.json'
        if not pilot.exists() or json.loads(pilot.read_text())['phase']!='complete':
            raise RuntimeError('Run --pilot successfully before the full study')
        read_protocol(output/'pilot')
    atomic_json(output/'study_status.json',{'phase':'running','pid':os.getpid(),'workers':workers,'started_at':timestamp()})
    failures=[];stopped=False
    # One job owns all state/files for its policy and training seed.
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(train_one,output,p,seed):(p,seed)
                 for seed in protocol['training_seeds'] for p in POLICIES}
        for future in as_completed(futures):
            policy,seed=futures[future]
            try:
                if future.result() is None:stopped=True
            except Exception as error:
                failures.append({'policy':policy,'seed':seed,'error':repr(error)})
                print('JOB FAILED',policy,seed,repr(error),flush=True)
            summarize(output)
    summary=summarize(output)
    phase='failed' if failures else 'stopped' if stopped else 'complete'
    atomic_json(output/'study_status.json',{'phase':phase,'failures':failures,'updated_at':timestamp()})
    if failures:raise RuntimeError(f'{len(failures)} job(s) failed; use status/logs and resume')
    if stopped:
        print('Study stopped at checkpoint boundaries. Remove STOP and use --launch to resume.',flush=True)
        return summary
    rows=['# 9 km overtaking study','',
        f"{protocol['episodes']} training episodes per agent across {len(protocol['training_seeds'])} seeds.",
        '9,000 m route; 360 s deadline; persistent 18–22 m/s traffic; gamma 0.999.','',
        '| Policy | Completion | Crash | Deadline failure | Successful time (s) | Overtakes | Sensor power (W) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in summary.iterrows():
        rows.append(f'| {name} | {row.completion:.1%} | {row.crash:.1%} | {row.deadline_failure:.1%} | '
                    f'{row.successful_time_s:.2f} | {row.overtakes:.2f} | {row.power_w:.2f} |')
    rows += ['', 'Completion times include successful trips only. Overtakes and power average all trips.',
        'Prototype methods reuse an extra 20-episode training-only corpus per seed for three passes.',
        'Episode budgets match; step counts, learning updates, and compute differ.',
        'Simulator truth is used only for diagnostic metrics and the feasibility oracle.',
        'Sensor energy excludes neural computation. Repeated weather trials are not independent training runs.',
        'Portable policy files and trusted local full-state resume checkpoints are stored per policy/seed.',
        'Matched-scenario and failure GIFs/interactive replays are under each model directory.']
    (output/'study_report.md').write_text('\n'.join(rows)+'\n')
    plot_results(output)
    print('STUDY COMPLETE',output,flush=True)


def launch(output,workers):
    output=Path(output)
    path=output/'study_status.json'
    if path.exists():
        state=json.loads(path.read_text())
        if state.get('phase')=='running':
            try:os.kill(state['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'Coordinator already running: PID {state["pid"]}')
    log=(output/'runner.log').open('a',buffering=1)
    process=subprocess.Popen([sys.executable,'-m','experiments.run_9000m_study','--coordinate',
            '--output',str(output),'--workers',str(workers)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
    log.close()
    atomic_json(output/'launch.json',{'pid':process.pid,'launched_at':timestamp(),'log':str(output/'runner.log')})
    print('Launched PID',process.pid,'log:',output/'runner.log',flush=True)
    return process.pid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    for flag in ['prepare','validate','pilot','launch','coordinate','status']:
        p.add_argument('--'+flag,action='store_true')
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();output=args.output.resolve()
    if not 1<=args.workers<=6:raise ValueError('workers must be between 1 and 6')
    if args.prepare:prepare(output)
    if args.validate:validate_scenario(output)
    if args.pilot:
        pilot=prepare(output/'pilot',episodes=3,seeds=[70042],pilot=True)
        coordinate(pilot,args.workers)
    if args.coordinate:coordinate(output,args.workers)
    if args.launch:launch(output,args.workers)
    if args.status:
        for path in [output/'study_status.json',*sorted(output.glob('seed*/*/status.json'))]:
            if path.exists():print(path.relative_to(output),path.read_text().strip())


if __name__=='__main__':main()
