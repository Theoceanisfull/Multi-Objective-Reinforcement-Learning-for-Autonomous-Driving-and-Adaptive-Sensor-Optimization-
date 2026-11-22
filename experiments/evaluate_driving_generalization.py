"""Evaluate existing driving checkpoints without training.

Usage: .venv/bin/python -m experiments.evaluate_driving_generalization
Progress and completed rows are saved after every trip. A STOP file in the output
directory stops cleanly between trips. Checkpoint bytes and policy parameters are
verified unchanged. The privileged oracle is feasibility evidence, never a policy
input or a training source.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from algorithms.driving_agents import DrivingDQN, DrivingEncoder, DrivingExpectedSarsa
from environments.sensor_highway.config import ExperimentConfig
from environments.sensor_highway.driving_evaluation import DrivingEvaluationEnv
from environments.sensor_highway.overtaking import oracle_action
from experiments.driving_policy_helpers import ROOT, check_q

DEFAULT_RUN = ROOT/'runs/exploratory/driving_policy_v1/driving_500ep_20261121_103801_682209'
CASES = [
    dict(name='control_1000m', traffic='control', route_m=1000, deadline_s=45),
    dict(name='varied_lead_1000m', traffic='varied_lead', route_m=1000, deadline_s=45),
    dict(name='blocked_left_1000m', traffic='blocked_left', route_m=1000, deadline_s=45),
    dict(name='blocked_right_1000m', traffic='blocked_right', route_m=1000, deadline_s=45),
    dict(name='repeated_2250m', traffic='repeated', route_m=2250, deadline_s=100),
    dict(name='repeated_5000m', traffic='repeated', route_m=5000, deadline_s=220),
    dict(name='repeated_9000m', traffic='repeated', route_m=9000, deadline_s=360),
]
POLICIES = ['sarsa', 'dqn', 'dyna_dqn', 'oracle', 'fixed_left_fast']


def load_agents(source):
    return dict(sarsa=DrivingExpectedSarsa.load(source/'sarsa/agent.npz'),
                dqn=DrivingDQN.load(source/'dqn/agent.pt'),
                dyna_dqn=DrivingDQN.load(source/'dyna_dqn/agent.pt'))


def initialize_worker(source):
    global WORKER_AGENTS
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    WORKER_AGENTS = load_agents(source)


def worker_rollout(config, policy, seed):
    return rollout(config,policy,WORKER_AGENTS.get(policy),seed)[0]


def make_config(source, case, weather='clear'):
    config = copy.deepcopy(source)
    config.reward.route_distance_m = float(case['route_m'])
    config.highway['duration'] = float(case['deadline_s'])
    config.highway['driving_evaluation'] = copy.deepcopy(case)
    config.highway['overtaking'] = dict(spacing_m=260., speed_min_m_s=18.,
        speed_max_m_s=22., remove_behind_m=100., spawn_ahead_m=600., entry_min_gap_m=300.)
    config.weather.preset = weather
    config.validate()
    return config


def state(agent):
    if isinstance(agent, DrivingExpectedSarsa):
        return [agent.weights.copy()]
    return [p.detach().numpy().copy() for p in agent.online.parameters()]


def select_action(policy, agent, obs, features, env, rng, step):
    if agent is not None:
        return int(agent.select(features, 0., rng))
    if policy == 'oracle':
        return int(oracle_action(env)['drive'])
    if policy == 'fixed_left_fast':
        # Ignores traffic: move to the leftmost lane, then accelerate to 30 m/s.
        # Uses only ego controller telemetry, never other vehicle states.
        controller = obs['controller']
        if controller[1] > 0 and controller[3] < .5:
            return 0
        return 3 if controller[2] < .999 else 1
    raise ValueError(policy)


def rollout(config, policy, agent, seed, capture=False, env_class=DrivingEvaluationEnv):
    encoder = DrivingEncoder(config)
    rng = np.random.default_rng(seed+1000000)
    before = state(agent) if agent is not None else []
    records, snapshots, frames = [], [], []
    counts = np.zeros(5, dtype=int)
    with env_class(config, render_mode='rgb_array' if capture else None) as env:
        obs, info = env.reset(seed=seed)
        initial_traffic = [dict(uid=int(v.experiment_id), lane=int(v.lane_index[2]),
            gap_m=float(v.position[0]-env.backend.vehicle.position[0]), speed_m_s=float(v.speed))
            for v in env.backend.road.vehicles if v is not env.backend.vehicle]
        initial_lane = int(env.backend.vehicle.lane_index[2])
        total = float(info['warmup_reward'])
        steps = 0
        max_q = 0.
        min_gap = float('inf')
        if capture:
            snapshots.extend([env.prelude_observation, obs])
            frames.extend([env.prelude_frame, env.render()])
        def record(name, reward):
            records.append(dict(time_s=info['time_s'], drive=name, sensing='all_on',
                reward=reward, speed_m_s=float(obs['ego'][0]), distance_m=info['distance_m'],
                total_energy_j=info['total_energy_j'], overtakes=info['overtakes'],
                lane_changes=info['lane_changes'], crashed=info.get('crashed', False),
                completed=info['route_completed']))
        record('IDLE (warmup)', info['warmup_reward'])
        while True:
            features = encoder(obs)
            if agent is not None:
                max_q = max(max_q, check_q(agent, features, 20.))
            action = select_action(policy, agent, obs, features, env, rng, steps)
            counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            ego = env.backend.vehicle
            front, _ = env.backend.road.neighbour_vehicles(ego, ego.lane_index)
            if front is not None:
                min_gap = min(min_gap, float(front.position[0]-ego.position[0]-5))
            record(env.backend.action_type.actions[action], reward)
            if capture:
                snapshots.append(obs)
                frames.append(env.render())
            if terminated or truncated:
                break
        metrics = dict(policy=policy, seed=seed, completed=info['route_completed'],
            crashed=info['crashed'], deadline_failure=bool(truncated and not info['route_completed']),
            return_=total, steps=steps, elapsed_s=info['time_s'],
            completion_time_s=info['time_s'] if info['route_completed'] else np.nan,
            distance_m=info['distance_m'], overtakes=info['overtakes'],
            lane_changes=info['lane_changes'], following_s=info['following_s'],
            traffic_spawned=info['traffic_spawned'], max_abs_q=max_q,
            energy_j=info['total_energy_j'], sensor_power_w=info['total_energy_j']/info['time_s'],
            min_front_gap_at_decisions_m=min_gap if np.isfinite(min_gap) else np.nan,
            initial_lane=initial_lane, initial_traffic=json.dumps(initial_traffic),
            **{f'action_{name}_fraction':float(counts[i]/steps)
               for i, name in enumerate(['left','idle','right','faster','slower'])})
        sensor_names = env.sensor_names
    if agent is not None:
        for a, b in zip(before, state(agent)):
            np.testing.assert_array_equal(a, b)
    from experiments.notebook_helpers import Rollout
    return metrics, Rollout(pd.DataFrame(records), snapshots, frames, sensor_names, config.grid.radius_m)


def summarize(rows):
    frame = pd.DataFrame(rows)
    return frame.groupby(['scenario','weather','policy'], sort=False).agg(
        trips=('completed','size'), completions=('completed','sum'), crashes=('crashed','sum'),
        deadline_failures=('deadline_failure','sum'), completion_rate=('completed','mean'),
        mean_finish_s=('completion_time_s','mean'), mean_elapsed_s=('elapsed_s','mean'),
        mean_distance_m=('distance_m','mean'), max_distance_m=('distance_m','max'),
        mean_overtakes=('overtakes','mean'), max_overtakes=('overtakes','max'),
        mean_lane_changes=('lane_changes','mean'), max_abs_q=('max_abs_q','max')).reset_index()


def export_replay(config, policy, agent, seed, folder, expected, env_class=DrivingEvaluationEnv):
    from experiments.replay_agents import HTML, replay_frames
    folder.mkdir(parents=True, exist_ok=True)
    metrics, roll = rollout(config, policy, agent, seed, capture=True,env_class=env_class)
    for key in ['completed','crashed','distance_m','elapsed_s','overtakes']:
        np.testing.assert_allclose(metrics[key], expected[key], rtol=0, atol=1e-8)
    data, gifs = replay_frames(roll, metrics, policy)
    gifs[0].save(folder/'policy.gif', save_all=True, append_images=gifs[1:],
                 duration=[500]*(len(gifs)-1)+[2000], loop=0)
    roll.history.to_csv(folder/'policy_history.csv', index=False)
    def clean(v):
        if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v, list): return [clean(x) for x in v]
        return None if isinstance(v, float) and not np.isfinite(v) else v
    payload = dict(seed=seed, weather=config.weather.preset,
        route_m=config.reward.route_distance_m, deadline_s=config.highway['duration'],
        step_s=.5, sensor_names=roll.sensor_names,
        agents=[dict(id='policy', label=policy, metrics=metrics, frames=data)])
    html = HTML.replace('Matched agent replays','Driving generalization evaluation').replace(
        'Frozen policies from the 1,000-episode run. Same starting scenario; actions produce different traffic interactions.',
        'Frozen driving policy evaluated on changed traffic. All sensors remain on; no training occurs.')
    (folder/'replay.html').write_text(html.replace('__DATA__',json.dumps(clean(payload),allow_nan=False)))
    (folder/'metrics.json').write_text(json.dumps(clean(metrics),indent=2)+'\n')


def hashes(source):
    return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [source/'sarsa/agent.npz',source/'dqn/agent.pt',source/'dyna_dqn/agent.pt']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--seed-start',type=int,default=300000)
    parser.add_argument('--cases',nargs='+',choices=[c['name'] for c in CASES])
    parser.add_argument('--policies',nargs='+',choices=POLICIES,default=POLICIES)
    parser.add_argument('--no-weather',action='store_true')
    parser.add_argument('--no-replays',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if args.seeds <= 0 or args.workers <= 0: parser.error('seeds/workers must be positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = args.run.resolve()
    output = args.output or ROOT/'runs/local/driving_generalization'/datetime.now(
        ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    config = ExperimentConfig.load(source/'config.json')
    agents = load_agents(source)
    before = hashes(source)
    source_metadata=json.loads((source/'metadata.json').read_text())
    training_seed=int(source_metadata['seed'])
    training_episodes=int(source_metadata['settings']['episodes'])
    if set(range(training_seed,training_seed+training_episodes)) & set(range(args.seed_start,args.seed_start+args.seeds)):
        raise ValueError('Evaluation seeds overlap training seeds')
    cases = [c for c in CASES if not args.cases or c['name'] in args.cases]
    batches = [(c,'clear',args.seeds) for c in cases]
    if not args.no_weather:
        batches += [(c,w,min(10,args.seeds)) for c in cases if c['name']=='repeated_2250m'
                    for w in ['rain','fog']]
    metadata = dict(source_run=str(source), source_checkpoints_sha256=before,
        training_settings=source_metadata['settings'], training_seed=training_seed,
        frozen=True, seed_start=args.seed_start, policies=args.policies, workers=args.workers,
        batches=[dict(case=c,weather=w,trips_per_policy=n) for c,w,n in batches],
        notes=['Oracle uses traffic truth for feasibility only; learned policies receive sensor observations.',
               'All sensors on. Route/deadline changes also change normalized task features.',
               'Overtakes count distinct traffic encountered ahead within 150 m and subsequently passed by >10 m.',
               'Road users hold lanes; this does not test cut-ins or lane-changing traffic.'])
    sources=['environments/sensor_highway/driving_evaluation.py',
             'experiments/evaluate_driving_generalization.py','environments/sensor_highway/driving.py',
             'environments/sensor_highway/overtaking.py','algorithms/driving_agents.py']
    metadata['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rows=[];start=time.perf_counter();total=sum(n*len(args.policies) for _,_,n in batches)
    print(f'Output: {output}\nScheduled: {total} frozen-policy trips',flush=True)
    stopped=False
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,initargs=(source,)) as pool:
        for case,weather,n in batches:
            c=make_config(config,case,weather)
            c.save(output/f"config_{case['name']}_{weather}.json")
            for policy in args.policies:
                batch_start=time.perf_counter()
                futures=[pool.submit(worker_rollout,c,policy,seed)
                         for seed in range(args.seed_start,args.seed_start+n)]
                for future in as_completed(futures):
                    if (output/'STOP').exists():
                        stopped=True
                        for pending in futures:pending.cancel()
                        break
                    metrics=future.result()
                    rows.append(dict(scenario=case['name'],weather=weather,**metrics))
                    pd.DataFrame(rows).sort_values(['scenario','weather','policy','seed']).to_csv(output/'evaluation.csv',index=False)
                    (output/'status.json').write_text(json.dumps(dict(phase='evaluation',done=len(rows),
                        total=total,scenario=case['name'],weather=weather,policy=policy,
                        wall_s=time.perf_counter()-start),indent=2)+'\n')
                if rows:summarize(rows).to_csv(output/'summary.csv',index=False)
                recent=[r for r in rows if r['scenario']==case['name'] and r['weather']==weather and r['policy']==policy]
                print(f"{case['name']} {weather} {policy}: {sum(r['completed'] for r in recent)}/{len(recent)} completed, "
                      f"{sum(r['crashed'] for r in recent)} crashes; {len(rows)}/{total}; "
                      f"batch {time.perf_counter()-batch_start:.1f}s, total {time.perf_counter()-start:.1f}s",flush=True)
                if stopped:break
            if stopped:break
    assert before == hashes(source), 'Checkpoint files changed during evaluation'
    if not stopped and not args.no_replays:
        (output/'status.json').write_text(json.dumps(dict(phase='replays',done=len(rows),total=total),indent=2)+'\n')
        df=pd.DataFrame(rows).sort_values(['scenario','weather','seed'])
        for policy in [p for p in args.policies if p in agents]:
            subset=df[(df.policy==policy)&(df.weather=='clear')&(df.scenario!='control_1000m')]
            failures=subset[~subset.completed]
            picks=[]
            if len(failures): picks.append(('first_failure',failures.iloc[0]))
            picks.append(('longest_distance',subset.loc[subset.distance_m.idxmax()]))
            for title,row in picks:
                if (output/'STOP').exists():stopped=True;break
                case=next(c for c in cases if c['name']==row.scenario)
                export_replay(make_config(config,case,row.weather),policy,agents[policy],int(row.seed),
                    output/'replays'/policy/title,row)
                print(f'Replay saved: {policy}/{title}',flush=True)
            if stopped:break
    assert before==hashes(source)
    (output/'status.json').write_text(json.dumps(dict(phase='stopped' if stopped else 'complete',
        done=len(rows),total=total,wall_s=time.perf_counter()-start,checkpoints_unchanged=True),indent=2)+'\n')
    print(f'Finished: {output}',flush=True)


if __name__=='__main__':
    main()
