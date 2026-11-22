"""Recreate the longest DQN training/evaluation trips for scenario inspection.

Training reconstruction starts from the original initialization and repeats every
update/action up to the earliest longest-duration episode. It never modifies the
study's models. Saved row metrics and the checkpoint hash are verified afterward.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import torch

from experiments import run_9000m_study as study
from experiments.run_9000m_preliminary import DEFAULT_OUTPUT
from experiments.replay_agents import HTML,replay_frames
from algorithms.sensor_agent import ObservationEncoder
from environments.sensor_highway import ExperimentConfig
from environments.sensor_highway.overtaking import OvertakingSensorEnv
from experiments.notebook_helpers import Rollout


def verify(metrics,row):
    for key in ['return','elapsed_s','distance_m','overtakes','crashed','completed','energy_j','steps']:
        np.testing.assert_allclose(metrics[key],row[key],rtol=1e-10,atol=1e-8,
                                   err_msg=f'Reconstructed trajectory differs: {key}')


def export(output,metrics,rollout,label,description,manifest):
    output.mkdir(parents=True,exist_ok=True)
    clean={k:None if isinstance(v,float) and not np.isfinite(v) else v for k,v in metrics.items()}
    frames,gifs=replay_frames(rollout,clean,label)
    durations=[500]*len(gifs);durations[-1]=2000
    # GIF previews include telemetry; HTML also keeps original road/map frames.
    gifs[0].save(output/'policy.gif',save_all=True,append_images=gifs[1:],duration=durations,loop=0)
    sample_indices=np.linspace(0,len(gifs)-1,6,dtype=int)
    contact=Image.new('RGB',(1800,1530),'white')
    for i,index in enumerate(sample_indices):contact.paste(gifs[index],((i%2)*900,(i//2)*510))
    contact.save(output/'scenario_overview.png')
    gifs[-1].save(output/'final_frame.png')
    del gifs
    rollout.history.to_csv(output/'policy_history.csv',index=False)
    replay={'seed':int(metrics['seed']),'weather':manifest['weather'],'step_s':.5,
            'route_m':9000,'deadline_s':360,'sensor_names':rollout.sensor_names,
            'agents':[{'id':'policy','label':label,'metrics':clean,'frames':frames}]}
    html=HTML.replace('Matched agent replays',label).replace(
        'Frozen policies from the 1,000-episode run. Same starting scenario; actions produce different traffic interactions.',description)
    html=html.replace('This first held-out scenario is illustrative;', 'This individual trajectory is illustrative;')
    (output/'replay.html').write_text(html.replace('__DATA__',json.dumps(replay,allow_nan=False)))
    study.atomic_json(output/'metrics.json',clean)
    study.atomic_json(output/'manifest.json',manifest)
    (output/'README.md').write_text(f'# {label}\n\n{description}\n\nOpen replay.html for play/pause, frame stepping and speed controls. policy.gif is a standalone preview. scenario_overview.png shows six times from the trajectory.\n\nThe road viewport is widened to show roughly 400 m of highway. This visual change does not change the traffic, policy observations, or dynamics. The map shows sensor evidence; the road view shows simulator truth. Metrics match the original saved row.\n')
    print('SAVED',output,flush=True)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root=DEFAULT_OUTPUT
    protocol=study.read_protocol(root)
    candidates=[]
    for p in sorted(root.glob('seed*/dqn/evaluation.csv')):
        d=pd.read_csv(p)
        d['model_folder']=str(p.parent)
        candidates.append(d)
    data=pd.concat(candidates,ignore_index=True)
    # Duration first, then distance, then stable scenario order for tied trips.
    row=data.sort_values(['elapsed_s','distance_m'],ascending=False,kind='stable').iloc[0]
    folder=Path(row.model_folder)
    before_hash=hashlib.sha256((folder/'agent.pt').read_bytes()).hexdigest()
    config=ExperimentConfig.load(root/'config.json')
    config.highway.update(screen_width=1200,screen_height=300,scaling=3.,centering_position=[.25,.5])
    config.weather.preset=row.weather
    agent=study.load_policy('dqn',folder)
    metrics,roll=study.rollout(config,agent,int(row['seed']),capture=True)
    verify(metrics,row)
    destination=root/'scenario_replays'/'dqn_longest_evaluation'
    export(destination,metrics,roll,'DQN — longest evaluation trip',
           f'Frozen DQN trained for 500 episodes, training seed {int(row.training_seed)}. Longest recorded held-out trip: {metrics["elapsed_s"]:.1f} s, ending in a crash. Traffic seed {int(row["seed"])}; weather {row.weather}.',
           {'kind':'evaluation','weather':row.weather,'training_seed':int(row.training_seed),
            'traffic_seed':int(row['seed']),'checkpoint_sha256':before_hash,'evaluation_match_verified':True})
    del roll,agent
    training=pd.read_csv(folder/'training.csv')
    # Multiple episodes reach the 360 s cap. Pick the earliest such episode so
    # reconstruction requires the fewest training updates; no cherry-picked reward.
    longest=float(training.elapsed_s.max())
    target=training[training.elapsed_s==longest].iloc[0]
    training_seed=int(row.training_seed)
    config=ExperimentConfig.load(root/'config.json')
    config.highway.update(screen_width=1200,screen_height=300,scaling=3.,centering_position=[.25,.5])
    encoder=ObservationEncoder(config)
    rng=np.random.default_rng(training_seed)
    snapshots,frames,records=[],[],[]
    with OvertakingSensorEnv(config,render_mode='rgb_array') as env:
        probe,_=env.reset(seed=training_seed)
        agent=study.make_agent('dqn',len(encoder(probe)),encoder.feature_schema,training_seed,protocol)
        # The training runner has a separate probe reset, then a reset per episode.
        env.reset(seed=training_seed)
        for episode in range(int(target.episode)):
            obs,info=env.reset(seed=training_seed+episode)
            features=encoder(obs)
            epsilon=1.-.9*episode/(protocol['episodes']-1)
            capturing=episode+1==int(target.episode)
            if capturing:snapshots.append(obs);frames.append(env.render())
            total=0.;steps=0
            while True:
                index=agent.select(features,epsilon,rng)
                action=agent.decode(index,env);drive,mode=agent.actions[index]
                obs,reward,done,truncated,info=env.step(action)
                next_features=encoder(obs)
                agent.observe(features,index,reward,next_features,done or truncated)
                total+=reward;steps+=1;features=next_features
                if capturing:
                    records.append({'time_s':info['time_s'],'drive':drive,'sensing':mode,'reward':reward,
                        'speed_m_s':float(obs['ego'][0]),'distance_m':info['distance_m'],
                        'total_energy_j':info['total_energy_j'],'overtakes':info['overtakes'],
                        'crashed':info['crashed'],'completed':info['route_completed'],
                        'lane_changes':info['lane_changes'],'following_s':info['following_s'],
                        'traffic_spawned':info['traffic_spawned'],'active_traffic':info['active_traffic']})
                    snapshots.append(obs);frames.append(env.render())
                if done or truncated:break
            metrics={'seed':training_seed+episode,'return':total,'elapsed_s':info['time_s'],
                'distance_m':info['distance_m'],'overtakes':info['overtakes'],'crashed':info['crashed'],
                'completed':info['route_completed'],'energy_j':info['total_energy_j'],'steps':steps,
                'deadline_failure':bool(truncated and not info['route_completed']),
                'completion_time_s':info['time_s'] if info['route_completed'] else np.nan}
            verify(metrics,training.iloc[episode])
            print('Verified training episode',episode+1,'duration',info['time_s'],flush=True)
        roll=Rollout(pd.DataFrame(records),snapshots,frames,env.sensor_names,config.grid.radius_m)
        weather=env.weather.preset
    destination=root/'scenario_replays'/'dqn_longest_training'
    export(destination,metrics,roll,'DQN — 360-second exploratory training trip',
           f'Reconstructed training episode {int(target.episode)}, traffic seed {int(target["seed"])}. Epsilon {float(target.epsilon):.4f}: approximately {100*float(target.epsilon):.2f}% random actions. The model learns during this replay. This is not the final frozen policy and does not show trained driving ability. It runs until the 360 s deadline without completing the route.',
           {'kind':'training_reconstruction','weather':weather,'training_seed':training_seed,
            'traffic_seed':int(target['seed']),'episode':int(target.episode),'epsilon':float(target.epsilon),
            'all_preceding_episodes_match_verified':True,'original_training_history':str(folder/'training.csv'),
            'recorded_duration_max_s':longest,'checkpoint_unchanged_verified':True})
    assert hashlib.sha256((folder/'agent.pt').read_bytes()).hexdigest()==before_hash


if __name__=='__main__':main()
