"""Export a frozen adaptive-reservoir replay, verified against its evaluation row.

From final_project: .venv/bin/python -m experiments.replay_adaptive_snn
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from algorithms.snn import AdaptiveExpectedSarsa
from environments.sensor_highway import ExperimentConfig
from experiments.agent_helpers import run_agent
from experiments.replay_agents import HTML, replay_frames

DEFAULT_RUN = Path('runs/exploratory/adaptive_snn_120s_60ep_3seeds_v1')


def export_replay(run_dir=DEFAULT_RUN, training_seed=42, seed=100000, weather='clear'):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run_dir = Path(run_dir).resolve()
    model_dir = run_dir / f'seed{training_seed}' / 'adaptive_reservoir'
    expected = pd.read_csv(model_dir / 'evaluation.csv')
    expected = expected[(expected.seed == seed) & (expected.weather == weather)]
    if len(expected) != 1:
        raise ValueError('Choose a traffic seed/weather recorded in the evaluation CSV')
    agent = AdaptiveExpectedSarsa.load(model_dir / 'agent.pt')
    config = ExperimentConfig.load(run_dir / 'config.json')
    config.weather.preset = weather
    before = {k: v.clone() for k, v in agent.network.state_dict().items()}
    updates = agent.q_updates
    metrics, rollout = run_agent(config, agent, seed=seed, capture=True)
    for k, value in before.items():
        torch.testing.assert_close(value, agent.network.state_dict()[k], rtol=0, atol=0)
    assert agent.q_updates == updates
    for key, value in metrics.items():
        if key in expected.columns:
            np.testing.assert_allclose(value, expected.iloc[0][key], rtol=1e-10, atol=1e-8,
                                       equal_nan=True, err_msg=f'Replay differs from evaluation: {key}')
    output = Path('runs/local/adaptive_snn_replays')/f'adaptive_seed{training_seed}_{weather}_traffic{seed}'
    output.mkdir(parents=True, exist_ok=False)
    label = f'Adaptive reservoir (training seed {training_seed})'
    frames, gifs = replay_frames(rollout, metrics, label)
    policy_id = 'adaptive_reservoir'
    step_s = 1/config.highway['policy_frequency']
    durations = [round(step_s*1000)]*len(gifs)
    durations[-1] = 2000
    gifs[0].save(output/f'{policy_id}.gif', save_all=True, append_images=gifs[1:],
                 duration=durations, loop=0)
    gifs[-1].save(output/'final_frame.png')
    rollout.history.to_csv(output/f'{policy_id}_history.csv', index=False)
    clean = {k: None if isinstance(v,float) and not np.isfinite(v) else v for k,v in metrics.items()}
    (output/f'{policy_id}_metrics.json').write_text(json.dumps(clean,indent=2,allow_nan=False)+'\n')
    config.save(output/'config.json')
    data = {'seed':seed,'weather':weather,'route_m':config.reward.route_distance_m,
            'deadline_s':config.highway['duration'],'step_s':step_s,'sensor_names':rollout.sensor_names,
            'agents':[{'id':policy_id,'label':label,'metrics':clean,'frames':frames}]}
    html = HTML.replace('Matched agent replays','Adaptive reservoir replay').replace(
        'Frozen policies from the 1,000-episode run. Same starting scenario; actions produce different traffic interactions.',
        f'Frozen adaptive-reservoir policy trained for 60 episodes (training seed {training_seed}). This replay reproduces a recorded held-out evaluation trip.')
    (output/'replay.html').write_text(html.replace('__DATA__',json.dumps(data,allow_nan=False)))
    (output/'manifest.json').write_text(json.dumps({
        'checkpoint':str(model_dir/'agent.pt'),
        'checkpoint_sha256':hashlib.sha256((model_dir/'agent.pt').read_bytes()).hexdigest(),
        'training_seed':training_seed,'traffic_seed':seed,'weather':weather,
        'evaluation_match_verified':True,'frozen_state_verified':True,
        'frame_count':len(frames),'capture_rate_hz':config.highway['policy_frequency'],
        'elapsed_s':metrics['elapsed_s'],'completed':metrics['completed'],'crashed':metrics['crashed']},indent=2)+'\n')
    (output/'README.md').write_text(f'''# Adaptive reservoir replay

Open `replay.html` in a browser for play/pause, frame stepping, and speed controls.
`adaptive_reservoir.gif` provides a standalone preview. The HTML embeds its frames and works offline.

Training seed: {training_seed}; traffic seed: {seed}; weather: {weather}.
Task: 2,250 m with a 120-second deadline. Outcome: completed={metrics['completed']}, crashed={metrics['crashed']}, elapsed={metrics['elapsed_s']} s.
The recorded metrics match the original evaluation row, and policy weights/thresholds stayed frozen.

Green car: the agent. Road view: simulator truth. Map: sensor evidence available to the policy.
Frames are sampled at {config.highway['policy_frequency']} Hz. Action labels describe the interval ending at that frame.
''')
    assert (output/'replay.html').read_text().count('data:image/png;base64,') == 2*len(frames)
    with Image.open(output/f'{policy_id}.gif') as gif:
        assert gif.n_frames == len(frames)
    print(json.dumps({'output':str(output),'frames':len(frames),'metrics':clean},indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-seed',type=int,default=42)
    parser.add_argument('--seed',type=int,default=100000)
    parser.add_argument('--weather',choices=['clear','rain','fog'],default='clear')
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args = parser.parse_args()
    export_replay(args.run,args.training_seed,args.seed,args.weather)
