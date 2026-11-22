"""Reproducible per-seed workers and summaries for the adaptive SNN notebook."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from algorithms.snn import AdaptiveExpectedSarsa, AdaptiveSNNSettings
from algorithms.sensor_agent import ObservationEncoder
from environments.sensor_highway import ExperimentConfig, SensorHighwayEnv
from experiments.agent_helpers import evaluate_agents

ROOT = Path(__file__).resolve().parents[1]
CONTROL = Path(os.environ.get('MORL_SNN_CONTROL', ROOT / 'runs/local/snn_expected_sarsa_120s')).resolve()
OUTPUT = Path(os.environ.get('MORL_SNN_OUTPUT', ROOT / 'runs/local/adaptive_snn_120s')).resolve()
SEEDS = [42, 1042, 2042]
TEST_SEEDS = list(range(100000, 100004))
WEATHER = ['clear', 'rain', 'fog']
VARIANTS = {'adaptive_reservoir': 'Adaptive reservoir',
            'prototype_control': 'Prototype control (no STDP)',
            'stdp_pretrained': 'STDP-pretrained prototypes'}
PRETRAIN_EPISODES = 20
PRETRAIN_PASSES = 3


def protocol():
    if not (CONTROL / 'metadata.json').is_file():
        raise FileNotFoundError('Run sensor_snn_expected_sarsa_120s.ipynb with RUN_TRAINING=True first')
    config = ExperimentConfig.load(CONTROL / 'config.json')
    assert config.highway['duration'] == 120 and config.reward.route_distance_m == 2250
    settings = AdaptiveSNNSettings(episodes=60)
    previous = json.loads((CONTROL / 'metadata.json').read_text())
    assert previous['snn_settings'] == {k: asdict(settings)[k] for k in previous['snn_settings']}
    # Legacy SNN code is unchanged: adaptive classes were appended below it.
    source = (ROOT / 'algorithms/snn.py').read_text()
    legacy = source.split('\n\n@dataclass\nclass AdaptiveSNNSettings')[0]
    snn_digests = {hashlib.sha256(content.encode()).hexdigest() for content in (source, legacy)}
    assert previous['sources_sha256']['algorithms/snn.py'] in snn_digests
    for path, digest in previous['sources_sha256'].items():
        if path != 'algorithms/snn.py':
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    import importlib.metadata
    assert all(importlib.metadata.version(p) == v for p, v in previous['versions'].items())
    data = {'config': asdict(config), 'settings': asdict(settings), 'train_seeds': SEEDS,
            'test_seeds': TEST_SEEDS, 'weather': WEATHER, 'pretrain_episodes': PRETRAIN_EPISODES,
            'pretrain_passes': PRETRAIN_PASSES, 'controls': str(CONTROL),
            'control_protocol_hash': previous['protocol_hash'],
            'versions': previous['versions'],
            'sources_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                               for p in ['algorithms/snn.py', 'experiments/adaptive_snn_study.py']},
            'note': 'Diehl-Cook-inspired current-based LIF, not a full replication. Readout uses Adam TD. '
                    'Prototype variants see 20 extra random-policy episodes per seed; adaptive reservoir does not. '
                    'Prototype thresholds are calibrated then frozen. Concurrent runtimes are descriptive.'}
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return config, settings, data, digest


def initialize():
    config, settings, data, digest = protocol()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/'metadata.json'
    if path.exists():
        assert json.loads(path.read_text())['protocol_hash'] == digest, 'Use a new output directory for a changed protocol'
    else:
        path.write_text(json.dumps({**data, 'protocol_hash': digest}, indent=2)+'\n')
        config.save(OUTPUT/'config.json')
    return config, settings


def collect_corpus(config, seed, folder):
    array_path = folder/'pretraining_features.npy'
    if array_path.exists():
        return np.load(array_path)
    encoder = ObservationEncoder(config)
    rng = np.random.default_rng(seed+10000)
    observations, records = [], []
    with SensorHighwayEnv(config) as env:
        obs, _ = env.reset(seed=seed)
        agent = AdaptiveExpectedSarsa(len(encoder(obs)), feature_schema=encoder.feature_schema, seed=seed)
        for episode in range(PRETRAIN_EPISODES):
            traffic_seed = seed + 10000 + episode
            assert traffic_seed not in TEST_SEEDS
            obs, _ = env.reset(seed=traffic_seed)
            steps = 0
            while True:
                features = encoder(obs)
                observations.append(features)
                action = agent.select(features, 1., rng)
                obs, _, terminated, truncated, info = env.step(agent.decode(action, env))
                steps += 1
                if terminated or truncated:
                    break
            records.append({'episode': episode+1, 'seed': traffic_seed, 'steps': steps,
                            'elapsed_s': info['time_s'], 'route_completed': info['route_completed']})
    corpus = np.asarray(observations, dtype=np.float32)
    pd.DataFrame(records).to_csv(folder/'pretraining_episodes.csv', index=False)
    np.save(array_path, corpus)
    return corpus


def train(config, settings, agent, seed):
    encoder = ObservationEncoder(config)
    rng = np.random.default_rng(seed)
    records = []
    start = time.perf_counter()
    frozen = {k: v.clone() for k, v in agent.network.state_dict().items() if not k.startswith('readout.')}
    with SensorHighwayEnv(config) as env:
        for episode in range(settings.episodes):
            epsilon = settings.epsilon_start + (settings.epsilon_end-settings.epsilon_start)*episode/max(1,settings.episodes-1)
            obs, _ = env.reset(seed=seed+episode)
            features = encoder(obs)
            rewards, tds, diagnostics = [], [], []
            while True:
                action = agent.select(features, epsilon, rng)
                obs, reward, terminated, truncated, info = env.step(agent.decode(action, env))
                next_features = encoder(obs)
                terminal = terminated or (truncated and config.reward.route_distance_m is not None)
                tds.append(abs(agent.update(features, action, reward, next_features, terminal, epsilon)))
                diagnostics.append(agent.last_metrics.copy())
                rewards.append(reward)
                features = next_features
                if terminated or truncated:
                    break
            records.append({'episode': episode+1, 'seed': seed+episode, 'epsilon': epsilon,
                            'return': sum(rewards), 'steps': len(rewards), 'mean_abs_td': np.mean(tds),
                            'energy_j': info['total_energy_j'], 'elapsed_s': info['time_s'],
                            'route_completed': info['route_completed'], 'crashed': info['crashed'],
                            **{k: np.mean([d[k] for d in diagnostics]) for k in diagnostics[0]}})
            if (episode+1)%10 == 0:
                print(f'{agent.variant} seed {seed}: {episode+1}/{settings.episodes}; '
                      f'return={sum(rewards):.2f}; wall={time.perf_counter()-start:.1f}s', flush=True)
    for k, value in frozen.items():
        torch.testing.assert_close(value, agent.network.state_dict()[k], rtol=0, atol=0)
    return pd.DataFrame(records)


def run_seed(seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config, settings = initialize()
    folder = OUTPUT/f'seed{seed}'
    folder.mkdir(exist_ok=True)
    corpus = collect_corpus(config, seed, folder)
    encoder = ObservationEncoder(config)
    with SensorHighwayEnv(config) as env:
        obs, _ = env.reset(seed=seed)
        probe = encoder(obs)
    orders = [np.random.default_rng(seed+30000+i).permutation(len(corpus)) for i in range(PRETRAIN_PASSES)]
    for variant, label in VARIANTS.items():
        target = folder/variant
        target.mkdir(exist_ok=True)
        if (target/'evaluation.csv').exists():
            print(f'Using completed {variant}, seed {seed}', flush=True)
            continue
        if (target/'agent.pt').exists() and (target/'training.csv').exists():
            agent = AdaptiveExpectedSarsa.load(target/'agent.pt')
        else:
            agent = AdaptiveExpectedSarsa(len(probe), settings, feature_schema=encoder.feature_schema,
                                        seed=seed, variant=variant)
            before = agent.network.input.weight.clone()
            if variant != 'adaptive_reservoir':
                pretraining = []
                for pass_index, order in enumerate(orders):
                    diagnostics = [agent.pretrain_step(corpus[i]) for i in order]
                    pretraining.append({'pass': pass_index+1, 'observations': len(corpus),
                                        **{k: np.mean([d[k] for d in diagnostics]) for k in diagnostics[0]}})
                pd.DataFrame(pretraining).to_csv(target/'pretraining.csv', index=False)
            pd.Series({'hidden_weight_l2_change': float(torch.linalg.vector_norm(agent.network.input.weight-before)),
                       'threshold_offset_mean': float(agent.network.threshold_offset.mean()),
                       'pretraining_observations': agent.pretraining_updates}).to_csv(target/'representation.csv')
            history = train(config, settings, agent, seed)
            history.to_csv(target/'training.csv', index=False)
            agent.save(target/'agent.pt')
        restored = AdaptiveExpectedSarsa.load(target/'agent.pt')
        np.testing.assert_array_equal(restored.values(probe), agent.values(probe))
        before = {k: v.clone() for k,v in agent.network.state_dict().items()}
        updates = agent.q_updates
        result = evaluate_agents(config, {label: agent}, TEST_SEEDS, WEATHER)
        for k, value in before.items():
            torch.testing.assert_close(value, agent.network.state_dict()[k], rtol=0, atol=0)
        assert agent.q_updates == updates
        result.assign(variant=variant, training_seed=seed).to_csv(target/'evaluation.csv', index=False)
        print(f'FINISHED {variant} seed {seed}: {int(result.completed.sum())}/12 completions', flush=True)


def summarize():
    evaluations, training = [], []
    for seed in SEEDS:
        for variant, label in VARIANTS.items():
            folder = OUTPUT/f'seed{seed}'/variant
            evaluations.append(pd.read_csv(folder/'evaluation.csv'))
            training.append(pd.read_csv(folder/'training.csv').assign(policy=label, training_seed=seed))
    # Import archived controls rather than retrain them; source/config/version checks above apply.
    control = pd.read_csv(CONTROL/'evaluation.csv')
    control = control[control.policy_id.isin(['linear', 'reservoir'])].copy()
    control['provenance'] = 'archived matched control'
    new = pd.concat(evaluations, ignore_index=True).assign(provenance='new adaptive experiment')
    combined = pd.concat([control, new], ignore_index=True)
    per_seed = combined.groupby(['policy','training_seed']).agg(
        completion=('completed','mean'), crash=('crashed','mean'), mean_return=('return','mean'),
        sensor_power_w=('mean_power_w','mean'), successful_time_s=('completion_time_s','mean'))
    summary = per_seed.groupby('policy').agg(completion_mean=('completion','mean'),
        completion_sd=('completion','std'), crash_mean=('crash','mean'),
        return_mean=('mean_return','mean'), sensor_power_w=('sensor_power_w','mean'))
    combined.to_csv(OUTPUT/'evaluation.csv', index=False)
    pd.concat(training, ignore_index=True).to_csv(OUTPUT/'training.csv', index=False)
    per_seed.to_csv(OUTPUT/'per_training_seed.csv')
    summary.to_csv(OUTPUT/'summary.csv')
    return summary, per_seed, combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    run_seed(parser.parse_args().seed)
