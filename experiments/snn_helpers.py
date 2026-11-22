"""Online spiking Expected SARSA training; reuse shared frozen evaluation."""
import time

import numpy as np
import pandas as pd

from algorithms.sensor_agent import ObservationEncoder, SENSING_MODES
from algorithms.snn import SpikingExpectedSarsa
from environments.sensor_highway import SensorHighwayEnv


def train_snn(config, settings, seed=42, learning_rule='surrogate', modes=SENSING_MODES):
    settings.validate()
    encoder = ObservationEncoder(config)
    rng = np.random.default_rng(seed)
    records = []
    start = time.perf_counter()
    with SensorHighwayEnv(config) as env:
        obs, _ = env.reset(seed=seed)
        agent = SpikingExpectedSarsa(len(encoder(obs)), settings, modes, encoder.feature_schema,
                                    seed, learning_rule)
        for episode in range(settings.episodes):
            epsilon = settings.epsilon_start + (settings.epsilon_end - settings.epsilon_start) * episode / max(1, settings.episodes - 1)
            obs, _ = env.reset(seed=seed + episode)
            features = encoder(obs)
            total_return, energy, steps, total_td = 0., 0., 0, 0.
            metrics = []
            while True:
                action = agent.select(features, epsilon, rng)
                obs, reward, terminated, truncated, info = env.step(agent.decode(action, env))
                next_features = encoder(obs)
                terminal = terminated or (truncated and config.reward.route_distance_m is not None)
                td = agent.update(features, action, reward, next_features, terminal, epsilon)
                metrics.append(agent.last_metrics.copy())
                total_return += reward
                energy += info['energy_j']
                total_td += abs(td)
                steps += 1
                features = next_features
                if terminated or truncated:
                    break
            records.append({'episode': episode + 1, 'seed': seed + episode, 'epsilon': epsilon,
                            'return': total_return, 'energy_j': energy, 'steps': steps,
                            'mean_abs_td': total_td / steps, 'crashed': info['crashed'],
                            'route_completed': info['route_completed'], 'elapsed_s': info['time_s'],
                            'distance_m': info['distance_m'], 'q_updates': agent.q_updates,
                            'training_wall_s': time.perf_counter() - start,
                            **{k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}})
            if (episode + 1) % 10 == 0 or episode + 1 == settings.episodes:
                print(f'SNN {learning_rule}, seed {seed}: {episode+1}/{settings.episodes}, '
                      f'return={total_return:.2f}, elapsed={time.perf_counter()-start:.1f}s', flush=True)
    return agent, pd.DataFrame(records)
