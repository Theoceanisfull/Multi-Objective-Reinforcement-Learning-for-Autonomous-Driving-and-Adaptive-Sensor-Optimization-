"""Controlled object-level sensing diagnostics and frozen-policy ablations."""
import copy
import numpy as np
import pandas as pd
from algorithms.sensor_agent import ObservationEncoder
from environments.sensor_highway import SensorHighwayEnv, WeatherConfig
from environments.sensor_highway.backend import VehicleTruth, Scene
from environments.sensor_highway.sensors import SensorSuite, wrap_angle
from environments.sensor_highway.weather import WeatherModel

MODALITIES = ('rgb', 'event', 'radar', 'lidar')


def modality_mask(config, condition):
    if condition not in {'all_on', 'off', *MODALITIES, *('without_' + k for k in MODALITIES)}:
        raise ValueError(f'Unknown sensing condition: {condition}')
    return np.array([condition == 'all_on' or s.kind == condition
                     or (condition.startswith('without_') and s.kind != condition[8:])
                     for s in config.sensors], dtype=np.int8)


def measure_sensor(sensor, distance, relative_speed, trials=300, seed=42):
    """Single unobstructed target in clear weather; suppress false positives.

    Calls _capture directly: no acquisition scheduling, latency, or map fusion.
    Errors are conditional on detection; NaN means absent measurement/no returns.
    """
    if trials <= 0 or distance <= 0:
        raise ValueError('trials and distance must be positive')
    sensor = copy.deepcopy(sensor)
    sensor.false_positive_rate_hz = 0
    def car(x, speed):
        return VehicleTruth(np.array([x, 0.]), np.array([speed, 0.]), 0., 5., 2.)
    scene = Scene(car(0, 25), (car(distance, 25 + relative_speed),))
    suite = SensorSuite([sensor], WeatherModel(WeatherConfig(preset='clear'),
                        np.random.default_rng(seed)), np.random.default_rng(seed + 1))
    ds = [d for _ in range(trials) for d in suite._capture(0, 0., scene).detections]
    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values else np.nan
    fraction = len(ds) / trials
    return {'kind': sensor.kind, 'range_setting_m': sensor.range_m,
            'distance_m': distance, 'relative_speed_m_s': relative_speed,
            'trials': trials, 'detections': len(ds), 'detection_fraction': fraction,
            'binomial_se': np.sqrt(fraction * (1 - fraction) / trials),
            'bearing_rmse_deg': rmse([np.rad2deg(wrap_angle(d.bearing)) for d in ds]),
            'range_rmse_m': rmse([d.range_m - distance for d in ds if d.range_m is not None]),
            'velocity_rmse_m_s': rmse([d.radial_velocity - relative_speed for d in ds
                                     if d.radial_velocity is not None])}


def frozen_policy_rollout(config, agent, condition, seed):
    """Freeze weights, retain observation-driven driving, override sensing only."""
    encoder = ObservationEncoder(config)
    if encoder.feature_schema != agent.feature_schema:
        raise ValueError('Agent feature schema does not match task')
    rng = np.random.default_rng(seed + 1_000_000)
    records = []
    with SensorHighwayEnv(config) as env:
        obs, _ = env.reset(seed=seed)
        mask = modality_mask(config, condition)
        while True:
            index = agent.select(encoder(obs), 0., rng)
            action = agent.decode(index, env)
            action['sensors'] = mask.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            valid = obs['valid_cells'].astype(bool)
            records.append({'time_s': info['time_s'], 'drive': agent.actions[index][0],
                            'reward': reward, 'uncertainty_proxy': info['uncertainty'],
                            'supported_fraction': float((obs['grid'][1][valid] > .05).mean())})
            if terminated or truncated:
                break
        history = pd.DataFrame(records)
        metrics = {'condition': condition, 'seed': seed, 'weather': config.weather.preset,
                   'completed': info['route_completed'], 'crashed': info['crashed'],
                   'elapsed_s': info['time_s'], 'distance_m': info['distance_m'],
                   'completion_time_s': info['time_s'] if info['route_completed'] else np.nan,
                   'mean_speed_m_s': info['distance_m'] / info['time_s'],
                   'energy_j': info['total_energy_j'],
                   'mean_power_w': info['total_energy_j'] / info['time_s'],
                   'mean_supported_fraction': history.supported_fraction.mean(),
                   'mean_uncertainty_proxy': history.uncertainty_proxy.mean()}
    return metrics, history
