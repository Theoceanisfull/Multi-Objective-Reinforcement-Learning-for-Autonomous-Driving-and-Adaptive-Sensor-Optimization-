"""Training, held-out evaluation, and captured rollouts for the agent notebook."""
import copy
import numpy as np
import pandas as pd
from environments.sensor_highway import SensorHighwayEnv
from algorithms.sensor_agent import ObservationEncoder, LinearExpectedSarsa, SENSING_MODES, DRIVE_NAMES
from experiments.notebook_helpers import Rollout


def train_agent(config, settings, seed=42, modes=SENSING_MODES):
    if settings.episodes <= 0:
        raise ValueError("episodes must be positive")
    if not (0 < settings.alpha <= 1 and 0 <= settings.gamma < 1
            and 0 <= settings.epsilon_end <= settings.epsilon_start <= 1):
        raise ValueError("Invalid learning rate, discount, or exploration settings")
    rng = np.random.default_rng(seed)
    encoder = ObservationEncoder(config)
    records = []
    with SensorHighwayEnv(config) as env:
        obs, _ = env.reset(seed=seed)
        agent = LinearExpectedSarsa(len(encoder(obs)), modes=modes, feature_schema=encoder.feature_schema)
        for episode in range(settings.episodes):
            epsilon = settings.epsilon_start + (settings.epsilon_end - settings.epsilon_start) * episode / max(1, settings.episodes - 1)
            obs, _ = env.reset(seed=seed + episode)
            features = encoder(obs)
            total_return, energy, total_uncertainty, steps, total_td = 0., 0., 0., 0, 0.
            while True:
                index = agent.select(features, epsilon, rng)
                obs, reward, terminated, truncated, info = env.step(agent.decode(index, env))
                next_features = encoder(obs)
                # Route deadlines are failures in a finite task; legacy time limits bootstrap.
                terminal_target = terminated or (truncated and config.reward.route_distance_m is not None)
                td = agent.update(features, index, reward, next_features, terminal_target,
                                  epsilon, settings.alpha, settings.gamma)
                total_return += reward
                energy += info["energy_j"]
                total_uncertainty += info["uncertainty"]
                total_td += abs(td)
                steps += 1
                features = next_features
                if terminated or truncated:
                    break
            records.append({"episode": episode + 1, "seed": seed + episode,
                            "epsilon": epsilon, "return": total_return, "energy_j": energy,
                            "mean_uncertainty": total_uncertainty / steps,
                            "mean_abs_td": total_td / steps, "steps": steps,
                            "crashed": info["crashed"], "route_completed": info["route_completed"],
                            "elapsed_s": info["time_s"], "distance_m": info["distance_m"]})
            if (episode + 1) % 20 == 0 or episode + 1 == settings.episodes:
                print(f"{'+'.join(modes)}: episode {episode+1}/{settings.episodes}, "
                      f"return={total_return:.2f}, epsilon={epsilon:.2f}", flush=True)
    return agent, pd.DataFrame(records)


def run_agent(config, agent=None, seed=100000, capture=False, baseline=None):
    """baseline: None for learned agent, 'idle_all_on', or 'random_all_on'."""
    encoder = ObservationEncoder(config)
    if agent is not None and agent.feature_schema != encoder.feature_schema:
        raise ValueError("Agent feature schema does not match the environment task configuration")
    rng = np.random.default_rng(seed + 1000000)
    snapshots, frames, records = [], [], []
    mode_counts = {}
    drive_counts = {}
    with SensorHighwayEnv(config, render_mode="rgb_array" if capture else None) as env:
        obs, info = env.reset(seed=seed)
        start_x = float(env.backend.vehicle.position[0])  # Evaluation metric only.
        if capture:
            snapshots.append(obs)
            frames.append(env.render())
        total_return, driving_return, uncertainty_sum, steps = 0., 0., 0., 0
        while True:
            if baseline is None:
                if agent is None:
                    raise ValueError("Supply a trained agent or a baseline")
                index = agent.select(encoder(obs), 0., rng)
                action = agent.decode(index, env)
                drive_name, mode = agent.actions[index]
            else:
                if baseline not in {"idle_all_on", "random_all_on"}:
                    raise ValueError("Unknown baseline")
                drive_name = "IDLE"
                if baseline == "random_all_on":
                    drive_name = str(rng.choice(list(env.backend.action_type.actions.values())))
                drive_id = next(i for i, name in env.backend.action_type.actions.items() if name == drive_name)
                mode = "all_on"
                action = {"drive": drive_id, "sensors": np.ones(len(config.sensors), np.int8)}
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_return += reward
            driving_return += info["driving_reward"]
            uncertainty_sum += info["uncertainty"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            drive_counts[drive_name] = drive_counts.get(drive_name, 0) + 1
            records.append({"step": steps, "time_s": info["time_s"], "drive": drive_name,
                            "sensing": mode, "reward": reward, "energy_j": info["energy_j"],
                            "uncertainty": info["uncertainty"], "crashed": info["crashed"],
                            "speed_m_s": float(obs["ego"][0]), "distance_m": info["distance_m"],
                            "total_energy_j": info["total_energy_j"],
                            "route_completed": info["route_completed"],
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            **info["reward_components"]})
            if capture:
                snapshots.append(obs)
                frames.append(env.render())
            if terminated or truncated:
                break
        distance = float(env.backend.vehicle.position[0]) - start_x
        metrics = {"seed": seed, "return": total_return, "driving_return": driving_return,
                   "energy_j": info["total_energy_j"], "mean_power_w": info["total_energy_j"] / info["time_s"],
                   "mean_uncertainty": uncertainty_sum / steps, "crashed": info["crashed"],
                   "completed": (info["route_completed"] if config.reward.route_distance_m is not None
                                 else bool(truncated and not terminated)),
                   "completion_time_s": info["time_s"] if info["route_completed"] else np.nan, "distance_m": distance,
                   "elapsed_s": info["time_s"], "mean_speed_m_s": distance / info["time_s"], "steps": steps}
        metrics.update({f"mode_fraction_{mode}": mode_counts.get(mode, 0) / steps for mode in SENSING_MODES})
        metrics.update({f"drive_fraction_{drive}": drive_counts.get(drive, 0) / steps for drive in DRIVE_NAMES})
        rollout = Rollout(pd.DataFrame(records), snapshots, frames, env.sensor_names, config.grid.radius_m)
    return metrics, rollout


def evaluate_agents(config, agents, seeds, weather_presets):
    records = []
    for weather in weather_presets:
        trial = copy.deepcopy(config)
        trial.weather.preset = weather
        for name, agent in agents.items():
            baseline = name if name in {"idle_all_on", "random_all_on"} else None
            for seed in seeds:
                metrics, _ = run_agent(trial, agent, seed=seed, baseline=baseline)
                records.append({"policy": name, "weather": weather, **metrics})
        print(f"Evaluated {weather}", flush=True)
    return pd.DataFrame(records)
