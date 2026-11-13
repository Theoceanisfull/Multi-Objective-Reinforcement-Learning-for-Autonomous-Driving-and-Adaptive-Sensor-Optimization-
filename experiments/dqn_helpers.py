"""Train either neural agent; evaluation reuses experiments.agent_helpers."""
import numpy as np
import pandas as pd

from algorithms.dqn_agent import DQNAgent, DynaDQNAgent
from algorithms.sensor_agent import ObservationEncoder, SENSING_MODES
from environments.sensor_highway import SensorHighwayEnv


def train_dqn(config, settings, seed=42, modes=SENSING_MODES, dyna=False):
    settings.validate()
    encoder = ObservationEncoder(config)
    rng = np.random.default_rng(seed)
    records = []
    with SensorHighwayEnv(config) as env:
        obs, _ = env.reset(seed=seed)
        agent_class = DynaDQNAgent if dyna else DQNAgent
        agent = agent_class(len(encoder(obs)), settings, modes, encoder.feature_schema, seed)
        for episode in range(settings.episodes):
            epsilon = settings.epsilon_start + (settings.epsilon_end - settings.epsilon_start) * episode / max(1, settings.episodes - 1)
            obs, _ = env.reset(seed=seed + episode)
            features = encoder(obs)
            total_return, energy, steps = 0.0, 0.0, 0
            losses = {name: [] for name in ("q_loss", "model_loss", "planning_loss")}
            while True:
                action = agent.select(features, epsilon, rng)
                obs, reward, terminated, truncated, info = env.step(agent.decode(action, env))
                next_features = encoder(obs)
                # Same rule as SARSA: finite-route deadlines stop future value;
                # a legacy continuing-task time limit still bootstraps.
                terminal = terminated or (truncated and config.reward.route_distance_m is not None)
                metrics = agent.observe(features, action, reward, next_features, terminal)
                for name, value in metrics.items():
                    losses[name].append(value)
                total_return += reward
                energy += info["energy_j"]
                steps += 1
                features = next_features
                if terminated or truncated:
                    break
            records.append({"episode": episode + 1, "seed": seed + episode,
                            "epsilon": epsilon, "return": total_return, "energy_j": energy,
                            "steps": steps, "crashed": info["crashed"],
                            "route_completed": info["route_completed"], "elapsed_s": info["time_s"],
                            "distance_m": info["distance_m"], "q_updates": agent.q_updates,
                            **{name: float(np.mean(values)) if values else np.nan
                               for name, values in losses.items()}})
            if (episode + 1) % 20 == 0 or episode + 1 == settings.episodes:
                print(f"{agent_class.__name__}: episode {episode + 1}/{settings.episodes}, "
                      f"return={total_return:.2f}, epsilon={epsilon:.2f}", flush=True)
    return agent, pd.DataFrame(records)
