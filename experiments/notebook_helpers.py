"""Small, readable helpers shared by the initial sensor notebook."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environments.sensor_highway import SensorHighwayEnv


@dataclass
class Rollout:
    history: pd.DataFrame
    snapshots: list
    frames: list
    sensor_names: list
    radius_m: float


def sensor_mask(config, mode):
    """Fixed baselines; direction-specific experiments can also supply a mask."""
    choices = {
        "off": lambda s: False,
        "radar": lambda s: s.kind == "radar",
        "rgb": lambda s: s.kind == "rgb",
        "radar_rgb": lambda s: s.kind in {"radar", "rgb"},
        "all_on": lambda s: True,
        "front": lambda s: s.name in {"rgb_N", "event_N", "radar_N"},
    }
    if mode not in choices:
        raise ValueError(f"Choose one of {list(choices)}")
    return np.array([choices[mode](sensor) for sensor in config.sensors], dtype=np.int8)


def run_rollout(config, seed=42, mode="all_on", max_steps=None, capture=False, schedule=None):
    """Hold initial lane/speed. A schedule(time_s, config) can choose sensor masks.

    Diagnostics are for experiment analysis; they never select driving actions.
    Snapshots/frames include reset, so index zero intentionally has no sensing.
    """
    records, snapshots, frames = [], [], []
    with SensorHighwayEnv(config, render_mode="rgb_array" if capture else None) as env:
        observation, info = env.reset(seed=seed)
        names = env.sensor_names
        if capture:
            snapshots.append(observation)
            frames.append(env.render())
        idle = next(i for i, name in env.backend.action_type.actions.items() if name == "IDLE")
        fixed_mask = sensor_mask(config, mode)
        step = 0
        while max_steps is None or step < max_steps:
            mask = fixed_mask if schedule is None else np.asarray(schedule(info["time_s"], config), dtype=np.int8)
            observation, reward, terminated, truncated, info = env.step({"drive": idle, "sensors": mask})
            step += 1
            valid = observation["valid_cells"].astype(bool)
            records.append({
                "step": step, "time_s": info["time_s"], "reward": reward,
                "driving_reward": info["driving_reward"],
                "energy_penalty": info["reward_components"]["energy_penalty"],
                "uncertainty_penalty": info["reward_components"]["uncertainty_penalty"],
                "energy_j": info["energy_j"], "total_energy_j": info["total_energy_j"],
                "uncertainty": info["uncertainty"],
                "supported_fraction": float((observation["grid"][1][valid] > .05).mean()),
                "active_sensors": int(mask.sum()), "speed_m_s": float(observation["ego"][0]),
                "crashed": info["crashed"], "terminated": terminated, "truncated": truncated,
            })
            if capture:
                snapshots.append(observation)
                frames.append(env.render())
            if terminated or truncated:
                break
        if not records:
            raise ValueError("max_steps must allow at least one step")
    return Rollout(pd.DataFrame(records), snapshots, frames, names, config.grid.radius_m)


def plot_snapshot(rollout, index=-1):
    """Display simulator truth separately from the information available to policy."""
    observation = rollout.snapshots[index]
    actual_index = index % len(rollout.snapshots)
    time = 0 if actual_index == 0 else rollout.history.iloc[actual_index - 1].time_s
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].imshow(rollout.frames[index])
    axes[0, 0].set_title(f"Simulator visualization only — t={time:.1f}s")
    axes[0, 0].axis("off")
    radius = rollout.radius_m
    valid = observation["valid_cells"].astype(bool)
    for axis, channel, title in zip([axes[0, 1], axes[1, 0], axes[1, 1]], [0, 1, 2],
                                    ["Occupancy (unknown = 0.5)", "Evidence support", "Age (1 = stale / unknown)"]):
        values = np.ma.masked_where(~valid, observation["grid"][channel])
        artist = axis.imshow(values, extent=(-radius, radius, -radius, radius), vmin=0, vmax=1)
        axis.plot(0, 0, "rx")
        axis.set(xlabel="Right / E (m)", ylabel="Forward / N (m)", title=title)
        fig.colorbar(artist, ax=axis, shrink=.7)
    fig.tight_layout()
    return fig


def summarize(rollout):
    h = rollout.history
    return {"return": h.reward.sum(), "driving_return": h.driving_reward.sum(),
            "energy_j": h.energy_j.sum(), "mean_power_w": h.energy_j.sum() / h.time_s.iloc[-1],
            "mean_uncertainty": h.uncertainty.mean(),
            "mean_supported_fraction": h.supported_fraction.mean(),
            "elapsed_s": h.time_s.iloc[-1], "crashed": bool(h.crashed.iloc[-1]),
            "completed": bool(h.truncated.iloc[-1] and not h.terminated.iloc[-1])}
