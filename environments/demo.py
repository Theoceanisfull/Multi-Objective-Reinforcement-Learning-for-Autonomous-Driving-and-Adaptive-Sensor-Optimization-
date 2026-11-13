"""Run an inspectable sensing baseline: python -m environments.demo --help."""
import argparse
import csv
from pathlib import Path
import numpy as np
from .sensor_highway import ExperimentConfig, SensorHighwayEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Full experiment JSON; otherwise use defaults.")
    parser.add_argument("--weather", choices=["clear", "night", "rain", "fog", "glare", "random"])
    parser.add_argument("--sensing", choices=["all_on", "radar_rgb", "front", "off"], default="all_on")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("runs/environment/demo"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    config = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    if args.weather:
        config.weather.preset = args.weather
    args.output.mkdir(parents=True, exist_ok=True)
    config.save(args.output / "config.json")
    records = []
    with SensorHighwayEnv(config, render_mode="rgb_array") as env:
        obs, info = env.reset(seed=args.seed)
        mask = np.array([
            args.sensing == "all_on"
            or (args.sensing == "radar_rgb" and s.kind in {"radar", "rgb"})
            or (args.sensing == "front" and s.name in {"rgb_N", "event_N", "radar_N"})
            for s in config.sensors], dtype=np.int8)
        # IDLE maintains targets. No learning or automatic safety claim in this demo.
        idle = next(i for i, name in env.backend.action_type.actions.items() if name == "IDLE")
        print("Active sensors:", [name for name, on in zip(env.sensor_names, mask) if on])
        for step in range(args.steps):
            obs, reward, terminated, truncated, info = env.step({"drive": idle, "sensors": mask})
            records.append({"step": step + 1, "time_s": info["time_s"], "reward": reward,
                            "energy_j": info["energy_j"], "uncertainty": info["uncertainty"],
                            "crashed": info["crashed"], "terminated": terminated, "truncated": truncated})
            if terminated or truncated:
                break
        frame = env.render()
        np.savez_compressed(args.output / "observation.npz", **obs)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        axes[0, 0].imshow(frame)
        axes[0, 0].set_title("Simulator view (privileged visualization)")
        axes[0, 0].axis("off")
        radius = config.grid.radius_m
        for ax, channel, title in zip([axes[0, 1], axes[1, 0], axes[1, 1]], [0, 1, 2],
                                      ["Occupancy: unknown = 0.5", "Evidence support", "Normalized measurement age"]):
            values = np.ma.masked_where(obs["valid_cells"] == 0, obs["grid"][channel])
            artist = ax.imshow(values, extent=(-radius, radius, -radius, radius), vmin=0, vmax=1)
            ax.plot(0, 0, "rx")
            ax.set(xlabel="Right / E (m)", ylabel="Forward / N (m)", title=title)
            fig.colorbar(artist, ax=ax, shrink=.75)
        fig.tight_layout()
        fig.savefig(args.output / "preview.png", dpi=130)
        plt.close(fig)
    with (args.output / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"{len(records)} decisions; return={sum(r['reward'] for r in records):.3f}; "
          f"energy={info['total_energy_j']:.1f} J; crashed={info['crashed']}")
    print("Artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
