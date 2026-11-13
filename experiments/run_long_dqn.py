"""Train the three independent policies concurrently, using a prepared run folder.

The long-training notebook creates config.json and run_settings.json. This runner
reads them, starts one CPU process per policy, and saves each finished model and
history. Run the notebook afterward to load those results and evaluate them.
"""
import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path

import torch

from algorithms.dqn_agent import DQNSettings
from algorithms.sensor_agent import AgentSettings
from environments.sensor_highway import ExperimentConfig
from experiments.agent_helpers import train_agent
from experiments.dqn_helpers import train_dqn


def train_policy(name, output):
    """A process owns its environment, random generators, and learned weights."""
    torch.set_num_threads(1)
    config = ExperimentConfig.load(output / "config.json")
    run = json.loads((output / "run_settings.json").read_text())
    seed = run["training_seed"]
    if name == "sarsa":
        settings = AgentSettings(**run["sarsa_settings"])
        agent, history = train_agent(config, settings, seed=seed)
        agent.save(output / "sarsa.npz")
    else:
        settings = DQNSettings(**run["dqn_settings"])
        agent, history = train_dqn(config, settings, seed=seed, dyna=(name == "dyna_dqn"))
        agent.save(output / f"{name}.pt")
        if name == "dyna_dqn":
            torch.save(agent.model.state_dict(), output / "dyna_prediction_model.pt")
    history.to_csv(output / f"{name}_training.csv", index=False)
    (output / f"{name}_training_complete.json").write_text(json.dumps({
        "policy": name, "settings": asdict(settings), "training_seed": seed,
        "episodes": len(history), "real_steps": int(history.steps.sum()),
    }, indent=2))
    print(f"Saved completed {name} training to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    # Validate inputs before starting any expensive training.
    run = json.loads((output / "run_settings.json").read_text())
    DQNSettings(**run["dqn_settings"]).validate()
    ExperimentConfig.load(output / "config.json").validate()
    context = mp.get_context("spawn")
    names = ("dqn", "dyna_dqn", "sarsa")
    completed = [name for name in names if (output / f"{name}_training_complete.json").exists()]
    if completed:
        raise FileExistsError(f"Already completed: {completed}; use saved results or a new run folder")
    processes = []
    for name in names:
        process = context.Process(target=train_policy, args=(name, output), name=name)
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failures = [p.name for p in processes if p.exitcode != 0]
    if failures:
        raise RuntimeError(f"Training failed for: {', '.join(failures)}")
    print("All three policies trained and saved. Ready for notebook evaluation.", flush=True)


if __name__ == "__main__":
    main()
