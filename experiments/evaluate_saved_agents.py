"""Evaluate frozen SARSA/DQN/Dyna-DQN policies on an extended route.

Example (no retraining):
    .venv/bin/python -m experiments.evaluate_saved_agents --duration 120 --route-distance 2250

One process evaluates each policy. Results are saved incrementally. The original
model files are never overwritten; their hashes are checked after evaluation.
"""
import argparse
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import torch

from algorithms.dqn_agent import DQNAgent
from algorithms.sensor_agent import LinearExpectedSarsa
from environments.sensor_highway import ExperimentConfig
from experiments.agent_helpers import run_agent
from experiments.replay_agents import DEFAULT_RUN, POLICIES


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_policy(policy_id, label, filename, source, output, seeds, weather):
    torch.set_num_threads(1)
    agent = (LinearExpectedSarsa.load(source / filename) if policy_id == "sarsa"
             else DQNAgent.load(source / filename))
    config = ExperimentConfig.load(output / "config.json")
    records = []
    for condition in weather:
        config.weather.preset = condition
        for index, seed in enumerate(seeds):
            metrics, _ = run_agent(config, agent, seed=seed)
            records.append({"policy": label, "weather": condition, **metrics})
            if (index + 1) % 5 == 0 or index + 1 == len(seeds):
                pd.DataFrame(records).to_csv(output / f"{policy_id}_evaluation.csv", index=False)
                print(f"{label}: {condition}, {index + 1}/{len(seeds)} seeds", flush=True)
    print(f"Finished frozen-policy evaluation: {label}", flush=True)


def markdown_table(data):
    return "\n".join(["| " + " | ".join(data.columns) + " |",
                       "| " + " | ".join(["---"] * len(data.columns)) + " |"] +
                      ["| " + " | ".join(map(str, row)) + " |" for row in data.itertuples(index=False, name=None)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--route-distance", type=float, default=2250)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--weather", nargs="+", default=["clear", "rain", "fog"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("seeds must be positive")
    source = args.run.resolve()
    config = ExperimentConfig.load(source / "config.json")
    original_task = {"duration_s": config.highway["duration"], "route_distance_m": config.reward.route_distance_m}
    config.highway["duration"] = args.duration
    config.reward.route_distance_m = args.route_distance
    config.validate()
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or source / f"evaluation_{args.duration:g}s_{args.route_distance:g}m_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    config.save(output / "config.json")
    hashes = {filename: file_hash(source / filename) for _, _, filename in POLICIES}
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    metadata = {"status": "running", "source_run": str(source), "frozen_policies": True,
                "original_task": original_task, "duration_s": args.duration,
                "route_distance_m": args.route_distance, "test_seeds": seeds,
                "weather": args.weather, "model_sha256": hashes}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Results folder: {output}", flush=True)
    context = mp.get_context("spawn")
    jobs = [context.Process(target=evaluate_policy,
            args=(*policy, source, output, seeds, args.weather), name=policy[0]) for policy in POLICIES]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()
    failures = [job.name for job in jobs if job.exitcode != 0]
    if failures:
        metadata["status"] = "failed"
        metadata["failed_policies"] = failures
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        raise RuntimeError(f"Evaluation failed: {failures}")
    evaluation = pd.concat([pd.read_csv(output / f"{name}_evaluation.csv") for name, _, _ in POLICIES], ignore_index=True)
    evaluation.to_csv(output / "evaluation.csv", index=False)
    summary = evaluation.groupby("policy", sort=False).agg(
        episodes=("seed", "count"), completions=("completed", "sum"), crashes=("crashed", "sum"),
        completion_fraction=("completed", "mean"), crash_fraction=("crashed", "mean"),
        successful_finish_time_s=("completion_time_s", "mean"), mean_power_w=("mean_power_w", "mean"),
        mean_energy_j=("energy_j", "mean"), mean_distance_m=("distance_m", "mean"),
        mean_return=("return", "mean"), sensors_off_fraction=("mode_fraction_off", "mean"))
    summary.to_csv(output / "summary.csv")
    assert hashes == {filename: file_hash(source / filename) for filename in hashes}, "Source weights changed"
    metadata["status"] = "complete"
    metadata["model_hashes_unchanged"] = True
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    display = summary.reset_index()[["policy", "completions", "crashes", "successful_finish_time_s", "mean_power_w", "mean_energy_j"]].round(2)
    report = f"""# Frozen models: {args.duration:g}-second evaluation

The same saved policies trained for 1,000 episodes on a {original_task['route_distance_m']:g} m /
{original_task['duration_s']:g} s task were evaluated without learning. The new task is
**{args.route_distance:g} m within {args.duration:g} s**. A crash, off-road ending, or safe
finish can end a run earlier. Reward coefficients, sensor models, and traffic settings
are unchanged; remaining route/time features are normalized using the new task.

This tests transfer to a longer route, not newly trained 120-second agents.
Source model hashes were verified unchanged. Dyna's prediction model is not used
to choose actions during evaluation.

## Results

{markdown_table(display)}

Finish time averages successful episodes only; missing values mean no successes.
Power is mean per-episode sensing power. Energy includes failed episodes and can be
low because of early crashes. Neither measure includes training electricity.

Each policy used {len(seeds)} held-out traffic seeds across {', '.join(args.weather)}:
{len(seeds) * len(args.weather)} trials per policy. Seeds match the previous comparison.
Weather conditions reuse traffic seeds; these are not independent training runs.

## How to interpret the comparison

Longer routes expose agents to traffic for longer, so episode crash rates are not
an equal-exposure comparison with the original short task. Route progress reward
and cumulative costs also grow with trip length: raw return and total energy are
not directly comparable between the two task lengths. Inspect completion, power,
distance, and successful completion time together.

The policies were not retrained for the new horizon. Performance changes can reflect
that task shift as well as longer exposure. Only one training seed underlies these
models. Full trial data, matching configuration, and provenance are saved beside
this report in `evaluation.csv`, `config.json`, and `metadata.json`.
"""
    (output / "study_report.md").write_text(report)
    print(summary.to_string(), flush=True)
    print(f"Saved report: {output / 'study_report.md'}", flush=True)


if __name__ == "__main__":
    main()
