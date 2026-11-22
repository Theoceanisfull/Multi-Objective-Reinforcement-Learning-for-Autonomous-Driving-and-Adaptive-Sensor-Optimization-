"""Run a frozen policy comparison across the three route stages.

Complete the overtaking training cohort with DQN and Dyna-DQN. On the long
original route, test frozen policies with forced sensing and train a paired
neural cohort with only the sensing-energy reward coefficient set to zero.
Evaluation always uses the original rewards and reports actual sensing energy.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch

from algorithms.dqn_agent import DQNAgent, DQNSettings
from algorithms.sensor_agent import LinearExpectedSarsa
from environments.sensor_highway import ExperimentConfig
from experiments.agent_helpers import run_agent


ROOT = Path(__file__).resolve().parents[1]
SHORT = ROOT / "runs/comparisons/dqn_long_training_20261112_170403_004536"
LONG_NEURAL = ROOT / "runs/training/dqn_120s_2250m_1000ep_20261122_094440_534028"
LONG_SARSA = ROOT / "runs/training/time_completion_120s_1000ep_20261121_204241_241658"
PASSING_SARSA = ROOT / "runs/training/overtaking_120s_2250m_1000ep_20261122_100614_534843"
NEURAL = ("dqn", "dyna_dqn")
WEATHER = ("clear", "rain", "fog")
SEED_PANELS = {"matched": tuple(range(100000, 100004)), "additional": tuple(range(200000, 200004))}


def load_policy(path):
    return LinearExpectedSarsa.load(path) if path.suffix == ".npz" else DQNAgent.load(path)


def evaluate_policy(config, checkpoint, forced_sensing=None):
    torch.set_num_threads(1)
    policy = load_policy(checkpoint)
    rows = []
    for weather in WEATHER:
        trial = copy.deepcopy(config)
        trial.weather.preset = weather
        for seed_panel, seeds in SEED_PANELS.items():
            for seed in seeds:
                metrics, _ = run_agent(trial, policy, seed=seed, forced_sensing=forced_sensing)
                rows.append({"weather": weather, "seed_panel": seed_panel, **metrics})
    return pd.DataFrame(rows)


def summarize(rows, keys):
    metrics = {
        "trials": ("completed", "size"), "finishes": ("completed", "sum"),
        "crashes": ("crashed", "sum"), "mean_finish_s": ("completion_time_s", "mean"),
        "mean_power_w": ("mean_power_w", "mean"), "mean_energy_j": ("energy_j", "mean"),
        "mean_distance_m": ("distance_m", "mean"), "mean_off_fraction": ("mode_fraction_off", "mean"),
    }
    if "slow_leaders_passed" in rows:
        metrics["mean_leaders_passed"] = ("slow_leaders_passed", "mean")
    overall = rows.groupby(keys, sort=True).agg(**metrics).reset_index().assign(seed_panel="all")
    panels = rows.groupby([*keys, "seed_panel"], sort=True).agg(**metrics).reset_index()
    return pd.concat([overall, panels], ignore_index=True)


def write_csv(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_evaluations(output, name, frames, keys):
    rows = pd.concat(frames, ignore_index=True).sort_values([*keys, "seed_panel", "weather", "seed"])
    write_csv(rows, output / f"{name}_trials.csv")
    write_csv(summarize(rows, keys), output / f"{name}_summary.csv")
    return rows


def start_training(path, config, settings, seed):
    path.mkdir()
    config.save(path / "config.json")
    (path / "run_settings.json").write_text(json.dumps({
        "dqn_settings": settings, "training_seed": seed,
    }, indent=2))
    with (path / "training.log").open("w") as log:
        return subprocess.Popen([
            sys.executable, "-u", "-m", "experiments.run_long_dqn",
            "--output", str(path), "--policies", *NEURAL,
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def run(output, workers):
    configs = {name: ExperimentConfig.load(path / "config.json") for name, path in (
        ("short", SHORT), ("long", LONG_NEURAL), ("overtaking", PASSING_SARSA),
    )}
    assert [(cfg.reward.route_distance_m, cfg.highway["duration"]) for cfg in configs.values()] == [
        (300, 16), (2250, 120), (2250, 120),
    ]
    assert configs["overtaking"].highway.get("slow_lead_scenario")
    assert not configs["long"].highway.get("slow_lead_scenario")
    old_info = json.loads((SHORT / "run_settings.json").read_text())
    long_info = json.loads((LONG_NEURAL / "run_settings.json").read_text())
    settings, seed = long_info["dqn_settings"], long_info["training_seed"]
    assert settings == old_info["dqn_settings"] and seed == old_info["training_seed"]
    DQNSettings(**settings).validate()
    assert not set().union(*SEED_PANELS.values()).intersection(range(seed, seed + settings["episodes"]))

    checkpoints = {
        "short": {"sarsa": SHORT / "sarsa.npz", **{p: SHORT / f"{p}.pt" for p in NEURAL}},
        "long": {"sarsa": LONG_SARSA / "adaptive_agent.npz", **{p: LONG_NEURAL / f"{p}.pt" for p in NEURAL}},
        "overtaking": {"sarsa": PASSING_SARSA / "adaptive_agent.npz"},
    }
    for paths in checkpoints.values():
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
    (output / "study_settings.json").write_text(json.dumps({
        "training_seed": seed, "dqn_settings": settings, "seed_panels": SEED_PANELS,
        "weather": WEATHER, "existing_checkpoints": {
            stage: {name: str(p.relative_to(ROOT)) for name, p in cohort.items()}
            for stage, cohort in checkpoints.items()
        },
        "reward_ablation": {"changed_field": "energy_weight", "training_value": 0,
                            "evaluation_value": configs["long"].reward.energy_weight},
        "forced_sensing": ["front", "all_on"],
    }, indent=2))
    for name, cfg in configs.items():
        cfg.save(output / f"{name}_evaluation_config.json")

    ablation = copy.deepcopy(configs["long"])
    ablation.reward.energy_weight = 0
    training = {}
    matrix_keys = ["training_stage", "test_stage", "policy"]
    diagnostic_keys = ["condition", "policy"]
    context = mp.get_context("spawn")
    try:
        for name, config in (("overtaking_neural", configs["overtaking"]), ("no_energy_penalty", ablation)):
            training[name] = start_training(output / name, config, settings, seed)
            print(f"Started {name} training (PID {training[name].pid})", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {}
            results = defaultdict(list)

            def schedule(group, config, path, labels, forced_sensing=None):
                future = pool.submit(evaluate_policy, config, path, forced_sensing)
                futures[future] = group, labels

            for policy in NEURAL:
                for mode in (None, "front", "all_on"):
                    condition = "learned" if mode is None else f"forced_{mode}"
                    schedule("sensing_diagnostic", configs["long"], checkpoints["long"][policy],
                             {"condition": condition, "policy": policy}, mode)
            for train_stage, cohort in checkpoints.items():
                for test_stage, config in configs.items():
                    for policy, path in cohort.items():
                        schedule("stage_matrix", config, path, {
                            "training_stage": train_stage, "test_stage": test_stage, "policy": policy,
                        })

            def collect():
                for future in as_completed(futures):
                    group, labels = futures[future]
                    results[group].append(future.result().assign(**labels))
                    keys = matrix_keys if group == "stage_matrix" else diagnostic_keys
                    rows = save_evaluations(output, group, results[group], keys)
                    if group == "stage_matrix":
                        matched = rows[rows.training_stage == rows.test_stage]
                        if not matched.empty:
                            write_csv(summarize(matched, keys), output / "stage_comparison_summary.csv")
                    print(f"Evaluated {labels}", flush=True)
                futures.clear()

            collect()
            for name, process in training.items():
                if process.wait() != 0:
                    raise RuntimeError(f"{name} training failed; see {output / name / 'training.log'}")
                print(f"Completed {name} training", flush=True)
            for policy in NEURAL:
                for test_stage, config in configs.items():
                    schedule("stage_matrix", config, output / "overtaking_neural" / f"{policy}.pt", {
                        "training_stage": "overtaking", "test_stage": test_stage, "policy": policy,
                    })
                schedule("sensing_diagnostic", configs["long"],
                         output / "no_energy_penalty" / f"{policy}.pt",
                         {"condition": "trained_without_energy_penalty", "policy": policy})
            collect()
            matrix = pd.read_csv(output / "stage_matrix_trials.csv")
            matched = matrix[matrix.training_stage == matrix.test_stage]
            assert len(matched.groupby(matrix_keys)) == 9
    finally:
        for process in training.values():
            if process.poll() is None:
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.txt"
    with status.open("x") as file:
        file.write("running\n")
    try:
        run(output, args.evaluation_workers)
    except BaseException:
        status.write_text("failed\n")
        raise
    else:
        status.write_text("complete\n")
        print(f"Completed study: {output}", flush=True)


if __name__ == "__main__":
    main()
