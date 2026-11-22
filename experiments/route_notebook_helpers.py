"""Shared output and evaluation for the route training notebooks."""

from dataclasses import asdict
from datetime import datetime
import json

from algorithms.sensor_agent import AgentSettings, SENSING_MODES
from experiments.agent_helpers import evaluate_agents, train_agent


def new_run_directory(repo, study):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = repo / "runs" / "local" / f"{study}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def train_sarsa_pair(config, protocol, output):
    """Train Expected SARSA with adaptive and always-on sensing."""
    settings = AgentSettings(**protocol["settings"])
    config.save(output / "config.json")
    metadata = dict(protocol, algorithm="linear Expected SARSA", settings=asdict(settings))
    agents = {}
    for policy, stem, modes in (
        ("adaptive", "adaptive", SENSING_MODES),
        ("learned_all_on", "always_on", ("all_on",)),
    ):
        agent, history = train_agent(config, settings, seed=protocol["training_seed"], modes=modes)
        agent.save(output / f"{stem}_agent.npz")
        history.to_csv(output / f"{stem}_training.csv", index=False)
        metadata["feature_schema"] = agent.feature_schema
        metadata[f"{stem}_actions"] = agent.actions
        agents[policy] = agent
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return agents


def summarize_trials(trials):
    metrics = {
        "episodes": ("seed", "size"),
        "completion_fraction": ("completed", "mean"),
        "crash_fraction": ("crashed", "mean"),
        "mean_completion_time_s": ("completion_time_s", "mean"),
        "mean_power_w": ("mean_power_w", "mean"),
    }
    if "slow_leaders_passed" in trials.columns:
        metrics["mean_leaders_passed"] = ("slow_leaders_passed", "mean")
    return trials.groupby("policy", sort=False).agg(**metrics)


def evaluate_route(config, agents, protocol):
    return evaluate_agents(config, agents, protocol["test_seeds"], protocol["test_weather"])
