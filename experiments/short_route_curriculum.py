"""Compare original short-route training with varied traffic and crash restarts.

On a crash, keep the collision penalty and terminal learning target. The next
episode starts in a newly sampled, feasible passing scenario. Successful trips
and timeouts lead back to the ordinary curriculum mix. The held-out routes keep
their original termination rules. No simulator truth enters the policies.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import asdict
import copy
import json
import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch

from algorithms.dqn_agent import DQNAgent, DQNSettings, DynaDQNAgent
from algorithms.sensor_agent import AgentSettings, LinearExpectedSarsa, ObservationEncoder
from environments.sensor_highway import ExperimentConfig, SensorHighwayEnv
from environments.sensor_highway.backend import HighwayAdapter
from experiments.agent_helpers import run_agent
from experiments.compare_route_stages import LONG_NEURAL, PASSING_SARSA, SHORT, summarize, write_csv


POLICIES = ("sarsa", "dqn", "dyna_dqn")
ARMS = ("original", "varied_restart")
WEATHERS = ("clear", "rain", "fog")
VALIDATION_SEEDS = tuple(range(300000, 300006))
ORIGINAL_PROBABILITY = 0.35


def lead(offset, gap, speed):
    return {"lane_offset": offset, "gap_m": gap, "speed_m_s": speed}


def passing_config(base, leaders, starting_lane=None):
    config = copy.deepcopy(base)
    config.highway.update(vehicles_count=0, slow_lead_scenario=leaders)
    if starting_lane is not None:
        config.highway["initial_lane_id"] = starting_lane
    config.validate()
    return config


def training_layouts(base):
    layouts = {}
    for gap in (42, 57, 72):
        for speed in (10.5, 13.0):
            leader = lead(0, gap, speed)
            assert gap + speed * base.highway["duration"] < base.reward.route_distance_m
            for lane in range(3):
                layouts[f"single_l{lane}_g{gap}_v{speed}"] = passing_config(base, [leader], lane)
            for blocker_offset in (1, 2):
                blocker = lead(blocker_offset, gap - 16, speed - 1)
                layouts[f"blocked_{blocker_offset}_g{gap}_v{speed}"] = passing_config(
                    base, [leader, blocker], starting_lane=1,
                )
    return layouts


def held_out_layouts(base):
    return {
        "wide_gaps": passing_config(base, [lead(0, 82, 13.5), lead(1, 250, 11), lead(2, 465, 12)], 0),
        "staggered_right": passing_config(base, [lead(0, 95, 11.5), lead(1, 405, 14), lead(2, 225, 12)], 1),
        "mixed_speeds": passing_config(base, [lead(0, 72, 12.5), lead(1, 190, 10.5), lead(2, 530, 13.5)], 2),
    }


def verify_passable(name, config):
    """Use simulator truth to rule out impossible layouts before training."""
    with HighwayAdapter(config.highway) as env:
        env.reset(seed=42)
        start = env.vehicle.position[0]
        commands = {name: index for index, name in env.action_type.actions.items()}
        max_steps = int(config.highway["duration"] * config.highway["policy_frequency"])
        for step in range(max_steps):
            ego = env.vehicle
            lane = ego.lane_index[2]
            nearby = [(v.lane_index[2], v.position[0] - ego.position[0]) for v in env.slow_leaders]
            action = "IDLE"
            if step > 0 and any(other == lane and 0 < gap <= 75 for other, gap in nearby):
                for candidate, direction in ((lane - 1, "LANE_LEFT"), (lane + 1, "LANE_RIGHT")):
                    if 0 <= candidate < 3 and all(
                        other != candidate or gap < -18 or gap > 50 for other, gap in nearby
                    ):
                        action = direction
                        break
            _, _, terminated, truncated, _ = env.step(commands[action])
            if ego.crashed:
                break
            if ego.position[0] - start >= config.reward.route_distance_m:
                passed = env.slow_lead_diagnostics()
                if passed["slow_leaders_passed"] == passed["slow_leaders_total"]:
                    return
            if terminated or truncated:
                break
    raise ValueError(f"Feasibility controller did not finish {name} without crashing and pass all leaders")


def select_layout(arm, layouts, rng, after_crash, previous):
    if arm == "original" or (not after_crash and rng.random() < ORIGINAL_PROBABILITY):
        return "original"
    if after_crash:
        layouts = tuple(layout for layout in layouts if layout != previous)
    return str(rng.choice(layouts))


def train(base, layouts, name, arm, seed, total_steps, dqn_settings, sarsa_settings):
    torch.set_num_threads(1)
    action_rng = np.random.default_rng(seed)
    layout_rng = np.random.default_rng(np.random.SeedSequence([seed, 7091]))
    encoder = ObservationEncoder(base)
    settings = AgentSettings(**sarsa_settings) if name == "sarsa" else DQNSettings(**dqn_settings)
    records, real_steps, after_crash, layout = [], 0, False, "original"
    with ExitStack() as stack:
        environments = {"original": stack.enter_context(SensorHighwayEnv(base))}
        obs, _ = environments["original"].reset(seed=seed)
        features_count = len(encoder(obs))
        if name == "sarsa":
            agent = LinearExpectedSarsa(features_count, feature_schema=encoder.feature_schema)
        else:
            cls = DynaDQNAgent if name == "dyna_dqn" else DQNAgent
            agent = cls(features_count, settings, feature_schema=encoder.feature_schema, seed=seed)
        while real_steps < total_steps:
            episode = len(records) + 1
            layout = select_layout(arm, tuple(layouts), layout_rng, after_crash, previous=layout)
            if layout not in environments:
                environments[layout] = stack.enter_context(SensorHighwayEnv(layouts[layout]))
            env = environments[layout]
            episode_seed = seed + episode - 1
            obs, _ = env.reset(seed=episode_seed)
            features = encoder(obs)
            steps, total_return, energy = 0, 0., 0.
            restarted_after_crash = after_crash
            while real_steps < total_steps:
                fraction = real_steps / max(1, total_steps - 1)
                epsilon = settings.epsilon_start + fraction * (settings.epsilon_end - settings.epsilon_start)
                action = agent.select(features, epsilon, action_rng)
                obs, reward, terminated, truncated, info = env.step(agent.decode(action, env))
                following = encoder(obs)
                terminal = terminated or truncated  # All training configs are finite routes.
                if name == "sarsa":
                    agent.update(features, action, reward, following, terminal,
                                 epsilon, settings.alpha, settings.gamma)
                else:
                    agent.observe(features, action, reward, following, terminal)
                features = following
                steps += 1
                real_steps += 1
                total_return += reward
                energy += info["energy_j"]
                if terminal:
                    break
            after_crash = info["crashed"]
            records.append({
                "episode": episode, "seed": episode_seed, "layout": layout,
                "restarted_after_crash": restarted_after_crash, "epsilon": epsilon,
                "steps": steps, "cumulative_steps": real_steps, "return": total_return,
                "crashed": after_crash, "route_completed": info["route_completed"],
                "budget_cutoff": not terminal, "elapsed_s": info["time_s"],
                "distance_m": info["distance_m"], "energy_j": energy,
                "slow_leaders_passed": info.get("slow_leaders_passed"),
            })
            if episode % 100 == 0 or real_steps == total_steps:
                print(f"{name} / {arm} / seed {seed}: {real_steps}/{total_steps} decisions, "
                      f"{episode} episodes, epsilon={epsilon:.3f}", flush=True)
    history = pd.DataFrame(records)
    assert int(history.steps.sum()) == total_steps
    if arm == "varied_restart":
        assert (history.loc[history.restarted_after_crash, "layout"] != "original").all()
    return agent, history


def evaluate(agent, seed, arm, policy, tests, seeds=VALIDATION_SEEDS, weathers=WEATHERS):
    rows = []
    for stage, layouts in tests.items():
        for weather_index, weather in enumerate(weathers):
            for seed_index, trial_seed in enumerate(seeds):
                layout, config = layouts[(seed_index + weather_index) % len(layouts)]
                config = copy.deepcopy(config)
                config.weather.preset = weather
                metrics, _ = run_agent(config, agent, seed=trial_seed)
                rows.append({"arm": arm, "policy": policy, "training_seed": seed,
                             "test_stage": stage, "layout": layout, "weather": weather,
                             "seed_panel": "held_out", **metrics})
    return pd.DataFrame(rows)


def run_job(output, name, arm, seed, steps, base, layouts, dqn_settings, sarsa_settings, tests):
    folder = output / arm / f"seed_{seed}" / name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            agent, history = train(base, layouts, name, arm, seed, steps, dqn_settings, sarsa_settings)
            agent.save(folder / ("policy.npz" if name == "sarsa" else "policy.pt"))
            if name == "dyna_dqn":
                torch.save(agent.model.state_dict(), folder / "prediction_model.pt")
            write_csv(history, folder / "training.csv")
            print("Training complete, starting held-out evaluation", flush=True)
            trials = evaluate(agent, seed, arm, name, tests)
            write_csv(trials, folder / "evaluation.csv")
            print("Evaluation complete", flush=True)
            return trials
        except Exception:
            traceback.print_exc()
            raise


def save_results(output, frames):
    keys = ["test_stage", "policy", "arm"]
    rows = pd.concat(frames, ignore_index=True).sort_values([*keys, "training_seed", "weather", "seed"])
    write_csv(rows, output / "evaluation.csv")
    per_seed = summarize(rows, [*keys, "training_seed"])
    per_seed = per_seed[per_seed.seed_panel == "all"].drop(columns="seed_panel")
    write_csv(per_seed, output / "summary_by_training_seed.csv")
    aggregate = summarize(rows, keys)
    aggregate = aggregate[aggregate.seed_panel == "all"].drop(columns="seed_panel")
    aggregate = aggregate.merge(rows.groupby(keys).training_seed.nunique().rename("training_seeds"), on=keys)
    successful = rows[rows.completed].groupby(keys).agg(
        successful_mean_power_w=("mean_power_w", "mean"), successful_mean_energy_j=("energy_j", "mean"),
    )
    write_csv(aggregate.merge(successful, on=keys, how="left"), output / "summary.csv")


def run(output, args):
    short = ExperimentConfig.load(SHORT / "config.json")
    long = ExperimentConfig.load(LONG_NEURAL / "config.json")
    reference = ExperimentConfig.load(PASSING_SARSA / "config.json")
    old_settings = json.loads((SHORT / "run_settings.json").read_text())
    dqn_settings, sarsa_settings = old_settings["dqn_settings"], old_settings["sarsa_settings"]
    layouts = training_layouts(short)
    unseen = held_out_layouts(long)
    for name, config in {**layouts, **unseen}.items():
        verify_passable(name, config)
    print(f"Verified {len(layouts)} training layouts and {len(unseen)} unseen validation layouts", flush=True)
    tests = {
        "short_original": [("original", short)],
        "long_original": [("original", long)],
        "long_overtaking_reference": [("previous_staggered", reference)],
        "long_overtaking_unseen": list(unseen.items()),
    }
    (output / "design.json").write_text(json.dumps({
        "training_seeds": args.training_seeds, "real_decisions_per_policy": args.steps,
        "epsilon_schedule": "Linear by real decision, shared within each algorithm and seed",
        "training": {"original_config": asdict(short), "dqn_settings": dqn_settings,
                     "sarsa_settings": sarsa_settings, "original_probability_after_non_crash": ORIGINAL_PROBABILITY,
                     "after_crash": "Terminal update, then reset to a sampled passing layout in varied_restart",
                     "passing_layouts": {name: asdict(cfg) for name, cfg in layouts.items()}},
        "evaluation": {"seeds": VALIDATION_SEEDS, "weather": WEATHERS,
                       "stages": {stage: {name: asdict(cfg) for name, cfg in group}
                                  for stage, group in tests.items()}},
    }, indent=2))
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        jobs = {
            pool.submit(run_job, output, policy, arm, seed, args.steps, short, layouts,
                        dqn_settings, sarsa_settings, tests): (policy, arm, seed)
            for seed in args.training_seeds for policy in POLICIES for arm in ARMS
        }
        for job in as_completed(jobs):
            label = jobs[job]
            try:
                results.append(job.result())
            except Exception as error:
                failures.append(label)
                print(f"Failed {label}: {error}", flush=True)
                continue
            save_results(output, results)
            print(f"Completed {label} ({len(results)}/{len(jobs)})", flush=True)
    if failures:
        raise RuntimeError(f"Failed jobs: {failures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=22000)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[42, 30042, 60042])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    ordered = sorted(args.training_seeds)
    if (args.steps <= 0 or args.workers <= 0 or ordered[0] < 0
            or any(b-a < args.steps for a, b in zip(ordered, ordered[1:]))
            or any(seed <= trial < seed+args.steps for seed in ordered for trial in VALIDATION_SEEDS)):
        parser.error("Use positive steps/workers and nonoverlapping training and validation seed ranges")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.txt"
    with status.open("x") as file:
        file.write("running\n")
    try:
        run(output, args)
    except BaseException:
        status.write_text("failed\n")
        raise
    status.write_text("complete\n")
    print(f"Study complete: {output}", flush=True)


if __name__ == "__main__":
    main()
