# Interactive experiments

Open `sensor_environment_initial_test.ipynb` and select the project's `.venv`
kernel (Python MORL4AVSO). Run cells top to bottom.

The notebook shows maps and an animation, compares sensing configurations and
weather on matched traffic seeds, and switches sensors off to inspect stale
information. It uses fixed driving actions, not training. Start by changing the
settings in section 2. `notebook_helpers.py` contains the short rollout and plot
helpers so the notebook stays readable.

Implementation, assumptions, and literature: [environment README](../environments/README.md).
Optional saved runs go under `runs/`.

## Train an agent

Open `sensor_agent_experiment.ipynb` for the first learning experiment. It trains
an adaptive sensing/driving agent and a learned always-on driving agent, using
linear Expected SARSA implemented with NumPy. A menu of four sensing modes and
five driving actions produces 20 joint actions. There are no new dependencies.

The notebook includes training curves, held-out comparisons against IDLE and
random driving, sensing-mode usage, an animated learned rollout, action-value
inspection, and model saving. Defaults are 60 episodes per learned policy and
four held-out seeds in each of three weather conditions. This is an exploratory
run, not evidence of convergence or safety. Change settings in section 2 and
restart/run all for a fresh experiment.

Agent implementation: `algorithms/sensor_agent.py`; experiment loops:
`experiments/agent_helpers.py`. Feature construction uses only observations.
The linear model pools the grid into local regions and has no recurrent memory,
true lane index, or simulator-derived gaps. The sensing menu does not yet allow
independent per-camera decisions. Evaluation reports energy, power, crash rate,
completion, distance, reward, and the environment's confidence proxy together.

Saved timestamped runs contain both models and their matching configuration,
metadata, histories, and evaluation CSVs. Load using
`LinearExpectedSarsa.load(path)` with the matching encoder/configuration; model
files alone do not specify the meaning of their input features.

Agent checks:

```bash
.venv/bin/python -m unittest discover -s experiments/tests -v
```

## Time-to-completion experiment

Open [sensor_time_completion_experiment.ipynb](sensor_time_completion_experiment.ipynb)
to train two fresh Expected SARSA agents with a 300 m route and a 16 s deadline.
It uses the same 60 episodes per policy and matched held-out seeds/weather as the
first notebook. Route progress (+0.02/m), elapsed time (-0.1/s), a safe finish
(+10), and early completion (+5 times the fraction of time remaining) replace
the old speed-only incentive. The stock speed reward is disabled; collision
reward is -10, with another -10 for any unsuccessful ending. Sensor costs remain.

Both policies receive normalized remaining distance and time, assuming navigation
and clock telemetry. Successful completion, crash/off-road, and deadline failures
have terminal learning targets. The new model schema is `sensor_grid_regions_route_v2`;
use its saved configuration when loading. The original notebook and old models
retain `sensor_grid_regions_v1` and continuing-driving cutoff behavior.

Compare completion/crash fractions and successful completion times together.
Failures have missing completion times, not zero. Return cannot be compared
directly to the previous task. Discount is 0.99 in this notebook; reward, features,
and task have changed, so a single run cannot isolate a causal explanation.
Outputs go to fresh `runs/training/time_completion_<timestamp>/` directories and
include models, configuration, training histories, evaluation, and metadata.

## Longer route experiment: 120 seconds, 200 episodes

[sensor_time_completion_120s_200episodes.ipynb](sensor_time_completion_120s_200episodes.ipynb)
trains fresh adaptive and always-on agents for 200 episodes each. The deadline
is 120 seconds and the route is 2,250 m, scaling the previous 300 m / 16 s task
by 7.5. A finish or crash can end the episode before the deadline. Reward
coefficients, learning settings other than episode count, seed, and held-out
weather/traffic conditions are retained. Longer trips change the cumulative
reward/cost balance and discounting, so compare safety, completion, speed,
and power together rather than raw returns across task lengths.

The notebook checkpoints each trained model before evaluation, saves cell outputs
as it executes, and writes final artifacts under
`runs/training/time_completion_120s_200ep_<timestamp>/`.

## Sensor Realism and Calibration Study

Study ID: `sensor_realism_validation_v1`.
Open [sensor_realism_validation_study.ipynb](sensor_realism_validation_study.ipynb)
to record representative hardware choices, parameter evidence, source conditions,
uncertainty, acceptance criteria, research notes, and calibration decisions.
The notebook inventories current settings and runs three controlled diagnostics:
approaching-target detection, fog sensitivity, and shutdown/measurement aging.
It includes plots and implementation checks, with no policy training.

The initial run is an **uncalibrated baseline**. Hardware selection and empirical
validation are pending; synthetic checks do not establish real-world accuracy.
Each save creates `runs/sensor_studies/sensor_realism_study_<timestamp>/` with configuration,
evidence and hardware registers, scenario CSVs, findings, notes, and provenance.
Update evidence records as research progresses, then rerun and save a new revision.

## Sensor Contribution and Range Sensitivity Study

Follow-up study ID: `sensor_contribution_sensitivity_v1`.
[sensor_contribution_sensitivity_study.ipynb](sensor_contribution_sensitivity_study.ipynb)
compares bearing/range/closing-speed measurement availability, sweeps unvalidated
optical range cutoffs, and removes sensor modalities from a frozen observation-driven
driver. It uses the saved short-route always-on model, with six matched validation
trials per sensing condition and no new training. `BASE_RUN` identifies that model
and its configuration. Helpers live in `sensor_study_helpers.py`.

This differs from fixed-action rollouts: the driver can respond to observations,
but its weights stay fixed. Ablations introduce distribution shifts, so outcomes
measure this driver's reliance on sensors, not the best performance possible with
each modality. Driving-sequence comparisons help identify a sensor-insensitive
policy. Uncertainty/support remain model proxies. External datasets/models are
listed as references; no empirical calibration has been performed.

Exports under `runs/sensor_studies/sensor_contribution_study_<timestamp>/` contain measurement
and ablation CSVs, driving histories, range-profile configurations, and provenance.

## RGB versus Event Camera: Controlled Cut-in Study

Study ID: `camera_cutin_timing_v1`.
[camera_cutin_timing_study.ipynb](camera_cutin_timing_study.ipynb) scripts a
right-to-ego-lane cut-in at 20/40 m forward gaps, with 1/2/4 m/s lateral motion
and a stationary relative-motion control. Thirty noise seeds per condition
compare project defaults, matched camera coverage, and equalized timing: 1,440
sensor trials in total. `cutin_study_helpers.py` uses the existing SensorSuite
capture/delivery pipeline, with false positives disabled and no sensor-model or
policy changes. The cameras start before the maneuver so warm-up is not confused
with motion latency.

A causal bearing-history fit estimates angular motion, not metric lateral speed.
The study records first detection/cue delay, information age, conditional angular
rate error, availability, false motion alerts, and what a 2 Hz consumer sees.
Matched coverage tests the proposed temporal advantage without hardcoding an RGB
range advantage. A single-target tracker avoids association ambiguity and is
separate from production grid fusion. All parameters remain uncalibrated; these
are model diagnostics, not a real hardware benchmark or a collision-safety test.
Results, profile settings, traces, and provenance are saved under
`runs/sensor_studies/camera_cutin_study_<timestamp>/`.

## DQN and Dyna-DQN comparison

Open [sensor_dqn_experiment.ipynb](sensor_dqn_experiment.ipynb) for a simple
Markdown walkthrough of each part: neural Q values, replay, target network,
learning targets, a learned one-step model, and planning. It trains fresh DQN,
Dyna-DQN, and Expected SARSA policies on the same 300 m / 16 s task, then reuses
the held-out evaluator and saves models, configuration, and measurements.

Install the optional PyTorch dependency first with
`.venv/bin/python -m pip install -r requirements-dqn.txt`. Code is in
`algorithms/dqn_agent.py` and `dqn_helpers.py`; a code-by-code explanation is in
[the algorithm README](../algorithms/README.md). Dyna uses real replay to train
a feature/reward/terminal model and makes extra Q updates from one-step
predictions. Extra replay alone is not called Dyna. Saved `.pt` files restore
evaluation policies, not full training state. Short runs do not establish
convergence or improvement over SARSA.

## Longer DQN training

[sensor_dqn_long_training.ipynb](sensor_dqn_long_training.ipynb) repeats the
short-route comparison with 1,000 episodes per agent (instead of 60), keeping
network, reward, learning rate, and planning settings fixed. Exploration decays
across the longer training schedule. It evaluates 20 held-out traffic seeds
under clear, rain, and fog weather, and separately compares the original 12
evaluation scenarios with the saved short run. This remains a single training
seed; it tests training budget and exploration duration together.

In sequential notebook runs, each policy and history is saved before the next
agent trains. Concurrent workers each save their own completed results. The
Dyna prediction network is saved separately for later inspection. Outputs go
under `runs/comparisons/dqn_long_training_<timestamp>/`; the executed notebook retains
training output and plots. Policy saves do not resume exact training state.

For concurrent CPU training, prepare a run folder containing `config.json` and
`run_settings.json` using the notebook setup cells, then run:

```bash
.venv/bin/python -m experiments.run_long_dqn --output runs/comparisons/<run_folder>
FINAL_PROJECT_DQN_RUN_DIR=runs/comparisons/<run_folder> .venv/bin/python -m jupyterlab
```

Open the long-training notebook in that Jupyter session and run all cells to
load the completed policies and evaluate them. Matching training settings and
environment configuration are checked before loading. Every evaluation also
records driving-action fractions, helping identify policies that repeat one
action regardless of observations.

## Inspect saved-policy replays

Generate matched replays without retraining:

```bash
.venv/bin/python -m experiments.replay_agents --seed 100000 --weather clear
```

Open the printed `replay.html` path in a browser for synchronized playback,
frame stepping, speed control, road views, sensor-evidence maps, and telemetry.
Each agent also gets a GIF preview, per-decision CSV, and metrics JSON.
The default source is the completed 1,000-episode run; use `--run` to choose
another compatible saved run. Captures are at 2 Hz decision boundaries.
Use `--seed 100011 --weather clear` to inspect a matched case where SARSA
crashed in the recorded evaluation. These are illustrative scenarios.

## Frozen models on a 120-second route

Evaluate the existing 1,000-episode policies without retraining:

```bash
.venv/bin/python -m experiments.evaluate_saved_agents --duration 120 --route-distance 2250
.venv/bin/python -m experiments.replay_agents --duration 120 --route-distance 2250 --seed 100000 --weather clear
```

The 2,250 m route scales the 300 m / 16 s task by 7.5, matching the earlier
120-second experiment. Crash, off-road, or completion can end an episode early.
The evaluator uses the same 20 traffic seeds across clear, rain, and fog, and
saves per-policy CSVs incrementally, aggregate results, configuration, model
hashes, and a short report. Model hashes are checked after evaluation.

This is a transfer test: remaining-route/time features use the new normalization,
so actions can differ even early in a trip. Reward coefficients are retained,
but cumulative costs and progress reward grow with the route. Raw returns and
episode crash rates should not be treated as equal-task/equal-exposure
comparisons with the 16-second run.
