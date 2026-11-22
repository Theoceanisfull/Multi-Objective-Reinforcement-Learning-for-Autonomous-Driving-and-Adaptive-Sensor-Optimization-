# Experiments

Run commands from the repository root using the virtual environment in the [quick start](../README.md#quick-start). Recorded data lives in [`runs/`](../runs/). Save fresh experiment outputs under `runs/local/`, which Git ignores.

## Review recorded results

- [Analyze recorded route experiments](analyze_route_results.ipynb) plots training progress, compares against always-on sensing, and evaluates long-route transfer using bundled CSVs.
- [Environment inspection](sensor_environment_initial_test.ipynb) shows the observation map, delayed and missing detections, and alternative sensing modes.
- [Sensor realism](sensor_realism_validation_study.ipynb), [modality contribution](sensor_contribution_sensitivity_study.ipynb), and [camera cut-in timing](camera_cutin_timing_study.ipynb) probe individual sensing assumptions.

## Train joint driving and sensing policies

These notebooks start fresh training when run from the top, unless a supported saved-run directory is supplied.

| Experiment | Notebook |
| --- | --- |
| Initial adaptive versus always-on comparison | [Sensor agent](sensor_agent_experiment.ipynb) |
| Short fixed route | [300 m Expected SARSA](sensor_time_completion_experiment.ipynb) |
| Initial three-agent comparison | [Expected SARSA, DQN, and Dyna-DQN](sensor_dqn_experiment.ipynb) |
| 1,000 short-route episodes per agent | [Short-route comparison](sensor_dqn_long_training.ipynb) |
| More Dyna planning updates per real step | [Planning sensitivity](sensor_dqn_planning_ablation.ipynb) |
| 200 or 1,000 long-route episodes of Expected SARSA | [200 episodes](sensor_time_completion_120s_200episodes.ipynb) · [1,000 episodes](sensor_time_completion_120s_1000episodes.ipynb) |
| 1,000 long-route episodes of DQN and Dyna-DQN | [Long-route neural training](sensor_dqn_time_completion_120s_2250m_1000episodes.ipynb) |
| 1,000 overtaking episodes of adaptive and always-on Expected SARSA | [Controlled overtaking](sensor_overtaking_120s_2250m_1000episodes.ipynb) |

The short- and long-route DQN notebooks accept `FINAL_PROJECT_DQN_RUN_DIR` to load trained policies. Set it to an absolute path before starting Jupyter. To review the recorded short-route comparison:

```bash
FINAL_PROJECT_DQN_RUN_DIR="$PWD/runs/comparisons/dqn_long_training_20261112_170403_004536" python -m jupyterlab
```

Open [short-route comparison](sensor_dqn_long_training.ipynb). It loads the saved models and runs evaluation. To evaluate without modifying an existing saved run or launching training, use the command-line evaluator below.

## Run the core studies from the command line

```bash
# Re-evaluate the saved short-trained agents on the longer route.
python -m experiments.evaluate_saved_agents --duration 120 --route-distance 2250

# Generate synchronized replays of frozen policies.
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy   python -m experiments.replay_agents --duration 120 --route-distance 2250 --seed 100000 --weather clear

# Three-stage study, including additional neural training and sensing interventions.
python -m experiments.compare_route_stages --output runs/local/three_stages

# Matched training budgets across original traffic and the crash-restart curriculum.
python -m experiments.short_route_curriculum --output runs/local/short_route_curriculum --workers 3
```

The evaluator and replay generator default to the bundled short-route policies and write to timestamped subfolders in `runs/local/`. Open the generated `replay.html` locally in a browser. The last two commands launch training and need fresh output directories.

## Exploratory studies

These experiments use different simulator settings or policy inputs. Compare them within their own protocols.

| Study | Entry point |
| --- | --- |
| Spiking Expected SARSA | [Base spiking notebook](sensor_snn_expected_sarsa_120s.ipynb) · [Adaptive thresholds and STDP notebook](sensor_adaptive_snn_120s.ipynb) |
| Driving with all sensors on | [Driving-only training notebook](driving_policy_first.ipynb) · [Frozen-policy analysis](driving_policy_generalization.ipynb) |
| Warm-start driving curriculum | [2,000 m driving curriculum notebook](driving_dqn_dyna_2000m_curriculum.ipynb) · [Runner](train_driving_curriculum.py) |
| Persistent traffic on a 9 km route | [500-episode protocol](sensor_dqn_long_training_9000m.ipynb) · [5,000-episode protocol](sensor_dqn_long_training_9000m_5000ep.ipynb) |

The two SNN notebooks open recorded results by default. For a fresh comparison, turn on `RUN_TRAINING` in the base notebook first, then in the adaptive notebook. The recorded 9 km and driving curriculum files are historical outputs. Fresh training creates local protocols and checkpoints. Only load training-resume files from your own runs.

[Algorithms](../algorithms/README.md) · [Environment](../environments/README.md) · [Saved runs](../runs/)
