# Experiments

Start Jupyter with `python -m jupyterlab` from the environment in the [quick start](../README.md#quick-start). Open a notebook and use **Run All**.

| Notebook | What it shows |
| --- | --- |
| [Inspect highway sensing](sensor_environment_initial_test.ipynb) | Occupancy evidence and power for the same fixed drive in clear, rain, and fog |
| [Short-route comparison](sensor_dqn_long_training.ipynb) | Expected SARSA, DQN, and Dyna-DQN on the 300 m / 16 s route |
| [Long-route Expected SARSA](sensor_time_completion_120s_1000episodes.ipynb) | Adaptive sensing and all-on baselines on the 2,250 m / 120 s route |
| [Long-route DQN comparison](sensor_dqn_time_completion_120s_2250m_1000episodes.ipynb) | DQN, Dyna-DQN, and frozen adaptive Expected SARSA on the same route |
| [Controlled overtaking](sensor_overtaking_120s_2250m_1000episodes.ipynb) | Long-route Expected SARSA trained with slow leaders and evaluated in both traffic conditions |
| [Recorded results](analyze_route_results.ipynb) | Training curves, baseline comparison, and transfer across three training seeds |

The four training notebooks display bundled trial results by default. Set `RUN_TRAINING = True` to train using the saved configuration and evaluation seeds. A fresh run writes weights and results to a new directory under `runs/local/`. The DQN notebooks accept `FINAL_PROJECT_DQN_RUN_DIR` for a compatible saved run.

Run the final studies from the repository root with fresh output directories:

```bash
python -m experiments.compare_route_stages --output runs/local/three_stages
python -m experiments.short_route_curriculum --output runs/local/short_route_curriculum --workers 3
```

Evaluate or replay the bundled short-route policies on the original long route:

```bash
python -m experiments.evaluate_saved_agents --duration 120 --route-distance 2250
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m experiments.replay_agents --duration 120 --route-distance 2250 --seed 100000 --weather clear
```
