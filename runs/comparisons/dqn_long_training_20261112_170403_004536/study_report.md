# Longer DQN training results

All three agents trained from scratch for 1,000 episodes using seed 42, up from
60 episodes. The task remains 300 m within 16 s. Reward, network, optimizer,
replay, and Dyna planning settings are unchanged. Epsilon decays from 1.0 to 0.1
across the longer run, so training duration and exploration duration change together.

Training used separate CPU processes, each with one PyTorch thread. This run
used the project .venv; the original saved run records a different Python version.
Configuration, package versions, histories, policies, and the executed notebook
are retained for inspection. This comparison is exploratory rather than an
isolated causal estimate of training-budget effects.

## Evaluation on 20 traffic seeds in three weather conditions

There are 60 evaluations per policy. The same traffic seeds are reused across
weather and policies; these are not 60 independent training runs.

| Policy | Trials | Completion | Crashes | Mean return | Mean sensing power (W) |
| --- | --- | --- | --- | --- | --- |
| Expected SARSA | 60 | 85.0% | 15.0% | 8.73 | 12.1 |
| DQN | 60 | 98.3% | 1.7% | 11.77 | 71.4 |
| Dyna-DQN | 60 | 96.7% | 3.3% | 11.37 | 75.4 |
| idle_all_on | 60 | 40.0% | 60.0% | -5.83 | 123.0 |
| random_all_on | 60 | 20.0% | 80.0% | -12.67 | 123.0 |

## Direct comparison on the original 12 scenarios

These use the same four traffic seeds and three weather conditions as the short run.

| Policy | 60-episode completion | 1,000-episode completion | 60-episode crashes | 1,000-episode crashes |
| --- | --- | --- | --- | --- |
| Expected SARSA | 91.7% | 100.0% | 8.3% | 0.0% |
| DQN | 25.0% | 100.0% | 75.0% | 0.0% |
| Dyna-DQN | 50.0% | 91.7% | 50.0% | 8.3% |

## Training experience

Equal episodes do not imply equal real steps because crashes and finishes end trips.
Last-200-episode training completion includes exploration and is not held-out performance.

| Policy | Episodes | Real steps | Q updates | Last 200 training completion |
| --- | --- | --- | --- | --- |
| Expected SARSA | 1000 | 19580 | one per real step | 66.5% |
| DQN | 1000 | 19961 | 19898 | 69.0% |
| Dyna-DQN | 1000 | 21630 | 64429 | 83.5% |

## Learned behavior during evaluation

Action and sensing fractions are averaged per episode. A high frequency of one
command does not by itself establish that the agent ignores observations.

| Policy | Sensors off | Most frequent command | Command fraction | Mean speed (m/s) |
| --- | --- | --- | --- | --- |
| Expected SARSA | 80.3% | SLOWER | 68.8% | 21.7 |
| DQN | 6.1% | SLOWER | 88.5% | 20.2 |
| Dyna-DQN | 4.5% | SLOWER | 54.8% | 20.7 |

## What this run suggests

The neural agents improved substantially on the original evaluation scenarios:
DQN completed 12/12 after longer training, up from 3/12, and Dyna-DQN completed
11/12, up from 6/12. On the expanded set, DQN completed 59/60 and Dyna-DQN 58/60,
compared with 51/60 for SARSA.

Both neural policies now spend most decisions with sensors enabled and drive
more slowly than in the short run. Dyna-DQN no longer repeats the same left/off
action throughout evaluation. These observations support insufficient training
as a substantial contributor to the earlier poor results. Training and the
exploration schedule changed together, so this run does not isolate their effects.
The results also do not establish that sensing caused the improvement: slower
driving is another plausible contributor.

SARSA uses much less sensing power but has more crashes on the expanded set.
DQN slightly outperformed Dyna-DQN here while making far fewer Q updates. This
single training seed does not establish an advantage for model-based planning.

## Files

- `summary.csv`: expanded held-out evaluation.
- `matched_budget_comparison.csv`: old/new results on matched scenarios.
- `evaluation.csv`: each evaluation, including sensing and driving frequencies.
- `*_training.csv`: full training histories.
- `dqn.pt`, `dyna_dqn.pt`, `sarsa.npz`: evaluation policies.
- `dyna_prediction_model.pt`: separately saved learned dynamics weights.
- `config.json`, `run_settings.json`, `metadata.json`: reproducibility settings.
- `evaluation_summary.png`: completion, crashes, and power comparison.

The executed notebook is `experiments/sensor_dqn_long_training.ipynb`.
Only one training seed was used. These results assess this run; repeat training
seeds before claiming that one algorithm reliably performs better.
