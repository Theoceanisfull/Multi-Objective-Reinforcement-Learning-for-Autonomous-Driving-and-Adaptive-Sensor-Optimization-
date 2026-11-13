# Frozen models: 120-second evaluation

The same saved policies trained for 1,000 episodes on a 300 m /
16 s task were evaluated without learning. The new task is
**2250 m within 120 s**. A crash, off-road ending, or safe
finish can end a run earlier. Reward coefficients, sensor models, and traffic settings
are unchanged; remaining route/time features are normalized using the new task.

This tests transfer to a longer route, not newly trained 120-second agents.
Source model hashes were verified unchanged. Dyna's prediction model is not used
to choose actions during evaluation.

## Results

| Policy | Finishes | Crashes | Successful finish (s) | Sensing power (W) | Sensing energy (J) |
| --- | --- | --- | --- | --- | --- |
| Expected SARSA | 11 | 49 | 108.09 | 18.16 | 248.31 |
| DQN | 57 | 3 | 112.59 | 71.96 | 7725.05 |
| Dyna-DQN | 28 | 32 | 111.89 | 74.87 | 5144.73 |

Finish time averages successful episodes only; missing values mean no successes.
Power is mean per-episode sensing power. Energy includes failed episodes and can be
low because of early crashes. Neither measure includes training electricity.

Each policy used 20 held-out traffic seeds across clear, rain, fog:
60 trials per policy. Seeds match the previous comparison.
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

## Comparison with the short task

| Frozen policy | Finishes: 300 m / 16 s | Finishes: 2,250 m / 120 s |
| --- | --- | --- |
| Expected SARSA | 51/60 | 11/60 |
| DQN | 59/60 | 57/60 |
| Dyna-DQN | 58/60 | 28/60 |

DQN retained high completion in this transfer test. SARSA and Dyna-DQN lost
substantial completion reliability. These results describe the existing short-task
policies; they do not establish how the algorithms would perform after training
for the longer task. Different normalized remaining-route/time inputs can change
actions early, before the original deadline, as well as longer trips adding exposure.

## Inspect matched replays

[Open the saved replay viewer](../replays_120s_2250m_clear_seed100000_20261112_193307_152187/replay.html)
in a browser. GIFs, action CSVs, and detailed metrics are beside it. The matched
clear-weather scenario uses seed 100000: SARSA crashed at 4.5 s, Dyna-DQN crashed
at 36.0 s, and DQN completed at 112.5 s. Replays end on crash or completion,
rather than being forced to continue for 120 seconds.
