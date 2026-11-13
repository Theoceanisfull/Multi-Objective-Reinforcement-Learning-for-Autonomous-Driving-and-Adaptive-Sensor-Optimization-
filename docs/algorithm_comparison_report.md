# Algorithm Comparison: Safety, Sensing Energy, and Completion Time

**Experiment:** 300 m highway route, 16 s deadline; 1,000 training episodes per learned agent.

**Main finding:** DQN and Dyna-DQN completed more trips, while Expected SARSA used substantially less sensing energy and finished its successful trips slightly faster. The results show a safety–energy tradeoff rather than one method winning every measure.

## 1. Fair comparison and shared reward

All agents shared observation features, 20 driving/sensing choices, and environment rewards. SARSA uses a linear value model; DQN adds a neural network, replay, and target network; Dyna-DQN adds a learned prediction model and two planning updates per real step after warm-up.

Training used seed 42, discount 0.99, and exploration decreasing from 1.0 to 0.1. Evaluation used **60 trials per policy**: 20 held-out traffic seeds across clear, rain, and fog, without learning or exploration. Both baselines kept all sensors on; IDLE maintains driving targets rather than stopping.

| Shared reward component | Value |
| --- | --- |
| Forward progress | +0.02 per meter |
| Elapsed time | −0.1 per second |
| Safe completion | +10 |
| Early completion | +5 × fraction of deadline remaining |
| Failed ending | −10 |
| Collision | Additional −10 |
| Sensor energy | −0.001 per joule |
| Map uncertainty | −0.1 × uncertainty × elapsed seconds |

The stock speed bonus was disabled. Dyna planning uses **predicted rewards**, which can differ from real outcomes.

## 2. Results after longer training

| Policy | Finishes | Crashes | Finish time (s)¹ | Sensing power (W)² | Sensing energy (J)¹ | Mean return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Expected SARSA | 51/60 (85.0%) | 9/60 | **14.31** | **12.1** | **128.7** | 8.73 |
| DQN | **59/60 (98.3%)** | **1/60** | 15.00 | 71.4 | 1,070.6 | **11.77** |
| Dyna-DQN | 58/60 (96.7%) | 2/60 | 14.76 | 75.4 | 1,108.0 | 11.37 |
| IDLE, all sensors on | 24/60 (40.0%) | 36/60 | 12.00 | 123.0 | 1,476.0 | −5.83 |
| Random, all sensors on | 12/60 (20.0%) | 48/60 | 13.00 | 123.0 | 1,599.0 | −12.67 |

¹ Means among successful trips only; failures are excluded, so subsets differ. Read time and energy alongside completion rate.

² Mean per-episode sensing power across all trials. These energy measures exclude computation and training electricity. Early crashes can reduce total energy, which is why successful-trip energy is reported separately.

## 3. What longer training changed

Comparing the **same 12 scenarios** used in the original short run:

| Agent | Finishes: 60 → 1,000 episodes | Real steps in longer training | Q updates in longer training |
| --- | ---: | ---: | ---: |
| Expected SARSA | 11/12 → 12/12 | 19,580 | 19,580 |
| DQN | 3/12 → 12/12 | 19,961 | 19,898 |
| Dyna-DQN | 6/12 → 11/12 | 21,630 | 64,429 |

The improvement supports insufficient training as a contributor to earlier failures. Exploration duration also increased, so its effect was not isolated. Equal episode counts did not mean equal interaction or computation budgets.

## 4. Analysis

**SARSA saved substantially more sensing energy.** Sensors were off for 80.3% of its decisions, versus 6.1% for DQN and 4.5% for Dyna-DQN. Relative to DQN, SARSA used **83% less sensing power** and **88% less energy among successful trips**, but crashed more often.

**Identical rewards allow different strategies.** A collision incurs 20 units in collision/failure penalties; 1,000 J of sensing costs one unit. Avoiding failures can outweigh considerable sensing expenditure. Higher return therefore does not establish better energy management.

**Completion time favors SARSA among successful trips.** It finished about 0.69 s faster than DQN and 0.45 s faster than Dyna-DQN. However, SARSA failed in more scenarios. The neural agents drove more slowly overall, so increased sensing and conservative driving are both plausible contributors to their higher completion rates. Their individual effects remain untested.

**Dyna planning showed no clear advantage.** Dyna-DQN finished successful trips about 0.24 s faster than DQN, but used more sensing power and approximately 3.2 times as many Q updates. One additional crash cannot establish a reliable ranking from one training seed.

## 5. Interpretation and next experiment

**DQN achieved the highest completion rate; SARSA used the least sensing energy.** Neither won every objective.

Next, repeat training seeds and vary the energy penalty, comparing energy at similar completion rates. Separately disable planning to test Dyna's contribution.

Limitations: one training seed, simplified sensing, reused traffic seeds across weather, different Python versions between short and long runs, and success-conditioned timing. These are exploratory results.

**Evidence:** [full evaluation](../runs/comparisons/dqn_long_training_20261112_170403_004536/evaluation.csv), [matched short/long comparison](../runs/comparisons/dqn_long_training_20261112_170403_004536/matched_budget_comparison.csv), [saved configuration](../runs/comparisons/dqn_long_training_20261112_170403_004536/config.json), and [executed notebook](../experiments/sensor_dqn_long_training.ipynb).

**Inspect a matched run:** [synchronized replays and individual GIFs](../runs/comparisons/dqn_long_training_20261112_170403_004536/replays_clear_seed100000_20261112_191411_099079/replay.html) (seed 100000, clear weather; open in a browser).

## Follow-up: frozen policies on a longer route

Using the same weights on **2,250 m within 120 s**, successful finishes were
**11/60 for SARSA, 57/60 for DQN, and 28/60 for Dyna-DQN**. This tests transfer
of short-task policies, not retraining for the new horizon. Both exposure and
normalized route/time inputs changed.

See the [120-second report](../runs/comparisons/dqn_long_training_20261112_170403_004536/evaluation_120s_2250m_20261112_193414_051178/study_report.md)
and [matched replays](../runs/comparisons/dqn_long_training_20261112_170403_004536/replays_120s_2250m_clear_seed100000_20261112_193307_152187/replay.html).
