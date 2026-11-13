> Archived project background. Use the [repository README](../README.md) for the organized layout and current commands.

# Adaptive Sensor Activation for Highway Driving

A reinforcement-learning final project investigating whether an agent can reduce
sensor energy consumption while maintaining safe driving under different weather
and lighting conditions.

The project extends **HighwayEnv** through a modular **Gymnasium** environment.
The agent chooses driving actions and sensor activation using imperfect, delayed
observations rather than direct access to simulator traffic truth.

## Start here

Run these notebooks in order, using the project's `.venv` kernel:

**HighwayEnv and Q-Learning tutorial was for self-learning to understand the baseline environment**

| Notebook | Purpose |
| --- | --- |
| [HighwayEnv and Q-learning tutorial](../basic/highway_q_learning_tutorial.ipynb) | Learn the driving environment, discrete states, tabular Q-learning, and evaluation |
| [Initial sensor environment test](../experiments/sensor_environment_initial_test.ipynb) | Inspect sensor maps and animations, compare weather and sensing modes, and observe measurement aging |
| [Agent experiment](../experiments/sensor_agent_experiment.ipynb) | Train joint driving/sensing and always-on agents, then compare held-out performance |
| [DQN and Dyna-DQN comparison](../experiments/sensor_dqn_experiment.ipynb) | Build on Expected SARSA with a small neural agent and learned-model planning |
| [Longer DQN training](../experiments/sensor_dqn_long_training.ipynb) | Train all three agents for 1,000 episodes and evaluate on 60 held-out scenarios |
| [Time-to-completion experiment](../experiments/sensor_time_completion_experiment.ipynb) | Train fresh Expected SARSA agents to finish a 300 m route safely and quickly |
| [Longer route experiment](../experiments/sensor_time_completion_120s_200episodes.ipynb) | Train for 200 episodes per agent with a 2,250 m route and 120 s deadline |
| [Sensor Realism and Calibration Study](../experiments/sensor_realism_validation_study.ipynb) | Record hardware evidence and check approaching targets, fog, and shutdown behavior |
| [Sensor Contribution and Range Sensitivity](../experiments/sensor_contribution_sensitivity_study.ipynb) | Compare measurement availability, remove sensor modalities, and vary optical range assumptions |
| [Camera cut-in timing study](../experiments/camera_cutin_timing_study.ipynb) | Compare RGB/event detection timing and angular-motion cues during controlled cut-ins |

The sensor test uses fixed driving actions. The agent experiment uses linear
Expected SARSA implemented with NumPy; no neural-network framework is required.
The optional [DQN notebook](../experiments/sensor_dqn_experiment.ipynb) adds PyTorch
(`.venv/bin/python -m pip install -r requirements-dqn.txt`) and explains both
DQN and Dyna-DQN in plain language. See the [algorithm walkthrough](../algorithms/README.md).
Run notebook cells from top to bottom. After changing settings, restart the
kernel and run all cells to keep results consistent.

## Setup

Commands below assume a terminal in the `final_project` directory. The existing
project environment uses Python 3.14.

If `.venv` is already available:

```bash
source .venv/bin/activate
python -m jupyterlab
```

To create the environment from scratch with Python 3.14 installed:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name morl4avso --display-name "Python (MORL4AVSO)"
python -m jupyterlab
```

In VS Code, select `.venv/bin/python` or **Python (MORL4AVSO)** as
the notebook kernel. Install dependencies through the root [requirements.txt](../requirements.txt), which
references the shared pins in [basic/requirements.txt](../basic/requirements.txt). The environment adapter was
implemented against HighwayEnv 1.12.1 and Gymnasium 1.4.0.

## Environment

The default simulation contains three lanes, an ego vehicle, and twelve traffic
vehicles. Sensing consists of eight RGB cameras and eight event cameras facing
eight vehicle-relative directions, one forward radar, and one 360-degree LiDAR.
Sensor fields of view can overlap.

| Component | Implementation |
| --- | --- |
| Weather/lighting | Clear, night, rain, fog, and directional glare presets |
| Sensor effects | Limited range/FoV, noise, missed and false detections, simplified occlusion, warm-up, and latency |
| Observation | Ego-centered 150 m radius map with 2 m cells, sensor status, and ego telemetry |
| Map layers | Occupancy evidence, support, age, radial velocity, and velocity support |
| Driving actions | Lane left, maintain targets, lane right, faster, slower |
| Sensor actions | Independent activation mask for 18 sensors; the initial agent uses four predefined modes |
| Timing | Physics at 100 Hz and agent decisions at 2 Hz |
| Episode ending | Crash/off-road termination or duration cutoff |

RGB and event-camera models provide bearing evidence without direct metric
depth. Radar provides range, bearing, and radial relative velocity; LiDAR provides
range and bearing. New sensor measurements arrive after their modeled delays.
Turning a sensor off stops new measurements while existing information ages out.

The default reward is:

```text
reward = HighwayEnv driving reward
       - 0.001 * sensing energy in joules
       - 0.1 * corridor uncertainty * decision duration in seconds
```

Driving reward includes a speed incentive up to +0.4 per decision and a -5
collision penalty. There is no explicit distance, safe-time, or completion bonus.
Uncertainty is a heuristic map-entropy measure, not validated perception error.

The new time-to-completion notebook enables an optional fixed-distance task:
300 m within 16 s. It replaces the stock speed bonus with +0.02 per meter,
charges 0.1 per second, and pays +10 for a safe finish plus up to +5 for time
remaining. Crash/off-road/deadline failure costs 10, in addition to the notebook's
stock collision penalty of 10. Energy and uncertainty costs still apply. Remaining
route and time are observation features; all finite-task endings stop TD
bootstrapping. Original defaults and the first experiment retain their old task.

## Run a short demo

```bash
.venv/bin/python -m environments.demo --weather fog --sensing front
```

The demo maintains the initial lane/speed targets and saves a map preview,
observation arrays, configuration, and trajectory CSV under
`runs/environment/demo/`. Reusing an output directory replaces its files.
Use a separate directory to preserve each run:

```bash
.venv/bin/python -m environments.demo --weather clear --sensing all_on --output runs/environment/clear_all_on
```

Available demo sensing modes are `all_on`, `radar_rgb`, `front`, and `off`.

## Configuration and implementation

Edit [environments/configs/default.json](../environments/configs/default.json), or
construct an `ExperimentConfig` in Python. Sensor mounting, range, frequency,
noise, detection reliability, latency, power, weather effects, grid settings, and
reward weights are configurable. Recreate the environment after changing them.

```python
from environments.sensor_highway import ExperimentConfig, SensorHighwayEnv

config = ExperimentConfig()
config.weather.preset = "rain"
config.weather.severity = 0.7
config.reward.uncertainty_weight = 0.0

with SensorHighwayEnv(config) as env:
    observation, info = env.reset(seed=42)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
```

The action contains `drive` and `sensors` fields. Stop on either ending flag and
reset before stepping again. Policy input should come from `observation`;
`info` contains diagnostic information intended for evaluation.

See the [environment README](../environments/README.md) for the observation contract,
model assumptions, literature, and module responsibilities. The
[experiments README](../experiments/README.md) explains the notebooks and saved runs.

## Project structure

```text
final_project/
├── algorithms/                 # Expected SARSA, DQN, Dyna-DQN, and shared encoder
├── basic/                      # Original HighwayEnv/Q-learning tutorial and dependencies
├── environments/
│   ├── sensor_highway/         # Simulator adapter, sensors, weather, grid, and Gymnasium interface
│   ├── configs/                # Editable experiment configuration
│   ├── tests/                  # Environment behavior checks
│   └── demo.py                 # Fixed-driving sensing demonstration
├── experiments/                # Inspection and learning notebooks, helpers, and agent tests
├── docs/                  # Project proposal and milestone docs
├── helpers/                    # Reserved for additional shared utilities
└── logs/                       # Reserved for additional experiment logs
```

Historical environment and experiment artifacts are tracked under runs/ in this organized repository. The agent
notebook saves timestamped model files, matching configuration and metadata,
training histories, and evaluation results. Keep these together: weights alone
do not define the feature representation or environment settings.

## Tests

Run from the project root:

```bash
.venv/bin/python -m unittest discover -s environments/tests -v
.venv/bin/python -m unittest discover -s experiments/tests -v
```

The suites cover the Gymnasium interface, masking,
timing, energy, occlusion, coordinate orientation, reproducibility, stock-motion
agreement, learning targets, model saving/loading, route rewards, and finite-task learning targets.

## Current scope and next steps

This is an object-level sensor/perception abstraction. It does not generate
physical RGB images, event streams, radar waveforms, or LiDAR point clouds for
the agent. Weather relationships are qualitatively informed by literature, but
exact coefficients, power budgets, and perception parameters are experimental
assumptions requiring calibration. Weather currently changes sensing rather
than friction or traffic dynamics.

The first learning run compares adaptive sensing, learned always-on driving,
fixed IDLE, and random driving. Its short training budget and small held-out set
demonstrate a functioning pipeline, not convergence or a safety benefit. Lower
energy may reflect slower driving or earlier crashes, so inspect power, speed,
distance, completion, and collisions together.

Next steps include repeated training seeds, controlled periodic-sensing
comparisons, reward refinement, parameter sensitivity studies, and improved
fusion or memory. Nonterminating collision recovery and a CARLA adapter are not
yet implemented.

Project proposal and milestone docs are stored in [docs/](.).
