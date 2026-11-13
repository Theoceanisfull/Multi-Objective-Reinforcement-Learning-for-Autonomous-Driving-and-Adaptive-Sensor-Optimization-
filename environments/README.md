# Modular HighwayEnv sensing experiment

This is an object-level sensor-management environment built on the existing
HighwayEnv backbone. It models imperfect perception, not physical camera images,
point clouds, or radar waveforms. The original tutorial is unchanged.

## Run it

From the **project root**, using the prepared environment:

```bash
.venv/bin/python -m environments.demo --weather clear --sensing all_on
.venv/bin/python -m environments.demo --weather fog --sensing front --output runs/environment/fog_front
.venv/bin/python -m unittest discover -s environments/tests -v
```

The demo maintains the initial target lane/speed (`IDLE`); it does not train a
policy. It saves `preview.png`, the final observation, a trajectory CSV, and the
resolved experiment configuration. Existing files in the output folder are
replaced. Use separate output folders to preserve runs. `--sensing` supports
`all_on`, `radar_rgb`, `front`, and `off`. Fixed driving across sensing modes is
useful for inspecting perception and energy; it cannot demonstrate safer driving
from sensing. A learning controller comes later.

Dependencies are the same as `basic/requirements.txt`. The adapter was checked
against HighwayEnv **1.12.1** and Gymnasium **1.4.0**. No new packages are required.
The physics callback copies the short HighwayEnv `_simulate` loop: check it when
upgrading HighwayEnv, and run the stock-motion parity test.

## Use it in Python

```python
import numpy as np
from environments.sensor_highway import ExperimentConfig, SensorHighwayEnv

config = ExperimentConfig()
config.weather.preset = "rain"
config.weather.severity = 0.8
config.reward.uncertainty_weight = 0.0  # Compare with/without confidence shaping.
config.sensors[0].range_m = 120         # rgb_N

with SensorHighwayEnv(config) as env:
    observation, info = env.reset(seed=42)
    mask = np.array([name in {"rgb_N", "radar_N"}
                     for name in env.sensor_names], dtype=np.int8)
    observation, reward, terminated, truncated, info = env.step({
        "drive": 1,  # IDLE in the default five-action HighwayEnv mapping.
        "sensors": mask,
    })
```

Requesting a sensor ON and choosing a driving action happen together. The driving
action uses information already available; new measurements arrive later. Reset
starts with every sensor OFF and an unknown map. Warm-up, active operation, and
standby all have specified costs. Turning a sensor off cancels its pending work;
already delivered measurements remain until they age out. Reset before stepping
again after either ending flag. Stock collision termination is retained.

All 18 switches are independent: RGB N/NE/E/SE/S/SW/W/NW, then event cameras in
the same order, then front radar and 360-degree LiDAR. Use `env.sensor_names`, not
hard-coded indices. The action is a Dict with a driving Discrete action and a
MultiBinary sensing vector; it is not one enormous enumerated Discrete action.
Many RL libraries will need a custom policy or a wrapper selecting a small menu
of sensor configurations. The tutorial's tabular Q-table cannot directly consume
this observation.

## Change an experiment without editing implementation

`configs/default.json` contains the full default configuration:

```bash
.venv/bin/python -m environments.demo --config environments/configs/default.json --steps 10
```

Load/save with `ExperimentConfig.load(path)` and `config.save(path)`. Reconstruct
the environment after changes: it owns a deep copy of the configuration. To
remove an installed sensor, remove its entry from the sensor list. To leave it
installed but inactive, request zero in the action mask.

Each sensor has its own name, type, mounting direction, field of view, range,
acquisition rate, detection probability, noise, false-positive rate, latency,
warm-up, active power, and standby power. Units are documented in `config.py`.
Weather multipliers can be changed in JSON using, for example:

```json
"effect_overrides": {
  "rain": {
    "radar": [0.9, 0.9, 1.3, 2.0],
    "rgb": [0.65, 0.7, 2.0, 2.0]
  }
}
```

The four multipliers are detection probability, useful range, noise standard
deviation, and false-positive rate. They describe severity 1; severity 0 linearly
interpolates to clear conditions. Missing overrides use `weather.py` defaults.
These values are assumed, not fitted to the cited studies. The presets are
`clear`, `night`, `rain`, `fog`, `glare`, or `random` (one fixed preset per episode).
Weather changes perception only; friction and traffic behavior are unchanged.
There are no combined presets or within-episode weather transitions yet.

## Observation contract

The grid covers 150 m in every direction, with 2 m square cells: a 150 x 150
array covering 300 x 300 m. A circular valid-cell mask excludes corners outside
150 m. It does not extend any sensor's useful range. Coordinates rotate with the
car: top=N/forward, right=E/right, bottom=S/rear, left=W/left.
HighwayEnv uses screen-style world coordinates: at heading zero, positive world
y is the right-hand side (increasing lane index). Map projection and sensor
mounting follow that convention. Cameras are centered
45 degrees apart; their default 100-degree cones overlap rather than being
clipped to direction bins.

`observation` is a Dict:

| Key | Shape | Meaning |
| --- | --- | --- |
| `grid` | `(5, 150, 150)` by default | Channels described below |
| `valid_cells` | `(150, 150)` | 1 inside sensing-map radius; 0 outside |
| `sensors` | `(sensor_count, 4)` | Requested ON, ready, recent packet exists, normalized packet age |
| `ego` | `(3,)` | Speed in m/s, sine of world heading, cosine of world heading |

The five grid channels are:

1. **Occupancy:** heuristic probability, unknown=0.5.
2. **Support:** decaying evidence support, clipped to [0, 1].
3. **Age:** most recent supporting measurement age divided by `max_age_s`;
   unsupported cells=1.
4. **Radial velocity:** measurement-weighted, divided by 50 m/s, clipped to [-1, 1].
5. **Velocity support:** support for radial-velocity measurements, clipped to [0, 1].

All channels outside the circular mask are zero and must be masked by consumers.
The grid resolution is not a claim of 2 m measurement accuracy. The radial
velocity channel summarizes range rates at acquisition; it is not a full x/y
velocity field, and values from different viewpoints/ages are approximate.

Only ego telemetry is assumed always available. The simulator's full traffic
state is internal. Diagnostics in `info` include weather, crash status, reward
components, energy, and uncertainty; do not feed `info` into the policy.

## How sensing and fusion work

- RGB/event detections supply noisy world-frame bearing, **no metric depth**.
  Their evidence is distributed along a ray, rather than positioned at true
  vehicle distance. RGB semantics and learned monocular depth are not modeled.
- Radar supplies noisy range, bearing, and radial relative velocity.
- Conventional time-of-flight LiDAR supplies noisy range/bearing, not instant
  velocity. Elevation and 3-D shape are outside this 2-D abstraction.
- Detection probability declines quadratically toward each weather-adjusted
  maximum range. A Poisson process generates false detections.
- A 2-D center-ray silhouette test approximates occlusion. It treats all sensors
  alike and omits multipath, partial occlusion, and sensor mounting height.
- Event detection additionally depends on a relative image-motion/looming proxy.
  This assumes visible contrast; it does not render brightness events, flicker,
  or estimate detector performance from data. Equal-velocity scenes may produce
  little modeled activity.
- Each sensor's latest delivered packet contributes once to each reconstructed
  map. Evidence is capped within a packet, decays, broadens with age, and expires.
  Positions are reprojected as the ego moves. There is no object tracking or
  motion compensation; old detections may trail moving objects.
- Complementary angular/range evidence can reinforce occupancy. Ranging returns
  provide conservative free-space evidence along narrow rays before the return.
  Absence of detections never clears an entire view cone.

This is **heuristic evidence fusion**, not a calibrated Bayesian posterior or a
full association-aware tracker. It avoids true object IDs. It does not explicitly
resolve cross-sensor correlations or contradictions; overlapping cameras and
false detections can still cause excessive confidence. Occupancy entropy is a
confidence proxy, not measured perception error. Evaluate calibration and actual
errors against truth separately before interpreting uncertainty as reliability.
The grid's inverse-model evidence weights currently live in `grid.py` and should
be treated as additional experimental assumptions.

Physics defaults to 100 Hz and decisions to 2 Hz. Acquisition and delivery occur
on physics ticks, rounded upward by less than one physics interval. Sub-10 ms
latency differences are not resolved at the default rate. Power is integrated
on every tick; a new sensor's full active budget includes warm-up. No separate
switching-energy impulse or sensor wear is modeled.

## Reward and energy

```
reward = stock_driving_reward
       - energy_weight * joules_this_decision
       - uncertainty_weight * corridor_uncertainty * decision_duration
```

Optional route mode (`config.reward.route_distance_m`, default `None`) adds:

```
+ progress_weight * change_in_clipped_forward_distance
- time_weight * elapsed_seconds
+ completion_bonus                                    # safe finish only
+ early_completion_bonus * fraction_of_deadline_left   # safe finish only
- failure_penalty                                     # unsuccessful ending only
```

Distance is world-x displacement relative to reset, capped to [0, route length]
for progress rewards. Backtracking reduces progress reward. Success is checked
at decision boundaries; crash/off-road takes precedence over crossing the line.
Duration is the route deadline. The observation gains `task` (shape `(2,)`),
normalized remaining distance and time, assuming navigation/clock telemetry.
`info` reports `distance_m`, `route_completed`, and each reward component.
Route success sets `terminated`; a missed deadline sets `truncated`, but the
route trainer treats both as terminal targets. Legacy cutoffs still bootstrap.
Reward coefficients are experimental assumptions, not a guarantee of safe driving.

The relevant corridor is +/-6 m sideways, 30 m behind, and up to a 3-second speed
horizon ahead (minimum 30 m). It is an explicit geometric heuristic, not a lookup
of hidden vehicles. The uncertainty penalty can be disabled. It can reward
incorrect confidence, so compare with/without it and report actual driving and
perception performance separately.

The initial active budgets, **per installed unit**, are RGB 8 W, event 3 W,
radar 10 W, LiDAR 25 W. All-on totals 123 W; all-off standby totals 1.8 W. These
include allocated dedicated processing; shared compute, heating, cleaning, and
propulsion are excluded. Energy is reported in joules. No energy-per-kilometer
or validated safety result is claimed.

## Evidence versus assumptions

These references support qualitative design decisions, not the numeric defaults:

- [Requested sensor review](https://www.mdpi.com/1424-8220/26/7/2153): background
  taxonomy and sensor complementarity; not a common calibrated benchmark.
- [YOdar](https://arxiv.org/abs/2010.03320): uncertainty-aware camera/radar fusion
  outperforms individual baselines in its evaluated detection setting. This does
  not establish a universal fusion improvement factor.
- [DSEC](https://arxiv.org/abs/2103.06011): event/frame driving data in challenging
  illumination. HDR potential does not imply fog penetration or darkness immunity.
- [Weather measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC13568112/): real
  rain/fog effects and complex sensor-dependent degradation.
- [TI radar reference design](https://training-dev.ti.com/lit/pdf/tiduf12): range,
  bearing, and velocity measurement capabilities; device-specific geometry.
- [Prophesee EVK4 brief](https://www.prophesee.ai/wp-content/uploads/2023/09/EVK4-HD-Prophesee-Evaluation-Kit-Brief-13042023.pdf):
  0.5 W typical camera power and 220 microsecond sensor latency. Neither is the
  full processing-system cost or complete object-detection latency.

All exact detection probabilities, weather multipliers, motion thresholds,
perception noise, false-positive rates, processing power, warm-up, and pipeline
latencies in this implementation are **assumed baseline parameters**. Perform
sensitivity sweeps rather than reporting them as physical measurements. Fog
reduces both optical modalities equally by default; rain modestly degrades radar
rather than improving it. Glare is directional in world coordinates. LiDAR has
360-degree angular access, not unlimited visibility through occluders.

## File map and future simulator bridge

| File | Responsibility |
| --- | --- |
| `sensor_highway/config.py` | Typed configuration and JSON validation |
| `sensor_highway/backend.py` | HighwayEnv stepping and internal truth snapshots |
| `sensor_highway/weather.py` | Condition-dependent measurement degradation |
| `sensor_highway/sensors.py` | Activation, timing, energy, imperfect measurements |
| `sensor_highway/grid.py` | Inverse measurements, fusion, grid, uncertainty |
| `sensor_highway/env.py` | Gymnasium composition and reward accounting |
| `demo.py` | Fixed-policy run and inspectable artifacts |
| `tests/` | Behavioral and Gymnasium checks |

For a future CARLA integration, `Scene`/`VehicleTruth` and `Packet`/`Detection`
are the main boundaries to preserve. Replace backend stepping and truth snapshots
to retain abstract sensors, or translate processed simulator sensor data into
packets. `env.py` also needs its small backend-specific calls adapted; this is
not an implemented CARLA adapter. Replace the inverse measurement model or reward
calculation independently when the experiment requires it.
