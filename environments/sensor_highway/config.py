"""Editable model assumptions; distances in m, time in s, power in W.

These defaults are hypotheses for sensitivity experiments, not calibrated hardware.
"""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

DIRECTIONS = {"N": 0, "NE": 45, "E": 90, "SE": 135,
              "S": 180, "SW": 225, "W": 270, "NW": 315}


@dataclass
class SensorConfig:
    name: str
    kind: str
    heading_deg: float = 0  # Clockwise from the vehicle's forward direction.
    fov_deg: float = 100
    range_m: float = 100
    rate_hz: float = 20
    latency_s: float = 0.08  # Full modeled perception pipeline, not pixel latency.
    warmup_s: float = 0.1
    power_w: float = 8  # Sensor plus allocated dedicated processing.
    standby_w: float = 0.1
    detection_probability: float = 0.95
    range_std_m: float = 1
    bearing_std_deg: float = 1
    radial_velocity_std_m_s: float = 0.5
    false_positive_rate_hz: float = 0.1


@dataclass
class WeatherConfig:
    preset: str = "clear"  # clear, night, rain, fog, glare; or random per episode.
    severity: float = 0.7
    glare_heading_deg: float = 0  # World angle: 0 follows HighwayEnv's +x axis.
    # Optional preset -> sensor kind -> [detection, range, noise, clutter] multipliers.
    effect_overrides: dict = field(default_factory=dict)


@dataclass
class GridConfig:
    radius_m: float = 150
    cell_size_m: float = 2
    evidence_decay_s: float = 0.75
    max_age_s: float = 3


@dataclass
class RewardConfig:
    # None preserves the original continuing-driving task.
    route_distance_m: float | None = None
    progress_weight: float = 0.02  # Reward per meter, capped at the finish line.
    time_weight: float = 0.1  # Penalty per elapsed second in route mode.
    completion_bonus: float = 10.0
    early_completion_bonus: float = 5.0  # Scaled by fraction of deadline remaining.
    failure_penalty: float = 10.0  # Crash, off-road, or deadline without completion.
    energy_weight: float = 0.001  # Reward units per joule.
    uncertainty_weight: float = 0.1  # Reward units per second of normalized uncertainty.


def default_sensors():
    sensors = []
    for kind in ("rgb", "event"):
        for direction, angle in DIRECTIONS.items():
            sensors.append(SensorConfig(
                name=f"{kind}_{direction}", kind=kind, heading_deg=angle,
                range_m=100 if kind == "rgb" else 60,
                rate_hz=20 if kind == "rgb" else 50,
                latency_s=0.08 if kind == "rgb" else 0.02,
                power_w=8 if kind == "rgb" else 3,
                bearing_std_deg=1 if kind == "rgb" else 1.5))
    sensors.extend([
        SensorConfig("radar_N", "radar", fov_deg=120, range_m=150,
                     latency_s=0.05, power_w=10, bearing_std_deg=3),
        SensorConfig("lidar", "lidar", fov_deg=360, range_m=100,
                     rate_hz=10, latency_s=0.1, power_w=25,
                     range_std_m=0.2, bearing_std_deg=0.3),
    ])
    return sensors


@dataclass
class ExperimentConfig:
    sensors: list[SensorConfig] = field(default_factory=default_sensors)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    highway: dict = field(default_factory=lambda: {
        "lanes_count": 3, "vehicles_count": 12, "duration": 20,
        "simulation_frequency": 100, "policy_frequency": 2,
        "action": {"type": "DiscreteMetaAction", "target_speeds": [20, 25, 30]},
        "collision_reward": -5, "high_speed_reward": 0.4,
        "right_lane_reward": 0, "normalize_reward": False,
        "reward_speed_range": [20, 30], "offroad_terminal": True,
        "offscreen_rendering": True,
    })

    def validate(self):
        if not self.sensors or len({s.name for s in self.sensors}) != len(self.sensors):
            raise ValueError("Supply sensors with unique names.")
        if self.weather.preset not in {"clear", "night", "rain", "fog", "glare", "random"}:
            raise ValueError("Unknown weather preset.")
        if not 0 <= self.weather.severity <= 1:
            raise ValueError("Weather severity must lie in [0, 1].")
        for preset, effects in self.weather.effect_overrides.items():
            if preset not in {"clear", "night", "rain", "fog", "glare"}:
                raise ValueError(f"Unknown weather override preset: {preset}")
            for kind, values in effects.items():
                if kind not in {"rgb", "event", "radar", "lidar"} or len(values) != 4:
                    raise ValueError("Weather overrides require a sensor kind and four multipliers.")
                if min(values) <= 0 or max(values[:2]) > 1:
                    raise ValueError("Detection/range multipliers must be in (0, 1]; noise/clutter positive.")
        g = self.grid
        if min(g.radius_m, g.cell_size_m, g.evidence_decay_s, g.max_age_s) <= 0:
            raise ValueError("Grid dimensions and timing must be positive.")
        if not math.isclose(2 * g.radius_m / g.cell_size_m,
                            round(2 * g.radius_m / g.cell_size_m)):
            raise ValueError("The grid diameter must be divisible by cell size.")
        sf, pf = self.highway["simulation_frequency"], self.highway["policy_frequency"]
        if sf < pf or pf <= 0 or not float(sf / pf).is_integer():
            raise ValueError("Physics frequency must be a positive integer multiple of policy frequency.")
        if self.highway.get("action", {}).get("type") != "DiscreteMetaAction":
            raise ValueError("This adapter requires DiscreteMetaAction.")
        for s in self.sensors:
            if s.kind not in {"rgb", "event", "radar", "lidar"}:
                raise ValueError(f"Unknown sensor kind: {s.kind}")
            if min(s.range_m, s.rate_hz, s.range_std_m, s.bearing_std_deg,
                   s.radial_velocity_std_m_s) <= 0 or not 0 < s.fov_deg <= 360:
                raise ValueError(f"Invalid geometry, frequency, or noise: {s.name}")
            if min(s.latency_s, s.warmup_s, s.power_w, s.standby_w,
                   s.false_positive_rate_hz) < 0 or not 0 <= s.detection_probability <= 1:
                raise ValueError(f"Invalid timing, power, or detection probability: {s.name}")
            if s.rate_hz > sf:
                raise ValueError(f"{s.name}: acquisition rate exceeds physics rate.")
        if min(self.reward.energy_weight, self.reward.uncertainty_weight) < 0:
            raise ValueError("Reward penalty weights must be nonnegative.")
        r = self.reward
        if r.route_distance_m is not None and r.route_distance_m <= 0:
            raise ValueError("Route distance must be positive or None.")
        if min(r.progress_weight, r.time_weight, r.completion_bonus,
               r.early_completion_bonus, r.failure_penalty) < 0:
            raise ValueError("Route reward weights must be nonnegative.")
        if self.highway["duration"] <= 0:
            raise ValueError("Episode duration must be positive.")
        # JSON serialization also rejects NaN and infinity in nested configuration.
        json.dumps(asdict(self), allow_nan=False)

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        for key, constructor in (("weather", WeatherConfig), ("grid", GridConfig),
                                 ("reward", RewardConfig)):
            if key in data:
                data[key] = constructor(**data[key])
        if "sensors" in data:
            data["sensors"] = [SensorConfig(**s) for s in data["sensors"]]
        result = cls(**data)
        result.validate()
        return result
