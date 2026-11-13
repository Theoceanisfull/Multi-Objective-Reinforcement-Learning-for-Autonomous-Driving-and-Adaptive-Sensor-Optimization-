"""Gymnasium interface composing the backend, sensors, weather, and observation."""
import copy
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .backend import HighwayAdapter
from .config import ExperimentConfig
from .grid import GridBuilder, CHANNELS
from .sensors import SensorSuite
from .weather import WeatherModel


class SensorHighwayEnv(gym.Env):
    """Joint actions: {'drive': HighwayEnv action ID, 'sensors': binary mask}.

    Reset starts with all sensors OFF and an unknown map. Each action requests
    sensing during the following interval; it cannot reveal information before
    the driving decision in that action. Stock crash termination is retained.
    Info includes privileged diagnostics and must not be used as policy input.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 2}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = copy.deepcopy(config if config is not None else ExperimentConfig())
        self.config.validate()
        self.render_mode = render_mode
        self.backend = HighwayAdapter(copy.deepcopy(self.config.highway), render_mode)
        self.mapper = GridBuilder(self.config.grid)
        count = len(self.config.sensors)
        self.action_space = spaces.Dict({"drive": spaces.Discrete(self.backend.action_space.n),
                                         "sensors": spaces.MultiBinary(count)})
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(-1, 1, (len(CHANNELS), self.mapper.size, self.mapper.size), np.float32),
            "valid_cells": spaces.MultiBinary((self.mapper.size, self.mapper.size)),
            "sensors": spaces.Box(0, 1, (count, 4), np.float32),
            # Internal vehicle telemetry is assumed always available. Units: m/s, radians.
            "ego": spaces.Box(-np.inf, np.inf, (3,), np.float32),
        })
        if self.config.reward.route_distance_m is not None:
            # Navigation/clock telemetry: normalized remaining route and deadline.
            self.observation_space.spaces["task"] = spaces.Box(0, 1, (2,), np.float32)
        self.suite = None
        self.finished = True

    @property
    def sensor_names(self):
        return [s.name for s in self.config.sensors]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.backend.on_tick = None
        self.backend.reset(seed=seed)
        # Dedicated streams isolate sensor/weather randomness from traffic generation.
        weather_rng = np.random.default_rng(int(self.np_random.integers(2**32)))
        sensor_rng = np.random.default_rng(int(self.np_random.integers(2**32)))
        self.weather = WeatherModel(self.config.weather, weather_rng)
        self.suite = SensorSuite(self.config.sensors, self.weather, sensor_rng)
        self.backend.on_tick = self.suite.tick
        self.time = 0.0
        self.start_x = float(self.backend.vehicle.position[0])
        self.distance_m = 0.0
        self.route_completed = False
        self.finished = False
        observation, uncertainty = self._observation()
        return observation, self._info(0, 0, uncertainty)

    def step(self, action):
        if self.finished:
            raise gym.error.ResetNeeded("Call reset() before stepping a new episode.")
        if not self.action_space.contains(action):
            raise ValueError("Action must contain a valid 'drive' and binary 'sensors' mask.")
        self.suite.set_active(action["sensors"], self.time)
        old_time, old_distance = self.time, self.distance_m
        old_energy = self.suite.total_energy_j
        _, driving_reward, terminated, truncated, backend_info = self.backend.step(int(action["drive"]))
        self.time = self.backend.steps / self.config.highway["simulation_frequency"]
        energy = self.suite.total_energy_j - old_energy
        self.distance_m = float(self.backend.vehicle.position[0]) - self.start_x
        r = self.config.reward
        route_mode = r.route_distance_m is not None
        self.route_completed = bool(route_mode and self.distance_m >= r.route_distance_m
                                    and not terminated and not backend_info["crashed"])
        terminated = bool(terminated or self.route_completed)
        observation, uncertainty = self._observation()
        dt = self.time - old_time
        energy_cost = self.config.reward.energy_weight * energy
        uncertainty_cost = self.config.reward.uncertainty_weight * uncertainty * dt
        components = {"driving": float(driving_reward),
                      "energy_penalty": -energy_cost,
                      "uncertainty_penalty": -uncertainty_cost}
        if route_mode:
            # Clipping stops overshoot earning extra progress. Backtracking loses reward.
            progress = np.clip(self.distance_m, 0, r.route_distance_m) - np.clip(old_distance, 0, r.route_distance_m)
            components.update(
                progress=float(r.progress_weight * progress), time_penalty=-r.time_weight * dt,
                completion_bonus=r.completion_bonus if self.route_completed else 0.,
                early_completion_bonus=(r.early_completion_bonus * max(0., 1 - self.time / self.config.highway["duration"])
                                        if self.route_completed else 0.),
                failure_penalty=-r.failure_penalty if (terminated or truncated) and not self.route_completed else 0.)
        reward = float(sum(components.values()))
        self.finished = bool(terminated or truncated)
        info = self._info(driving_reward, energy, uncertainty)
        info["reward_components"] = components
        info["crashed"] = bool(backend_info["crashed"])
        return observation, reward, bool(terminated), bool(truncated), info

    def _observation(self):
        ego = self.backend.scene().ego
        speed = float(np.linalg.norm(ego.velocity))
        grid, uncertainty = self.mapper.build(self.suite.latest.values(), ego.position,
                                              ego.heading, self.time, speed)
        observation = {"grid": grid, "valid_cells": self.mapper.valid.astype(np.int8),
                       "sensors": self.suite.status(self.time, self.config.grid.max_age_s),
                       "ego": np.array([speed, np.sin(ego.heading), np.cos(ego.heading)], dtype=np.float32)}
        if self.config.reward.route_distance_m is not None:
            observation["task"] = np.array([
                np.clip(1 - self.distance_m / self.config.reward.route_distance_m, 0, 1),
                np.clip(1 - self.time / self.config.highway["duration"], 0, 1)], dtype=np.float32)
        return observation, uncertainty

    def _info(self, driving_reward, energy, uncertainty):
        return {"weather": self.weather.preset, "time_s": self.time,
                "energy_j": float(energy), "total_energy_j": self.suite.total_energy_j,
                "uncertainty": uncertainty, "driving_reward": float(driving_reward),
                "distance_m": self.distance_m, "route_completed": self.route_completed}

    def render(self):
        return self.backend.render()

    def close(self):
        self.backend.close()
