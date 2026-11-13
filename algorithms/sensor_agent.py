"""Linear Expected SARSA using observations only; no extra ML dependencies.

A compact first experiment, not a guaranteed convergent or safe controller.
The feature map loses information and has no recurrent memory.
"""
from dataclasses import dataclass
import numpy as np
from experiments.notebook_helpers import sensor_mask

FEATURE_SCHEMA = "sensor_grid_regions_v1"
ROUTE_FEATURE_SCHEMA = "sensor_grid_regions_route_v2"
DRIVE_NAMES = ("LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER")
SENSING_MODES = ("off", "front", "radar_rgb", "all_on")


class ObservationEncoder:
    """Pool distance bands and left/front/right regions without reading truth.

    Values are bounded approximately to [-1,1]. Unknown occupancy is encoded as
    zero evidence, accompanied by separate support/age features, not as free road.
    """
    def __init__(self, config):
        self.config = config
        self.feature_schema = ROUTE_FEATURE_SCHEMA if config.reward.route_distance_m is not None else FEATURE_SCHEMA
        size = round(2 * config.grid.radius_m / config.grid.cell_size_m)
        centers = (np.arange(size) + .5) * config.grid.cell_size_m - config.grid.radius_m
        right, forward = np.meshgrid(centers, centers[::-1])
        self.regions = []
        for low, high in [(-30, 0), (0, 15), (15, 30), (30, 60), (60, 100)]:
            for left, right_edge in [(-8, -2), (-2, 2), (2, 8)]:
                region = (forward >= low) & (forward < high) & (right >= left) & (right < right_edge)
                self.regions.append(region)
        self.groups = [np.array([s.kind == kind for s in config.sensors])
                       for kind in ["rgb", "event", "radar", "lidar"]]

    def __call__(self, obs):
        speed, heading_sin, heading_cos = obs["ego"]
        features = [1., np.clip(speed / 40, 0, 1), heading_sin, heading_cos]
        grid, valid = obs["grid"], obs["valid_cells"].astype(bool)
        for region in self.regions:
            use = region & valid
            if not use.any():
                features.extend([0.] * 6)
                continue
            occupancy = grid[0][use]
            support = grid[1][use]
            velocity_support = grid[4][use]
            features.extend([
                max(0., float((occupancy.max() - .5) * 2)),
                float(np.maximum(occupancy - .5, 0).mean() * 2),
                float(np.maximum(.5 - occupancy, 0).mean() * 2),
                float(support.mean()), float(((1 - grid[2][use]) * support).mean()),
                float((grid[3][use] * velocity_support).sum() / max(1., velocity_support.sum())),
            ])
        for group in self.groups:
            features.extend(obs["sensors"][group].mean(axis=0) if group.any() else [0.] * 4)
        if self.config.reward.route_distance_m is not None:
            features.extend(obs["task"])
        return np.asarray(features, dtype=np.float64)


@dataclass
class AgentSettings:
    episodes: int = 80
    alpha: float = 0.12
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1


class LinearExpectedSarsa:
    def __init__(self, feature_count, modes=SENSING_MODES, feature_schema=FEATURE_SCHEMA):
        self.feature_schema = feature_schema
        self.modes = tuple(modes)
        self.actions = [(drive, mode) for mode in self.modes for drive in DRIVE_NAMES]
        self.weights = np.zeros((len(self.actions), feature_count), dtype=np.float64)

    def values(self, features):
        return self.weights @ features

    def select(self, features, epsilon, rng):
        if rng.random() < epsilon:
            return int(rng.integers(len(self.actions)))
        values = self.values(features)
        return int(rng.choice(np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-12))))

    def update(self, features, action, reward, next_features, terminated, epsilon, alpha, gamma):
        next_values = self.values(next_features)
        expectation = (1 - epsilon) * next_values.max() + epsilon * next_values.mean()
        target = reward + (0 if terminated else gamma * expectation)
        td_error = target - self.values(features)[action]
        # Normalize step length so feature dimension does not set the learning rate.
        self.weights[action] += alpha * td_error * features / max(1., features @ features)
        if not np.isfinite(self.weights).all():
            raise FloatingPointError("Learning diverged; reduce alpha and inspect features.")
        return float(td_error)

    def decode(self, action, env):
        drive_name, mode = self.actions[action]
        mapping = env.backend.action_type.actions
        drive_id = next(i for i, name in mapping.items() if name == drive_name)
        return {"drive": drive_id, "sensors": sensor_mask(env.config, mode)}

    def save(self, path):
        np.savez_compressed(path, weights=self.weights, modes=np.array(self.modes),
                            feature_schema=np.array(self.feature_schema))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data["feature_schema"]) not in {FEATURE_SCHEMA, ROUTE_FEATURE_SCHEMA}:
                raise ValueError("Incompatible feature schema")
            weights = data["weights"]
            agent = cls(weights.shape[1], modes=data["modes"].tolist(), feature_schema=str(data["feature_schema"]))
            if weights.shape != agent.weights.shape or not np.isfinite(weights).all():
                raise ValueError("Invalid agent weights")
            agent.weights[:] = weights
        return agent
