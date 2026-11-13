"""Conservative inverse measurement map in the ego frame.

This is a heuristic evidence fusion baseline, not a calibrated Bayesian posterior.
Unknown cells start at p=.5. Each sensor contributes its latest packet once, so
repeated policy observations do not count stale data as fresh evidence. No truth
or object association IDs enter this module.
"""
import numpy as np
from .sensors import wrap_angle

CHANNELS = ("occupancy", "support", "age", "radial_velocity", "velocity_support")


class GridBuilder:
    def __init__(self, config):
        self.config = config
        self.size = round(2 * config.radius_m / config.cell_size_m)
        centers = (np.arange(self.size) + .5) * config.cell_size_m - config.radius_m
        # Top of the image is forward (N), right is starboard (E).
        self.right, self.forward = np.meshgrid(centers, centers[::-1])
        self.valid = self.right ** 2 + self.forward ** 2 <= config.radius_m ** 2

    def build(self, packets, ego_position, ego_heading, time, speed):
        shape = self.forward.shape
        evidence = np.zeros(shape)
        support = np.zeros(shape)
        newest = np.full(shape, self.config.max_age_s)
        velocity_sum, velocity_weight = np.zeros(shape), np.zeros(shape)
        c, s = np.cos(ego_heading), np.sin(ego_heading)
        # HighwayEnv uses screen-style coordinates: at heading zero, +y
        # is the vehicle's right (increasing lane index).
        world_x = ego_position[0] + c * self.forward - s * self.right
        world_y = ego_position[1] + s * self.forward + c * self.right
        for packet in packets:
            age = max(0, time - packet.capture_time)
            if age >= self.config.max_age_s:
                continue
            decay = np.exp(-age / self.config.evidence_decay_s)
            dx, dy = world_x - packet.origin[0], world_y - packet.origin[1]
            ranges = np.hypot(dx, dy)
            angles = np.arctan2(dy, dx)
            packet_evidence = np.zeros(shape)
            packet_support = np.zeros(shape)
            for detection in packet.detections:
                # Cell size and unmodeled motion broaden old evidence.
                angular_std = np.maximum(detection.bearing_std,
                                         self.config.cell_size_m / (2 * np.maximum(ranges, 1)))
                angular_std = np.hypot(angular_std, age * 2 / np.maximum(ranges, 1))
                angular = np.exp(-.5 * (wrap_angle(angles - detection.bearing) / angular_std) ** 2)
                angular[angular < .05] = 0
                envelope = ranges <= detection.max_range
                if detection.range_m is None:
                    # Bearing-only evidence is spread along a ray, never at truth depth.
                    occupied = .35 * angular * envelope
                    free = np.zeros(shape)
                else:
                    std = np.hypot(max(detection.range_std, self.config.cell_size_m / 2), age * 2)
                    occupied = 2 * angular * np.exp(-.5 * ((ranges - detection.range_m) / std) ** 2)
                    occupied *= envelope
                    # Only narrow rays with a return assert free space; no detection
                    # alone never clears an entire FOV. This is intentionally conservative.
                    free = .7 * angular * (ranges < detection.range_m - 3 * std) * envelope
                packet_evidence += occupied - free
                packet_support = np.maximum(packet_support, np.maximum(occupied, free))
                if detection.radial_velocity is not None:
                    weight = occupied * decay
                    velocity_sum += weight * detection.radial_velocity
                    velocity_weight += weight
            # Cap correlated detections within a sensor; independent sensor packets
            # can corroborate. Cross-camera correlations remain an approximation.
            evidence += np.clip(packet_evidence, -2, 2) * decay
            support += np.minimum(packet_support, 1) * decay
            newest[packet_support > .05] = np.minimum(newest[packet_support > .05], age)
        probability = 1 / (1 + np.exp(-np.clip(evidence, -6, 6)))
        entropy = -(probability * np.log2(probability)
                    + (1 - probability) * np.log2(1 - probability))
        # A local driving corridor is a relevance heuristic, not hidden traffic truth.
        horizon = min(self.config.radius_m, max(30, abs(speed) * 3))
        relevant = self.valid & (abs(self.right) <= 6) & (self.forward >= -30) & (self.forward <= horizon)
        uncertainty = float(entropy[relevant].mean())
        radial_velocity = np.divide(velocity_sum, velocity_weight,
                                    out=np.zeros(shape), where=velocity_weight > 1e-6)
        grid = np.stack([probability, np.clip(support, 0, 1),
                         newest / self.config.max_age_s,
                         np.clip(radial_velocity / 50, -1, 1),
                         np.clip(velocity_weight, 0, 1)]).astype(np.float32)
        grid[:, ~self.valid] = 0
        return grid, uncertainty
