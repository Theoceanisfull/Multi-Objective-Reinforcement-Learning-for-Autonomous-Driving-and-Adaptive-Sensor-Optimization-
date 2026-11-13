"""Object-level sensor/perception approximations, not rendered sensor physics.

Optical detections contain bearing only. Ranging detections contain noisy range;
radar alone supplies instantaneous radial velocity. No simulator object IDs leave
this layer. Occlusion is a simple 2-D silhouette approximation.
"""
from dataclasses import dataclass
import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class Detection:
    bearing: float  # World-frame bearing, radians.
    bearing_std: float
    range_m: float | None
    range_std: float
    radial_velocity: float | None
    max_range: float


@dataclass(frozen=True)
class Packet:
    sensor_index: int
    capture_time: float
    delivery_time: float
    origin: np.ndarray
    detections: tuple[Detection, ...]


class SensorSuite:
    def __init__(self, configs, weather, rng):
        self.configs, self.weather, self.rng = configs, weather, rng
        count = len(configs)
        self.active = np.zeros(count, dtype=bool)
        self.ready_at = np.zeros(count)
        self.next_capture = np.zeros(count)
        self.pending = []
        self.latest = {}  # Most recently delivered packet per sensor.
        self.total_energy_j = 0.0

    def set_active(self, mask, time):
        for i, requested in enumerate(mask):
            if requested and not self.active[i]:
                self.ready_at[i] = time + self.configs[i].warmup_s
                self.next_capture[i] = self.ready_at[i]
            elif not requested and self.active[i]:
                # Powering down cancels in-flight work; delivered evidence can age.
                self.pending = [p for p in self.pending if p.sensor_index != i]
        self.active = np.asarray(mask, dtype=bool).copy()

    def tick(self, time, dt, scene):
        energy = sum((s.power_w if self.active[i] else s.standby_w) * dt
                     for i, s in enumerate(self.configs))
        self.total_energy_j += energy
        for i, sensor in enumerate(self.configs):
            if self.active[i] and time + 1e-9 >= self.next_capture[i]:
                # Acquisition is quantized UP to a physics tick. Do not fabricate
                # sub-tick ground truth or catch up multiple captures at one state.
                self.pending.append(self._capture(i, time, scene))
                self.next_capture[i] += 1 / sensor.rate_hz
        delivered = [p for p in self.pending if p.delivery_time <= time + 1e-9]
        self.pending = [p for p in self.pending if p.delivery_time > time + 1e-9]
        for packet in delivered:
            self.latest[packet.sensor_index] = packet

    def status(self, time, max_age):
        rows = []
        for i in range(len(self.configs)):
            packet = self.latest.get(i)
            age = max_age if packet is None else min(max_age, time - packet.capture_time)
            rows.append([float(self.active[i]),
                         float(self.active[i] and time + 1e-9 >= self.ready_at[i]),
                         float(packet is not None and time - packet.capture_time < max_age),
                         age / max_age])
        return np.asarray(rows, dtype=np.float32)

    def _capture(self, index, time, scene):
        sensor = self.configs[index]
        ego = scene.ego
        axis = ego.heading + np.deg2rad(sensor.heading_deg)
        half_fov = np.deg2rad(sensor.fov_deg / 2)
        objects = []
        for vehicle in scene.traffic:
            delta = vehicle.position - ego.position
            distance = float(np.linalg.norm(delta))
            bearing = float(np.arctan2(delta[1], delta[0]))
            # Project the rectangular silhouette perpendicular to the viewing ray.
            phi = vehicle.heading - bearing
            half_width = (abs(np.sin(phi)) * vehicle.length
                          + abs(np.cos(phi)) * vehicle.width) / 2
            objects.append((distance, bearing, np.arctan2(half_width, max(distance, .1)), vehicle))
        objects.sort(key=lambda row: row[0])
        blockers, detections = [], []
        for distance, bearing, span, vehicle in objects:
            # Center ray occlusion, shared across modalities for the first model.
            occluded = any(abs(wrap_angle(bearing - angle)) < width
                           for angle, width in blockers)
            blockers.append((bearing, span))
            if occluded or abs(wrap_angle(bearing - axis)) > half_fov:
                continue
            effect = self.weather.effect(sensor.kind, bearing)
            max_range = sensor.range_m * effect.range_factor
            if distance > max_range:
                continue
            probability = (sensor.detection_probability * effect.detection_factor
                           * max(0, 1 - (distance / max_range) ** 2))
            if sensor.kind == "event":
                relative_velocity = vehicle.velocity - ego.velocity
                unit = (vehicle.position - ego.position) / max(distance, .1)
                radial = float(relative_velocity @ unit)
                transverse = float(relative_velocity @ np.array([-unit[1], unit[0]]))
                # Angular motion plus looming proxy. Assumes contrast is present;
                # it is not an event stream renderer or a calibrated detector.
                activity = abs(transverse) / max(distance, 1) + abs(radial) * span / max(distance, 1)
                probability *= min(1, activity / .02)
            if self.rng.random() < probability:
                detections.append(self._measurement(sensor, effect, distance, bearing,
                                                    vehicle.velocity - ego.velocity,
                                                    max_range))
        center_effect = self.weather.effect(sensor.kind, axis)
        false_count = self.rng.poisson(sensor.false_positive_rate_hz
                                      * center_effect.clutter_factor / sensor.rate_hz)
        for _ in range(false_count):
            bearing = axis + self.rng.uniform(-half_fov, half_fov)
            max_range = sensor.range_m * center_effect.range_factor
            detections.append(self._measurement(sensor, center_effect,
                                                self.rng.uniform(1, max(1.01, max_range)),
                                                bearing, np.zeros(2), max_range))
        return Packet(index, time, time + sensor.latency_s,
                      ego.position.copy(), tuple(detections))

    def _measurement(self, sensor, effect, distance, bearing, velocity, max_range):
        bearing_std = np.deg2rad(sensor.bearing_std_deg * effect.noise_factor)
        measured_bearing = bearing + self.rng.normal(0, bearing_std)
        measured_range, radial = None, None
        if sensor.kind in {"radar", "lidar"}:
            measured_range = float(np.clip(distance + self.rng.normal(
                0, sensor.range_std_m * effect.noise_factor), 0, max_range))
        if sensor.kind == "radar":
            unit = np.array([np.cos(bearing), np.sin(bearing)])
            radial = float(velocity @ unit + self.rng.normal(
                0, sensor.radial_velocity_std_m_s * effect.noise_factor))
        return Detection(measured_bearing, bearing_std, measured_range,
                         sensor.range_std_m * effect.noise_factor, radial, max_range)
