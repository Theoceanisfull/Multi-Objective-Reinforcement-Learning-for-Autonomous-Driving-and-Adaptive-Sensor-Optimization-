"""Scripted, non-physical cut-in diagnostic using the existing sensor pipeline."""
import copy
from dataclasses import dataclass
import numpy as np
import pandas as pd
from environments.sensor_highway import WeatherConfig
from environments.sensor_highway.backend import VehicleTruth, Scene
from environments.sensor_highway.sensors import SensorSuite, wrap_angle
from environments.sensor_highway.weather import WeatherModel


@dataclass
class CutInSettings:
    onset_s: float = 1.0
    duration_s: float = 6.0
    physics_hz: int = 100
    lane_offset_m: float = 4.0
    tracking_window_s: float = 0.4
    minimum_samples: int = 5
    minimum_span_s: float = 0.2
    angular_alert_threshold_deg_s: float = 3.0
    decision_period_s: float = 0.5
    alert_sigma_margin: float = 2.0


def cutin_scene(time, gap_m, lateral_speed_m_s, settings):
    """Constant longitudinal gap; positive lateral_speed means moving from right to ego lane."""
    displacement = min(settings.lane_offset_m,
                       max(0., time - settings.onset_s) * lateral_speed_m_s)
    y = settings.lane_offset_m - displacement
    moving = settings.onset_s < time < settings.onset_s + settings.lane_offset_m / lateral_speed_m_s if lateral_speed_m_s else False
    vy = -lateral_speed_m_s if moving else 0.
    ego = VehicleTruth(np.array([25 * time, 0.]), np.array([25., 0.]), 0., 5., 2.)
    target = VehicleTruth(np.array([25 * time + gap_m, y]), np.array([25., vy]),
                          float(np.arctan2(vy, 25)), 5., 2.)
    delta = target.position - ego.position
    true_bearing = float(np.arctan2(delta[1], delta[0]))
    true_rate = float(np.rad2deg((delta[0] * vy) / (delta @ delta)))
    return Scene(ego, (target,)), true_bearing, true_rate, y


def bearing_rate(samples, settings):
    """Causal least-squares angular rate from capture-time-stamped bearings only."""
    if len(samples) < settings.minimum_samples:
        return np.nan
    times = np.array([s[0] for s in samples])
    if times[-1] - times[0] < settings.minimum_span_s - 1e-9:
        return np.nan
    bearings = np.unwrap([s[1] for s in samples])
    centered = times - times.mean()
    return float(np.rad2deg(centered @ (bearings - bearings.mean()) / (centered @ centered)))


def run_cutin(sensor, gap_m, lateral_speed_m_s, seed, settings=None, capture=False):
    """One known target, clutter disabled, sensors on before motion starts.

    Tracks only detections delivered so far. Empty packets do not refresh the
    track. This diagnostic cache is separate from the production map builder.
    Motion alerts infer bearing direction; no lane label or true range is input.
    """
    s = settings or CutInSettings()
    if gap_m <= 0 or lateral_speed_m_s < 0 or s.duration_s <= s.onset_s:
        raise ValueError('Invalid cut-in geometry/timing')
    if s.minimum_samples < 2 or s.tracking_window_s < s.minimum_span_s or s.physics_hz <= 0:
        raise ValueError('Invalid tracker settings')
    sensor = copy.deepcopy(sensor)
    sensor.false_positive_rate_hz = 0
    suite = SensorSuite([sensor], WeatherModel(WeatherConfig(preset='clear'),
                        np.random.default_rng(seed)), np.random.default_rng(seed + 1))
    suite.set_active([1], 0.)
    dt = 1 / s.physics_hz
    samples, seen_capture, last_detection = [], None, None
    first_detection = first_alert = first_decision = np.nan
    pre_alert = any_alert = False
    target_updates = motion_updates = motion_ticks = rate_ticks = age_ticks = 0
    errors, ages, bearing_errors, trace = [], [], [], []
    entry = s.onset_s + s.lane_offset_m / (2 * lateral_speed_m_s) if lateral_speed_m_s else np.nan
    for step in range(1, round(s.duration_s * s.physics_hz) + 1):
        t = step * dt
        scene, true_bearing, true_rate, y = cutin_scene(t, gap_m, lateral_speed_m_s, s)
        suite.tick(t, dt, scene)
        packet = suite.latest.get(0)
        new_detection = False
        if packet is not None and packet.capture_time != seen_capture:
            seen_capture = packet.capture_time
            if packet.detections:
                new_detection = True
                target_updates += 1
                d = packet.detections[0]  # Single target, zero false positives.
                last_detection = (packet.capture_time, d.bearing)
                samples.append(last_detection)
                _, capture_bearing, _, _ = cutin_scene(packet.capture_time, gap_m, lateral_speed_m_s, s)
                bearing_errors.append(np.rad2deg(wrap_angle(d.bearing - capture_bearing)))
                if packet.capture_time >= s.onset_s and np.isnan(first_detection):
                    first_detection = t - s.onset_s
        # Remove stale measurements against current time, not the last detection.
        samples = [(ct, b) for ct, b in samples if ct >= t - s.tracking_window_s - 1e-9]
        rate = bearing_rate(samples, s)
        # Known configured angular noise gives an uncertainty margin for the slope.
        times = np.array([ct for ct, _ in samples])
        spread = float(((times - times.mean()) ** 2).sum()) if len(times) else 0.
        slope_se = sensor.bearing_std_deg / np.sqrt(spread) if spread > 0 else np.inf
        alarm = bool(np.isfinite(rate) and rate + s.alert_sigma_margin * slope_se
                     < -s.angular_alert_threshold_deg_s)
        if alarm:
            any_alert = True
            if t < s.onset_s:
                pre_alert = True
            elif np.isnan(first_alert):
                first_alert = t - s.onset_s
        age = t - last_detection[0] if last_detection is not None else np.nan
        in_motion = lateral_speed_m_s > 0 and s.onset_s < t < s.onset_s + s.lane_offset_m / lateral_speed_m_s
        if in_motion:
            motion_ticks += 1
            rate_ticks += int(np.isfinite(rate))
            age_ticks += int(np.isfinite(age))
            if np.isfinite(age):
                ages.append(age)
            if new_detection:
                motion_updates += 1
                if np.isfinite(rate):
                    errors.append(rate - true_rate)  # Current-time tracking error includes delay.
        # A 2 Hz consumer can react only to the live causal tracker at its decision times.
        decision = abs(t / s.decision_period_s - round(t / s.decision_period_s)) < 1e-8
        if decision and t >= s.onset_s and alarm and np.isnan(first_decision):
            first_decision = t - s.onset_s
        if capture:
            trace.append({'time_s': t, 'target_y_m': y, 'true_bearing_deg': np.rad2deg(true_bearing),
                          'true_rate_deg_s': true_rate, 'estimated_rate_deg_s': rate,
                          'last_detection_age_s': age, 'motion_alarm': alarm,
                          'new_target_detection': new_detection, 'decision_tick': decision})
    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values else np.nan
    metrics = {'kind': sensor.kind, 'seed': seed, 'gap_m': gap_m,
               'lateral_speed_m_s': lateral_speed_m_s,
               'first_post_onset_detection_delay_s': first_detection,
               'first_post_onset_motion_alert_delay_s': first_alert,
               'first_2hz_motion_alert_delay_s': first_decision,
               'pre_onset_false_alert': pre_alert,
               'control_false_alert': any_alert if lateral_speed_m_s == 0 else np.nan,
               'alert_before_lane_entry': bool(np.isfinite(first_alert) and s.onset_s + first_alert < entry) if lateral_speed_m_s else np.nan,
               'lane_entry_time_s': entry, 'target_updates': target_updates,
               'motion_target_updates': motion_updates,
               'motion_mean_detection_age_s': float(np.mean(ages)) if ages else np.nan,
               'motion_rate_rmse_deg_s': rmse(errors),
               'motion_rate_available_fraction': rate_ticks / motion_ticks if motion_ticks else np.nan,
               'motion_detection_available_fraction': age_ticks / motion_ticks if motion_ticks else np.nan,
               'capture_bearing_rmse_deg': rmse(bearing_errors),
               'energy_j': suite.total_energy_j,
               'active_power_w': suite.total_energy_j / s.duration_s}
    return metrics, pd.DataFrame(trace)
