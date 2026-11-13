"""Qualitative literature-informed effects with explicitly assumed coefficients.

Rain never improves radar relative to clear weather. Event HDR benefits do not
imply fog immunity. Presets are fixed within an episode in this first version.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WeatherEffect:
    detection_factor: float = 1
    range_factor: float = 1
    noise_factor: float = 1
    clutter_factor: float = 1


# At severity=1: detection multiplier, useful-range multiplier, noise, clutter.
EFFECTS = {
    "clear": {},
    "night": {"rgb": (0.4, 0.6, 2.5, 1.5), "event": (0.65, 0.75, 2, 1.5)},
    "rain": {"rgb": (0.65, 0.7, 2, 2), "event": (0.65, 0.7, 2, 3),
             "radar": (0.9, 0.9, 1.3, 2), "lidar": (0.7, 0.7, 2, 3)},
    "fog": {"rgb": (0.35, 0.35, 3, 1.5), "event": (0.35, 0.35, 3, 1.5),
            "radar": (0.98, 0.98, 1.1, 1.1), "lidar": (0.4, 0.4, 3, 4)},
    "glare": {"rgb": (0.3, 0.6, 3, 2), "event": (0.75, 0.85, 1.5, 1.5),
              "lidar": (0.9, 0.9, 1.3, 1.3)},
}


class WeatherModel:
    def __init__(self, config, rng):
        self.config = config
        self.preset = (str(rng.choice(list(EFFECTS)))
                       if config.preset == "random" else config.preset)

    def effect(self, kind, world_bearing):
        severity = self.config.severity
        if self.preset == "glare":
            # Directional light source; compare the target ray to world sun angle.
            delta = world_bearing - np.deg2rad(self.config.glare_heading_deg)
            severity *= max(0.0, float(np.cos(delta))) ** 8
        overrides = self.config.effect_overrides.get(self.preset, {})
        values = overrides.get(kind, EFFECTS[self.preset].get(kind, (1, 1, 1, 1)))
        return WeatherEffect(*(1 + severity * (v - 1) for v in values))
