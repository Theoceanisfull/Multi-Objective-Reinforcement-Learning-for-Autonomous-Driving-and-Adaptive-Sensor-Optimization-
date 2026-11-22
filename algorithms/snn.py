"""Spiking Q approximators for online Expected SARSA.

Modes: surrogate (end-to-end surrogate semi-gradients), reservoir (fixed hidden
synapses), and td_stdp (heuristic TD-modulated hidden STDP + semi-gradient readout).
Membranes and spike traces reset for every observation. No replay/target network.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn

from algorithms.sensor_agent import (
    FEATURE_SCHEMA, ROUTE_FEATURE_SCHEMA, DRIVE_NAMES, SENSING_MODES,
)
from experiments.notebook_helpers import sensor_mask


@dataclass
class SNNSettings:
    episodes: int = 60
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    hidden_size: int = 64
    simulation_steps: int = 16
    membrane_decay: float = 0.9
    threshold: float = 1.0
    surrogate_slope: float = 5.0
    gradient_clip: float = 10.0
    stdp_learning_rate: float = 0.001
    trace_decay: float = 0.8
    stdp_depression: float = 1.05
    td_modulation_clip: float = 5.0
    weight_bound: float = 1.0

    def validate(self):
        for name in ('episodes', 'hidden_size', 'simulation_steps'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('learning_rate', 'threshold', 'surrogate_slope', 'gradient_clip',
                     'stdp_learning_rate', 'stdp_depression', 'td_modulation_clip', 'weight_bound'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not (0 <= self.gamma < 1 and 0 <= self.membrane_decay < 1
                and 0 <= self.trace_decay < 1
                and 0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError('Invalid discount, decay, or exploration settings')


class SurrogateSpike(torch.autograd.Function):
    """Hard forward threshold; fast-sigmoid surrogate derivative in backward."""
    @staticmethod
    def forward(ctx, voltage, slope):
        ctx.save_for_backward(voltage)
        ctx.slope = slope
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        voltage, = ctx.saved_tensors
        return grad_output / (1 + ctx.slope * voltage.abs()).square(), None


class SpikingQNetwork(nn.Module):
    """Deterministic ON/OFF input spikes -> LIF neurons -> signed analog Q readout.

    Positive and negative feature channels preserve signed observations. Each
    call simulates one independent observation window, so evaluating s' cannot
    contaminate s. The window has computational steps, not environment seconds.
    """
    def __init__(self, feature_count, action_count, settings):
        super().__init__()
        self.settings = settings
        self.input = nn.Linear(2 * feature_count, settings.hidden_size)
        self.readout = nn.Linear(settings.hidden_size, action_count)
        nn.init.normal_(self.input.weight, mean=0., std=0.12)
        nn.init.constant_(self.input.bias, 0.2)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, features, collect_stdp=False):
        cfg = self.settings
        drive = torch.cat((features.clamp(0, 1), (-features).clamp(0, 1)))
        phase = torch.zeros_like(drive)
        voltage = torch.zeros_like(self.input.bias)
        count = torch.zeros_like(voltage)
        pre_trace, post_trace = torch.zeros_like(drive), torch.zeros_like(voltage)
        eligibility = torch.zeros_like(self.input.weight)
        for _ in range(cfg.simulation_steps):
            phase = phase + drive
            pre = (phase >= 1).to(drive.dtype)
            phase = phase - pre
            voltage = cfg.membrane_decay * voltage + self.input(pre)
            post = SurrogateSpike.apply(voltage - cfg.threshold, cfg.surrogate_slope)
            # Detach reset: use the surrogate for spike production, not reset feedback.
            voltage = voltage - post.detach() * cfg.threshold
            count = count + post
            if collect_stdp:
                # Strictly earlier spikes contribute; simultaneous pairs are excluded.
                with torch.no_grad():
                    eligibility += (torch.outer(post, pre_trace)
                                    - cfg.stdp_depression * torch.outer(post_trace, pre))
                    pre_trace = cfg.trace_decay * pre_trace + pre
                    post_trace = cfg.trace_decay * post_trace + post
        rates = count / cfg.simulation_steps
        return self.readout(rates), rates, eligibility / cfg.simulation_steps


class SpikingExpectedSarsa:
    """Same policy interface as LinearExpectedSarsa; learning remains on-policy.

    td_stdp is a hybrid, not an exact gradient or a proven SARSA-STDP equivalence.
    Only the current observation's eligibility is modulated, never next-state
    activity. Eligibility spans spike timing within a decision, not SARSA(lambda)
    traces across environment decisions. Policy checkpoints do not resume Adam.
    """
    MODES = ('surrogate', 'reservoir', 'td_stdp')

    def __init__(self, feature_count, settings=None, modes=SENSING_MODES,
                 feature_schema=FEATURE_SCHEMA, seed=42, learning_rule='surrogate'):
        self.settings = SNNSettings(**asdict(settings or SNNSettings()))
        self.settings.validate()
        if (type(feature_count) is not int or feature_count <= 0 or not modes
                or any(mode not in SENSING_MODES for mode in modes)):
            raise ValueError('Supply a positive feature count and supported sensing modes')
        if feature_schema not in {FEATURE_SCHEMA, ROUTE_FEATURE_SCHEMA}:
            raise ValueError('Incompatible feature schema')
        if learning_rule not in self.MODES:
            raise ValueError(f'learning_rule must be one of {self.MODES}')
        self.feature_count, self.feature_schema = feature_count, feature_schema
        self.modes, self.learning_rule = tuple(modes), learning_rule
        self.actions = [(drive, mode) for mode in self.modes for drive in DRIVE_NAMES]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = SpikingQNetwork(feature_count, len(self.actions), self.settings)
        if learning_rule != 'surrogate':
            self.network.input.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            [p for p in self.network.parameters() if p.requires_grad], lr=self.settings.learning_rate)
        self.q_updates = 0
        self.last_metrics = {}

    def _features(self, features):
        result = torch.as_tensor(features, dtype=torch.float32)
        if result.shape != (self.feature_count,) or not torch.isfinite(result).all():
            raise ValueError('Expected one finite feature vector of the configured size')
        return result

    def values(self, features):
        with torch.no_grad():
            return self.network(self._features(features))[0].numpy().copy()

    def select(self, features, epsilon, rng):
        if not 0 <= epsilon <= 1:
            raise ValueError('epsilon must be in [0, 1]')
        if rng.random() < epsilon:
            return int(rng.integers(len(self.actions)))
        values = self.values(features)
        return int(rng.choice(np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-12))))

    def decode(self, action, env):
        drive_name, mode = self.actions[action]
        drive_id = next(i for i, name in env.backend.action_type.actions.items() if name == drive_name)
        return {'drive': drive_id, 'sensors': sensor_mask(env.config, mode)}

    def td_target(self, reward, next_features, terminal, epsilon):
        if not math.isfinite(reward) or not 0 <= epsilon <= 1:
            raise ValueError('Reward must be finite and epsilon in [0, 1]')
        with torch.no_grad():
            if terminal:
                return torch.tensor(float(reward), dtype=torch.float32)
            next_q = self.network(self._features(next_features))[0]
            expected_q = (1 - epsilon) * next_q.max() + epsilon * next_q.mean()
            return float(reward) + self.settings.gamma * expected_q

    def update(self, features, action, reward, next_features, terminal, epsilon):
        if not 0 <= action < len(self.actions):
            raise ValueError('Invalid action index')
        target = self.td_target(reward, next_features, terminal, epsilon)
        q, rates, eligibility = self.network(self._features(features), self.learning_rule == 'td_stdp')
        delta = target - q[action]
        loss = 0.5 * delta.square()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.settings.gradient_clip)
        self.optimizer.step()
        if self.learning_rule == 'td_stdp':
            with torch.no_grad():
                # Global TD modulation of local pair-STDP is a heuristic hidden
                # representation update. The Q readout above still learns by TD.
                modulation = delta.detach().clamp(-self.settings.td_modulation_clip,
                                                  self.settings.td_modulation_clip)
                self.network.input.weight.add_(self.settings.stdp_learning_rate * modulation * eligibility)
                self.network.input.weight.clamp_(-self.settings.weight_bound, self.settings.weight_bound)
        if not all(torch.isfinite(p).all() for p in self.network.parameters()):
            raise FloatingPointError('SNN learning diverged')
        self.q_updates += 1
        self.last_metrics = {'q_loss': float(loss.detach()), 'hidden_spike_rate': float(rates.detach().mean()),
                             'silent_hidden_fraction': float((rates.detach() == 0).float().mean()),
                             'mean_abs_eligibility': float(eligibility.abs().mean())}
        return float(delta.detach())

    def save(self, path):
        torch.save({'format_version': 1, 'feature_count': self.feature_count,
                    'feature_schema': self.feature_schema, 'modes': self.modes,
                    'learning_rule': self.learning_rule, 'settings': asdict(self.settings),
                    'network': self.network.state_dict(), 'q_updates': self.q_updates}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['format_version'] != 1:
            raise ValueError('Unsupported SNN checkpoint format')
        agent = cls(data['feature_count'], SNNSettings(**data['settings']), data['modes'],
                    data['feature_schema'], learning_rule=data['learning_rule'])
        agent.network.load_state_dict(data['network'])
        if not all(torch.isfinite(p).all() for p in agent.network.parameters()):
            raise ValueError('Invalid network weights')
        agent.q_updates = data['q_updates']
        return agent


@dataclass
class AdaptiveSNNSettings(SNNSettings):
    """Dimensionless adaptive LIF experiment, not a full Diehl-Cook replica."""
    threshold_decay: float = 0.95
    threshold_increment: float = 0.1
    prototype_weight_sum: float = 4.0
    prototype_weight_max: float = 0.25
    prototype_learning_rate: float = 0.002
    inhibition: float = 0.1

    def validate(self):
        super().validate()
        if not 0 <= self.threshold_decay < 1:
            raise ValueError('threshold_decay must be in [0, 1)')
        for name in ('threshold_increment', 'prototype_weight_sum',
                     'prototype_weight_max', 'prototype_learning_rate'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.inhibition) or self.inhibition < 0:
            raise ValueError('inhibition must be finite and nonnegative')


class AdaptiveSpikingQNetwork(SpikingQNetwork):
    """Explicit pretraining can persist theta; Q inference never mutates it.

    Adaptive reservoir: signed frozen random synapses, no inhibition, theta
    starts at zero and adapts locally within each observation window.
    Prototype variants: nonnegative normalized synapses, inhibitory feedback,
    theta carried during pretraining, then fixed during Q learning and tests.
    These differences are why a matched untrained prototype control is included.
    """
    def __init__(self, feature_count, action_count, settings, variant):
        super().__init__(feature_count, action_count, settings)
        self.variant = variant
        self.register_buffer('threshold_offset', torch.zeros(settings.hidden_size))
        if variant != 'adaptive_reservoir':
            with torch.no_grad():
                self.input.weight.abs_()
                if settings.prototype_weight_sum > 2 * feature_count * settings.prototype_weight_max:
                    raise ValueError('Prototype weight sum exceeds bounded synaptic capacity')
                self.normalize_prototypes()

    def normalize_prototypes(self):
        cfg = self.settings
        weights = self.input.weight
        weights.clamp_(min=1e-8)
        weights.mul_(cfg.prototype_weight_sum / weights.sum(dim=1, keepdim=True))
        for _ in range(30):
            weights.clamp_(max=cfg.prototype_weight_max)
            missing = cfg.prototype_weight_sum - weights.sum(dim=1, keepdim=True)
            capacity = (cfg.prototype_weight_max - weights).clamp(min=0)
            weights.add_(missing * capacity / capacity.sum(dim=1, keepdim=True).clamp(min=1e-12))

    def encode(self, features, collect_stdp=False, adapt=False):
        cfg = self.settings
        drive = torch.cat((features.clamp(0, 1), (-features).clamp(0, 1)))
        phase = torch.zeros_like(drive)
        voltage = torch.zeros_like(self.input.bias)
        count = torch.zeros_like(voltage)
        theta = self.threshold_offset.clone()
        pre_trace, post_trace = torch.zeros_like(drive), torch.zeros_like(voltage)
        eligibility = torch.zeros_like(self.input.weight)
        previous_post = torch.zeros_like(voltage)
        for _ in range(cfg.simulation_steps):
            phase = phase + drive
            pre = (phase >= 1).to(drive.dtype)
            phase = phase - pre
            inhibition = (cfg.inhibition * (previous_post.sum() - previous_post)
                          if self.variant != 'adaptive_reservoir' else 0.)
            voltage = cfg.membrane_decay * voltage + self.input(pre) - inhibition
            threshold = cfg.threshold + theta
            post = (voltage >= threshold).to(voltage.dtype)
            voltage = voltage - post * threshold
            count = count + post
            if collect_stdp:
                # Weight-dependent pair STDP; earlier events only. Not reward-modulated.
                eligibility += (torch.outer(post, pre_trace) * (cfg.prototype_weight_max - self.input.weight)
                                - cfg.stdp_depression * torch.outer(post_trace, pre) * self.input.weight)
                pre_trace = cfg.trace_decay * pre_trace + pre
                post_trace = cfg.trace_decay * post_trace + post
            if adapt:
                theta = cfg.threshold_decay * theta + cfg.threshold_increment * post
            previous_post = post
        return count / cfg.simulation_steps, eligibility / cfg.simulation_steps, theta

    def forward(self, features, collect_stdp=False):
        with torch.no_grad():
            rates, eligibility, _ = self.encode(features, collect_stdp,
                                               adapt=self.variant == 'adaptive_reservoir')
        return self.readout(rates), rates, eligibility


class AdaptiveExpectedSarsa(SpikingExpectedSarsa):
    """Frozen adaptive/prototype representation plus learned Expected SARSA readout."""
    VARIANTS = ('adaptive_reservoir', 'prototype_control', 'stdp_pretrained')

    def __init__(self, feature_count, settings=None, modes=SENSING_MODES,
                 feature_schema=FEATURE_SCHEMA, seed=42, variant='adaptive_reservoir'):
        cfg = settings or AdaptiveSNNSettings()
        cfg.validate()
        if variant not in self.VARIANTS:
            raise ValueError(f'variant must be one of {self.VARIANTS}')
        # Base initialization supplies policy validation and the common interface.
        base = SNNSettings(**{k: asdict(cfg)[k] for k in SNNSettings.__dataclass_fields__})
        super().__init__(feature_count, base, modes, feature_schema, seed, learning_rule='reservoir')
        self.settings = AdaptiveSNNSettings(**asdict(cfg))
        self.variant = variant
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = AdaptiveSpikingQNetwork(feature_count, len(self.actions), self.settings, variant)
        self.network.input.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.network.readout.parameters(), lr=cfg.learning_rate)
        self.pretraining_updates = 0

    def pretrain_step(self, features):
        if self.variant == 'adaptive_reservoir':
            raise ValueError('The adaptive reservoir needs no pretraining')
        cfg = self.settings
        with torch.no_grad():
            rates, eligibility, theta = self.network.encode(
                self._features(features), collect_stdp=self.variant == 'stdp_pretrained', adapt=True)
            if self.variant == 'stdp_pretrained':
                weights = self.network.input.weight
                weights.add_(cfg.prototype_learning_rate * eligibility).clamp_(min=1e-8)
                self.network.normalize_prototypes()
            self.network.threshold_offset.copy_(theta)
        self.pretraining_updates += 1
        return {'hidden_spike_rate': float(rates.mean()),
                'silent_hidden_fraction': float((rates == 0).float().mean()),
                'mean_threshold_offset': float(theta.mean()),
                'mean_abs_eligibility': float(eligibility.abs().mean())}

    def save(self, path):
        torch.save({'format_version': 'adaptive_v1', 'feature_count': self.feature_count,
                    'feature_schema': self.feature_schema, 'modes': self.modes,
                    'variant': self.variant, 'settings': asdict(self.settings),
                    'network': self.network.state_dict(), 'q_updates': self.q_updates,
                    'pretraining_updates': self.pretraining_updates}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['format_version'] != 'adaptive_v1':
            raise ValueError('Unsupported adaptive checkpoint format')
        agent = cls(data['feature_count'], AdaptiveSNNSettings(**data['settings']),
                    data['modes'], data['feature_schema'], variant=data['variant'])
        agent.network.load_state_dict(data['network'])
        if not all(torch.isfinite(p).all() for p in agent.network.state_dict().values()):
            raise ValueError('Invalid adaptive network state')
        agent.q_updates = data['q_updates']
        agent.pretraining_updates = data['pretraining_updates']
        return agent
