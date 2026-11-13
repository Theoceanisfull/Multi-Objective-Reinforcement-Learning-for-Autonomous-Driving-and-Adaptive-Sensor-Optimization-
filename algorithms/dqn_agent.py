"""Small DQN and Dyna-DQN agents using the Expected SARSA observation encoder."""
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from algorithms.sensor_agent import FEATURE_SCHEMA, ROUTE_FEATURE_SCHEMA, DRIVE_NAMES, SENSING_MODES
from experiments.notebook_helpers import sensor_mask


@dataclass
class DQNSettings:
    episodes: int = 60
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    hidden_size: int = 64
    batch_size: int = 32
    replay_capacity: int = 10_000
    learning_starts: int = 64  # Collect real transitions before updating Q.
    target_update_steps: int = 100  # Count Q updates, including planning updates.
    planning_steps: int = 2  # Used only by Dyna-DQN.
    model_learning_starts: int = 200  # Collect more data before trusting predictions.

    def validate(self):
        counts = (self.episodes, self.hidden_size, self.batch_size,
                  self.replay_capacity, self.learning_starts, self.target_update_steps,
                  self.model_learning_starts)
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in counts):
            raise ValueError("Episode, network, buffer, and update sizes must be positive integers")
        if not isinstance(self.planning_steps, int) or isinstance(self.planning_steps, bool) or self.planning_steps < 0:
            raise ValueError("planning_steps must be a nonnegative integer")
        if self.replay_capacity < max(self.batch_size, self.learning_starts, self.model_learning_starts):
            raise ValueError("Replay capacity must hold a batch and both warm-up thresholds")
        if not (0 < self.learning_rate < float('inf') and 0 <= self.gamma < 1
                and 0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError("Invalid learning rate, discount, or exploration settings")


def make_network(inputs, outputs, hidden_size):
    """One hidden layer keeps the network small and easy to inspect."""
    return nn.Sequential(nn.Linear(inputs, hidden_size), nn.ReLU(), nn.Linear(hidden_size, outputs))


class ReplayBuffer:
    """Store only real experience; discard the oldest item when full."""
    def __init__(self, capacity):
        self.items = deque(maxlen=capacity)

    def __len__(self):
        return len(self.items)

    def add(self, features, action, reward, next_features, terminal):
        self.items.append((np.array(features, dtype=np.float32, copy=True), int(action),
                           float(reward), np.array(next_features, dtype=np.float32, copy=True),
                           float(terminal)))

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self.items), size=batch_size, replace=False)
        columns = zip(*(self.items[int(i)] for i in indices))
        states, actions, rewards, next_states, terminals = [np.asarray(c) for c in columns]
        return (torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32), torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(terminals, dtype=torch.float32))


class DQNAgent:
    """Learn Q values from real replay batches with a separate target network."""
    def __init__(self, feature_count, settings=None, modes=SENSING_MODES,
                 feature_schema=FEATURE_SCHEMA, seed=42):
        self.settings = DQNSettings(**asdict(settings or DQNSettings()))
        self.settings.validate()
        if feature_count <= 0 or not modes or any(mode not in SENSING_MODES for mode in modes):
            raise ValueError("Supply a positive feature count and supported sensing modes")
        self.feature_count = feature_count
        self.feature_schema = feature_schema
        self.modes = tuple(modes)
        self.actions = [(drive, mode) for mode in self.modes for drive in DRIVE_NAMES]
        self.rng = np.random.default_rng(seed)
        # Local seeding avoids changing another agent's PyTorch random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online = make_network(feature_count, len(self.actions), self.settings.hidden_size)
            self.target = make_network(feature_count, len(self.actions), self.settings.hidden_size)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.settings.learning_rate)
        self.replay = ReplayBuffer(self.settings.replay_capacity)
        self.q_updates = 0

    def values(self, features):
        with torch.no_grad():
            return self.online(torch.as_tensor(features, dtype=torch.float32)).numpy()

    def select(self, features, epsilon, rng):
        if rng.random() < epsilon:
            return int(rng.integers(len(self.actions)))
        values = self.values(features)
        best = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-12))
        return int(rng.choice(best))

    def decode(self, action, env):
        drive_name, mode = self.actions[action]
        drive_id = next(i for i, name in env.backend.action_type.actions.items() if name == drive_name)
        return {"drive": drive_id, "sensors": sensor_mask(env.config, mode)}

    def td_targets(self, rewards, next_states, terminals):
        """Terminal targets contain the reward only; other targets include future value."""
        with torch.no_grad():
            best_next = self.target(next_states).max(dim=1).values
            return rewards + self.settings.gamma * (1 - terminals) * best_next

    def learn_batch(self, batch):
        states, actions, rewards, next_states, terminals = batch
        chosen_q = self.online(states).gather(1, actions[:, None]).squeeze(1)
        targets = self.td_targets(rewards, next_states, terminals)
        loss = F.smooth_l1_loss(chosen_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.q_updates += 1
        if self.q_updates % self.settings.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def observe(self, features, action, reward, next_features, terminal):
        """Save one real step and, after warm-up, take one Q-learning update."""
        self.replay.add(features, action, reward, next_features, terminal)
        if len(self.replay) < max(self.settings.batch_size, self.settings.learning_starts):
            return {}
        batch = self.replay.sample(self.settings.batch_size, self.rng)
        return {"q_loss": self.learn_batch(batch)}

    def save(self, path):
        """Save a policy for evaluation, not an exact training-resume checkpoint."""
        torch.save({"kind": type(self).__name__, "feature_count": self.feature_count,
                    "feature_schema": self.feature_schema, "modes": self.modes,
                    "settings": asdict(self.settings), "online": self.online.state_dict()}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["feature_schema"] not in {FEATURE_SCHEMA, ROUTE_FEATURE_SCHEMA}:
            raise ValueError("Incompatible feature schema")
        classes = {"DQNAgent": DQNAgent, "DynaDQNAgent": DynaDQNAgent}
        agent_class = classes[data["kind"]]
        agent = agent_class(data["feature_count"], DQNSettings(**data["settings"]),
                            data["modes"], data["feature_schema"])
        agent.online.load_state_dict(data["online"])
        if not all(torch.isfinite(p).all() for p in agent.online.parameters()):
            raise ValueError("Invalid network weights")
        agent.target.load_state_dict(agent.online.state_dict())
        return agent


class DynaDQNAgent(DQNAgent):
    """DQN plus a learned one-step feature/reward/terminal prediction model."""
    def __init__(self, feature_count, settings=None, modes=SENSING_MODES,
                 feature_schema=FEATURE_SCHEMA, seed=42):
        super().__init__(feature_count, settings, modes, feature_schema, seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1)
            # Input: current features + one-hot action. Output: feature change, reward, ending logit.
            self.model = make_network(feature_count + len(self.actions), feature_count + 2,
                                      self.settings.hidden_size)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings.learning_rate)

    def model_predictions(self, states, actions):
        action_flags = F.one_hot(actions, num_classes=len(self.actions)).float()
        prediction = self.model(torch.cat([states, action_flags], dim=1))
        return prediction[:, :self.feature_count], prediction[:, -2], prediction[:, -1]

    def learn_model(self, batch):
        states, actions, rewards, next_states, terminals = batch
        changes, predicted_rewards, terminal_logits = self.model_predictions(states, actions)
        loss = (F.smooth_l1_loss(changes, next_states - states)
                + F.smooth_l1_loss(predicted_rewards, rewards)
                + F.binary_cross_entropy_with_logits(terminal_logits, terminals))
        self.model_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.model_optimizer.step()
        return float(loss.detach())

    def imagine_batch(self, real_batch):
        """Start from real state/action pairs and predict just one step forward."""
        states, actions, _, _, _ = real_batch
        with torch.no_grad():
            changes, rewards, terminal_logits = self.model_predictions(states, actions)
            next_states = (states + changes).clamp(-1, 1)
            next_states[:, 0] = 1.0  # The encoder's first feature is always a constant bias.
            terminal_probabilities = terminal_logits.sigmoid()
        # A soft terminal probability weights future value by predicted survival probability.
        return states, actions, rewards, next_states, terminal_probabilities

    def observe(self, features, action, reward, next_features, terminal):
        metrics = super().observe(features, action, reward, next_features, terminal)
        if not metrics:
            return metrics
        batch = self.replay.sample(self.settings.batch_size, self.rng)
        metrics["model_loss"] = self.learn_model(batch)
        if len(self.replay) >= self.settings.model_learning_starts:
            losses = []
            for _ in range(self.settings.planning_steps):
                real_batch = self.replay.sample(self.settings.batch_size, self.rng)
                losses.append(self.learn_batch(self.imagine_batch(real_batch)))
            if losses:
                metrics["planning_loss"] = float(np.mean(losses))
        return metrics
