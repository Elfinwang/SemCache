"""The threshold network g_theta: phi(d) -> t(d) in (0, 1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int = 16
    epochs: int = 400
    learning_rate: float = 0.01
    weight_decay: float = 1e-2
    seed: int = 0


class ThresholdNet(nn.Module):
    """One-hidden-layer MLP with a sigmoid output."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features)).squeeze(-1)


class ThresholdPredictor:
    """Standardises features, fits g_theta with MSE, predicts clipped thresholds."""

    def __init__(self, input_dim: int, config: TrainingConfig = TrainingConfig()):
        self.input_dim = input_dim
        self.config = config
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.model: ThresholdNet | None = None

    def _scale(self, features: np.ndarray) -> torch.Tensor:
        return torch.tensor((features - self.mean) / self.std, dtype=torch.float32)

    def fit(self, features: np.ndarray, thresholds: np.ndarray) -> list[float]:
        """Train from scratch and return the per-epoch loss."""
        self.mean = features.mean(axis=0)
        self.std = np.maximum(features.std(axis=0), 1e-6)
        torch.manual_seed(self.config.seed)
        self.model = ThresholdNet(self.input_dim, self.config.hidden_dim)
        inputs = self._scale(features)
        target = torch.tensor(thresholds, dtype=torch.float32)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()
        history: list[float] = []
        self.model.train()
        for _ in range(self.config.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.model(inputs), target)
            loss.backward()
            optimizer.step()
            history.append(float(loss.item()))
        return history

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ThresholdPredictor.fit must be called before predict.")
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(self._scale(features)).numpy()
        return np.clip(predicted, 0.0, 1.0)

    def leave_one_out(self, features: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Prediction for each training tuple from a model trained without it."""
        held_out = np.zeros(len(thresholds), dtype=np.float32)
        for index in range(len(thresholds)):
            mask = np.ones(len(thresholds), dtype=bool)
            mask[index] = False
            fold = ThresholdPredictor(self.input_dim, self.config)
            fold.fit(features[mask], thresholds[mask])
            held_out[index] = fold.predict(features[index : index + 1])[0]
        return held_out
