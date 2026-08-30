from __future__ import annotations

import torch
from torch import nn


def initialise_linear_layers(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class Actor(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_sizes[1], action_size),
            nn.Tanh(),
        )
        initialise_linear_layers(self)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class TwinCritic(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        input_size = observation_size + action_size
        self.q1 = self._build_head(input_size, hidden_sizes)
        self.q2 = self._build_head(input_size, hidden_sizes)
        initialise_linear_layers(self)

    @staticmethod
    def _build_head(
        input_size: int,
        hidden_sizes: tuple[int, int],
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((observation, action), dim=-1)
        return self.q1(inputs), self.q2(inputs)

    def first(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.q1(torch.cat((observation, action), dim=-1))
