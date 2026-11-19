"""
Actor and Critic networks for the DDPG underwater-restoration agent.

Matches the architecture described in the capstone review:
  - Actor  mu(s | theta):  image state -> continuous enhancement action
  - Critic Q(s, a | phi):  (state, action) -> expected cumulative reward

Both share a small convolutional state encoder (the "compact numeric
summary of the image" from the workflow slide) so the Actor and Critic
see the same representation of the scene.
"""
import torch
import torch.nn as nn

# Action vector, all continuous, tanh-bounded to [-1, 1] and rescaled
# in enhancement/ops.py to their physical ranges:
#   0: red gain        3: gamma
#   1: green gain       4: contrast
#   2: blue gain        5: saturation
#                        6: dehaze strength
ACTION_DIM = 7
ACTION_NAMES = [
    "red_gain", "green_gain", "blue_gain",
    "gamma", "contrast", "saturation", "dehaze_strength",
]


class StateEncoder(nn.Module):
    """Downsized image -> feature vector. Shared trunk for Actor and Critic."""

    def __init__(self, in_channels: int = 3, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(64 * 4 * 4, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = torch.flatten(x, 1)
        return torch.relu(self.fc(x))


class Actor(nn.Module):
    """The chef: picks the enhancement action for the current image state."""

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.encoder = StateEncoder(feat_dim=feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, ACTION_DIM), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(state))


class Critic(nn.Module):
    """The food critic: scores how good a (state, action) pair really was."""

    def __init__(self, feat_dim: int = 128, action_dim: int = ACTION_DIM):
        super().__init__()
        self.encoder = StateEncoder(feat_dim=feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + action_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(state)
        return self.head(torch.cat([feat, action], dim=1))
