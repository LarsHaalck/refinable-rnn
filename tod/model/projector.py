from tod.model.base import BaseNetwork
import torch
from torch import nn


class ProjectorReg(BaseNetwork):

    def __init__(self, *, hidden_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, data):
        return self.mlp(data)


class Projector2x1d(BaseNetwork):

    def __init__(self, *, hidden_dim):

        super().__init__()

        self.mlp_x = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1024),
            nn.ReLU(inplace=True), nn.Linear(1024, 1024)
        )
        self.mlp_y = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1024),
            nn.ReLU(inplace=True), nn.Linear(1024, 1024)
        )

    def forward(self, data):
        data_x, data_y = self.mlp_x(data), self.mlp_y(data)
        return torch.stack([data_x, data_y], dim=1)


class Projector2dx4(BaseNetwork):

    def __init__(self):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 32, kernel_size=(2, 2), stride=(2, 2)), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, data):
        if isinstance(data, list):
            return [self.upsample(d).squeeze(1) for d in data]
        else:
            return data.squeeze(1)


class Projector2dx8(BaseNetwork):

    def __init__(self):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(2, 2), stride=(2, 2)), nn.ReLU(),
            nn.ConvTranspose2d(64, 16, kernel_size=(2, 2), stride=(2, 2)), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, data):
        if isinstance(data, list):
            return [self.upsample(d).squeeze(1) for d in data]
        else:
            return data.squeeze(1)
