from tod.model.base import BaseNetwork
import torch
from torch import nn
from typing import List
import numpy as np


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
        data = data.squeeze(1)
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
            return self.upsample(data).squeeze(1)


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
            return self.upsample(data).squeeze(1)


class Reprojector(nn.Module):

    def __init__(self, *, src_dim: int, tgt_dim: List[int]):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim

        num_tgt = np.array(tgt_dim).prod()
        steps = np.linspace(src_dim, num_tgt, 4).astype(int)

        self.mlp = nn.Sequential(
            nn.LayerNorm(src_dim),
            nn.Linear(steps[0], steps[1]),
            nn.ReLU(inplace=True),
            nn.Linear(steps[1], steps[2]),
            nn.ReLU(inplace=True),
            nn.Linear(steps[2], steps[3]),
        )

    def forward(self, data):
        b = data.shape[0]
        data = self.mlp(data)
        return data.view(b, *self.tgt_dim)

    def __repr__(self):
        return self.__class__.__name__
