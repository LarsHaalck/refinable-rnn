from .base import BaseNetwork

import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange
from typing import Tuple
from tod.model.type import ModelType


class RecurrentModule(BaseNetwork):

    def __init__(self, *, hidden_dim: int, num_layers: int = 4):
        super().__init__()

        self.rec = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.mlp_gt_to_hidden = nn.Sequential(
            nn.Linear(2, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim * num_layers),
            Rearrange('b (h l) -> l b h', h=hidden_dim, l=num_layers)
        )

    def forward(self, src: Tensor, hn: Tensor):
        out, hn = self.rec(src, hn)

        regs = out
        # concat = torch.cat([enc_feat, out], dim=-1).mean(axis=1)
        # data_x, data_y = self.mlp_x(concat), self.mlp_y(concat)
        # regs = torch.stack([data_x, data_y], dim=1)

        return regs, hn

    def get_hidden(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        # contiguous is required for recurrent module
        h = self.mlp_gt_to_hidden(src).contiguous()
        return (h, torch.zeros_like(h))


class RecurrentNet(BaseNetwork):

    def __init__(
        self, *, encoder, embedding, reprojector, projector, hidden_dim: int,
        model_type: ModelType
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.reprojector = reprojector
        self.projector = projector
        self.recurrent_mod = RecurrentModule(hidden_dim=hidden_dim)
        self.model_type = model_type

    def forward(self, src: Tensor, hn: Tensor):
        enc_feat = self.encoder(src)
        if self.model_type in [ModelType.HourGlass, ModelType.HourGlassSqueeze]:
            enc_feat = enc_feat[-1]

        emb_feat = self.embedding(enc_feat)
        curr_regs, hn = self.recurrent_mod(emb_feat.unsqueeze(1), hn)
        curr_regs = self.reprojector(curr_regs)

        curr_regs = enc_feat + curr_regs
        curr_regs = self.projector(curr_regs)
        return curr_regs, hn

    def get_hidden(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        return self.recurrent_mod.get_hidden(src)
