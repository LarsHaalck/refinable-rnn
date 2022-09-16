import tod.utils.logger as logger
from tod.utils.misc import pair
from .base import BaseNetwork, BaseEmbedding
from .embedding import PixelEmbedding

import torch
from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Tuple


class Sim(torch.nn.Module):

    def __init__(self, emb_shape, gru_dim):
        super().__init__()
        hidden_dim = emb_shape[1]
        self.mlp_enc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp_gru = nn.Sequential(
            nn.Linear(gru_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.cossim = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, enc_rep, gru_rep):
        gru_rep = rearrange(gru_rep, 'l b c -> b l c')
        gru_rep = gru_rep[:, -1:, :]
        query = self.mlp_gru(gru_rep)
        key = self.mlp_enc(enc_rep)
        sim = self.cossim(query, key)
        sim = torch.softmax(sim, dim=-1).unsqueeze(-1)
        key = key * sim
        sim = torch.sum(key, dim=1, keepdim=True)
        return sim


class Recurrent(BaseNetwork):

    def __init__(self, *, image_size, embedding: BaseEmbedding, num_layers: int = 4):
        super().__init__()
        log = logger.getLogger("Recurrent")
        log.info("image_size: {}, embedding: {}".format(image_size, embedding))
        image_height, image_width = pair(image_size)

        self.embedding = embedding
        hidden_dim = embedding.get_vec_dim()

        # self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_dim))
        self.gru = nn.LSTM(
            input_size=embedding.get_vec_dim() if embedding else hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # self.mlp_reg = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim // 2),
        #     nn.LayerNorm(hidden_dim // 2),
        #     nn.Dropout(.25),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, hidden_dim // 4),
        #     nn.LayerNorm(hidden_dim // 4),
        #     nn.Dropout(.25),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 4, 2),
        # )

        self.mlp_x = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, 1024),
            nn.ReLU(inplace=True), nn.Linear(1024, 1024)
        )
        self.mlp_y = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, 1024),
            nn.ReLU(inplace=True), nn.Linear(1024, 1024)
        )

        self.mlp_gt_to_hidden = nn.Sequential(
            nn.Linear(2, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim * num_layers),
            Rearrange('b (h l) -> l b h', h=hidden_dim, l=num_layers)
        )

        self.mlp_class = nn.Linear(hidden_dim, 1)
        self.sim = None
        if isinstance(self.embedding, PixelEmbedding):
            self.sim = Sim(
                emb_shape=(embedding.get_num_vecs(), embedding.get_vec_dim()),
                gru_dim=hidden_dim
            )

    def forward(self, src: Tensor, h0: Tensor):
        pass

    def forward_single(self, src: Tensor, h0: Tensor):
        enc_feat = self.embedding(src)
        if self.sim is not None:
            enc_feat = self.sim(enc_feat, h0[0])
        out, hn = self.gru(enc_feat, h0)

        # concat
        # regs = self.mlp_reg(torch.cat([enc_feat, out], dim=-1).mean(axis=1))
        concat = torch.cat([enc_feat, out], dim=-1).mean(axis=1)
        data_x, data_y = self.mlp_x(concat), self.mlp_y(concat)
        regs = torch.stack([data_x, data_y], dim=1)
        # only enc
        # regs = self.mlp_reg(enc_feat).mean(axis=1)
        # hn = h0

        # only gru
        # regs = self.mlp_reg(out.mean(axis=1))

        # return regs, hn
        return regs, hn

    def get_hidden(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        # contiguous is required for GRU
        h = self.mlp_gt_to_hidden(src).contiguous()
        return (h, torch.zeros_like(h))
