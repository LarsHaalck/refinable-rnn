from tod.model.base import BaseEmbedding

from torch import nn
from einops.layers.torch import Rearrange


def _get_linear(ft_in, ft_out):
    if ft_out is not None and ft_in != ft_out:
        return nn.Linear(ft_in, ft_out)
    return nn.Identity()

class GapEmbedding(BaseEmbedding):

    def __init__(self, *, encoder_dim, feature_dim):
        super().__init__()
        ft_in = encoder_dim[0]

        embedding = None
        if ft_in != feature_dim:
            embedding = nn.Sequential(
                Rearrange('n d () () -> n d'),
                _get_linear(ft_in, feature_dim),
            )

        self.embedding = embedding
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.vec_dim = feature_dim

    def forward(self, data):
        emb = self.gap(data)

        if self.embedding is not None:
            emb = self.embedding(emb)

        return emb

    def get_vec_dim(self):
        return int(self.vec_dim)

    def __repr__(self):
        return self.__class__.__name__


class Conv1dReshape(BaseEmbedding):

    def __init__(self, *, encoder_dim, feature_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(256, 1, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(encoder_dim[1] * encoder_dim[2], feature_dim),
        )

        self.vec_dim = feature_dim

    def forward(self, data):
        emb = self.embedding(data)
        return emb

    def get_vec_dim(self):
        return int(self.vec_dim)

    def __repr__(self):
        return self.__class__.__name__
