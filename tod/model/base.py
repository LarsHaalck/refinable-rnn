import torch.nn as nn
import tod.utils.logger as logger

from abc import ABC, abstractmethod
from torch import Tensor


class BaseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

    def print_network(self, verbose=False):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()

        log = logger.getLogger("Base")
        if verbose:
            log.info(self)
        log.info(
            'Network [%s] was created. '
            'Total number of parameters: %.1f million. ' %
            (type(self).__name__, num_params / 1000000)
        )


class BaseEncoder(BaseNetwork, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_hidden_dim(self) -> Tensor:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class BaseEmbedding(BaseNetwork, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_vec_dim(self) -> int:
        pass


class PseudoNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, data):
        return data
