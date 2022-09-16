import numpy as np
from abc import ABC
import random
from tod.utils.misc import bounding_box, pair


class Sampler(ABC):
    def time_steps(self) -> int:
        return -1

    def __repr__(self):
        return self.__class__.__name__


# helper function to sample based on list of ids
def _sample_ids(data, ids):
    for i in range(len(data) - 1):
        if len(data[i]) == 0:
            continue
        data[i] = [data[i][id] for id in ids]
    data[-1] = data[-1][ids]


# discards all ground-truth data except the center
# class KeepCenterGt(Sampler):
#     def __call__(self, data):
#         gt_len = len(data[-1][0])
#         data[-1] = np.expand_dims(data[-1][gt_len // 2], axis=0)
#         return data


# discards all data except center resulting in a single timestep
class SampleCenter(Sampler):
    def __call__(self, data):
        gt_len = len(data[-1][0])
        for i in range(len(data) - 1):
            if len(data[i]) == 0:
                continue
            data[i] = [data[i][gt_len // 2]]
        data[-1] = np.expand_dims(data[-1][gt_len // 2], axis=0)
        return data

    def time_steps(self):
        return 1


# sample neighborhood from input and gt
class SampleNeighbor(Sampler):
    def __init__(self, neighbors):
        self.ids = np.array(neighbors)

    def __call__(self, data):
        _sample_ids(data, self.ids)
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(neighbors={0})'.format(self.ids)

    def time_steps(self):
        return len(self.ids)


class SampleRandomNeighbor(Sampler):
    def __init__(self, n, m):
        self.n = n
        self.m = m

    def __call__(self, data):
        ids = np.random.choice(np.arange(min(len(data[0]), self.m)), size=self.n)
        ids = np.sort(ids)
        _sample_ids(data, ids)
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(n={}, m={})'.format(self.n, self.m)

    def time_steps(self):
        return self.n


class SampleNeighborFarNear(Sampler):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        ids = np.empty(11, dtype=int)
        ids[:6] = np.arange(0, 101, 20)  # 0, 20, 40, 60, 80, 100
        ids[6:] = np.random.choice(np.arange(41, 60), size=5)
        ids = np.sort(ids)
        _sample_ids(data, ids)
        return data

    def time_steps(self):
        return 11
