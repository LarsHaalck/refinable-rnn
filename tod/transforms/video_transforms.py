from tod.utils.misc import pair
import tod.utils.logger as logger

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np


####################################################
#             transform decorators
####################################################
# transforms the gt to [-1, 1]
class TransformGt(nn.Module):

    def __init__(self, crop):
        super().__init__()
        self.shape = np.array(pair(crop))

    def forward(self, data):
        data[-1] = (2 * (data[-1] / self.shape[::-1]) - 1).astype(np.float32)
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(shape={})'.format(self.shape.tolist())


# transforms the gt back from [-1, 1] to [0, h] x [0, w]
class InverseTransformGt(nn.Module):

    def __init__(self, image_shape):
        super().__init__()
        self.shape = np.array(pair(image_shape))

    def forward(self, data):
        data = (data + 1) / 2 * self.shape[-2:][::-1]
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(image_shape={0})'.format(self.shape.tolist())


class TransformGtClassification2x1d(nn.Module):

    def __init__(self, crop, *, kernel_size=31, kernel_sigma=1.):
        super().__init__()
        self.shape = np.array(pair(crop))
        self.gauss1d = self.gkern_1d(kernel_size, kernel_sigma)
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma

    @staticmethod
    def gkern_1d(w, sig):
        """
        Creates (not normalised) gaussian kernel with side length `l` and a sigma of `sig`
        Based on clemisch's answer on
        https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
        """
        ax = torch.linspace(-(w - 1) / 2., (w - 1) / 2., w)
        gauss = torch.exp(-0.5 * torch.square(ax) / sig**2)
        return gauss.unsqueeze(0).unsqueeze(0)

    def compute1d(self, gt):
        x = torch.zeros((gt.shape[0], 1, self.shape[0]), dtype=torch.float32)
        y = torch.zeros((gt.shape[0], 1, self.shape[1]), dtype=torch.float32)
        for i in range(gt.shape[0]):
            if gt[i, 0].astype(int) >= self.shape[0]:
                x[i, :, self.shape[0] - 1] = 1.
            else:
                x[i, :, gt[i, 0].astype(int)] = 1.
            if gt[i, 1].astype(int) >= self.shape[1]:
                y[i, :, self.shape[1] - 1] = 1.
            else:
                y[i, :, gt[i, 1].astype(int)] = 1.
        x = nn.functional.conv1d(x, self.gauss1d, padding="same")
        y = nn.functional.conv1d(y, self.gauss1d, padding="same")
        x, y = x[:, 0], y[:, 0]
        return torch.stack([x, y], dim=1)

    def forward(self, data):
        data[-1] = self.compute1d(data[-1])
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(shape={}, kernel_size={}, kernel_sigma={})'\
                .format(self.shape.tolist(), self.kernel_size, self.kernel_sigma)


class InverseTransformGtClassification2x1d(nn.Module):

    def __init__(self, image_shape):
        super().__init__()
        self.shape = np.array(pair(image_shape))

    def forward(self, data):
        data = torch.argmax(data, dim=-1)
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(image_shape={0})'.format(self.shape.tolist())


class TransformGtClassification2d(TransformGtClassification2x1d):

    def __init__(self, crop, *, kernel_size=31, kernel_sigma=1.):
        super().__init__(crop=crop, kernel_size=kernel_size, kernel_sigma=kernel_sigma)

    def forward(self, data):
        xy = self.compute1d(data[-1])
        data[-1] = torch.einsum("ab,ac->abc", xy[:, 0], xy[:, 1])
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(shape={}, kernel_size={}, kernel_sigma={})'\
                .format(self.shape.tolist(), self.kernel_size, self.kernel_sigma)


class InverseTransformGtClassification2d(nn.Module):

    def __init__(self, image_shape):
        super().__init__()
        self.shape = np.array(pair(image_shape))

    def forward(self, data):
        dim = data.dim()
        if dim == 4:
            b, t, _, _ = data.shape

        # argmax across last two dimensions
        data = (data == torch.amax(data, dim=(-2, -1), keepdims=True)).nonzero()
        # remove batch and temp indices
        if dim == 4:
            data = rearrange(data[:, -2:], "(b t) i -> b t i", b=b, t=t)
        else:
            data = data[:, 1:]
        # need to swap y,x to x,y
        return data[..., [1, 0]].contiguous()

    def __repr__(self):
        return self.__class__.__name__ + '(image_shape={0})'.format(self.shape.tolist())


####################################################
#                   reorder
####################################################
# rearrange temporal dimension to single big image
class ReorderHorizontal(nn.Module):

    def forward(self, data):
        a, b = data
        b[:, 0] += np.arange(a.shape[0]) * a.shape[-1]
        a = rearrange(a, 'b c h w -> c h (b w)')
        return [a, b]


####################################################
#             augmentation
####################################################
# Random drop image from a squence with p_drop
# for those droped set p_unary with:
# p_unary drop prob for unary, 1-p_unary drop prob for image part
# or
# None to drop both everytime
class RandomDrop(nn.Module):

    def __init__(self, p_drop=0.1, p_unary=0.5):
        super().__init__()
        self.p_drop = p_drop
        self.p_unary = p_unary

    def forward(self, data):
        choice = np.random.choice(
            (0, 1), size=data[0].shape[0], p=[1 - self.p_drop, self.p_drop]
        )
        idx = np.nonzero(choice)[0]

        if self.p_unary is not None:
            choice = np.random.choice(
                (0, 1), size=len(idx), p=[1 - self.p_unary, self.p_unary]
            )
            for i in idx:
                if choice[i] == 0:
                    data[0][i, :3] = 0  # drop image part
                else:
                    data[0][i, -1] = 0  # drop unary part
        else:
            data[0][idx] = 0  # drop everything

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(p_drop={}, p_unary={})'.format(
            self.p_drop, self.p_unary
        )


class RandomDropN(nn.Module):

    def __init__(self, n=1):
        super().__init__()
        self.n = n

    def forward(self, data):
        idx = np.random.choice(data[0].shape[0], size=self.n, replace=False)
        data[0][idx] = 0
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(n={0})'.format(self.n)


class DropList(nn.Module):

    def __init__(self, ids):
        super().__init__()
        self.ids = np.array(ids)

    def forward(self, data):
        data[0][self.ids] = 0
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(ids={0})'.format(self.ids.tolist())


class RandomNoise(nn.Module):

    def __init__(self, *, p=0.1, mean=0, sigma=0.01, unary_only=False, verbose=False):
        super().__init__()
        self.p = p
        self.mean = mean
        self.sigma = sigma
        self.verbose = verbose
        self.unary_only = unary_only
        self.log = logger.getLogger("RandomNoise")

    def _gaussian(self, shape):
        return np.random.normal(self.mean, self.sigma, shape[1:]).astype(np.float32)

    def forward(self, data):
        choice = np.random.choice((0, 1), size=data[0].shape[0], p=[1 - self.p, self.p])
        idx = np.nonzero(choice)[0]
        if len(idx) > 0:
            if self.verbose:
                self.log.info("Selecting ids: {}".format(idx))
            if self.unary_only:
                data[0][idx, -1] += self._gaussian(data[0][-1].shape)
            else:
                data[0][idx] += self._gaussian(data[0].shape)
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, mean={}, sigma={}, unary_only={})'.format(
            self.p, self.mean, self.sigma, self.unary_only
        )


class RandomRepeat(nn.Module):

    def __init__(self, *, n=10):
        super().__init__()
        self.n = n
        self.log = logger.getLogger("RandomRepeat")

    def forward(self, data):
        ids = np.arange(data[0].shape[0])
        repeats = np.random.choice(ids, self.n)
        repeats, counts = np.unique(repeats, return_counts=True)
        repeat = np.ones_like(ids)
        repeat[repeats] = counts

        # keep only len of data many repeats
        for i in range(len(data)):
            data[i] = np.repeat(data[i], repeat, axis=0)[:data[i].shape[0]]

        # remove unaries for multiple elements
        if data[0][0].shape[0] in [1, 4]:
            ids = np.repeat(ids, repeat)
            for i in range(len(data[0]) - 1):
                if ids[i + 1] == ids[i]:
                    data[0][i, -1] = 0
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(n={})'.format(self.n)


class RandomManualUnary(nn.Module):

    def __init__(self, *, p=0.1, sigma=5, crop):
        super().__init__()
        self.p = p
        self.sigma = 5
        self.crop = np.array(pair(crop))
        self.x = np.arange(self.crop[1])
        self.y = np.arange(self.crop[0])

    def forward(self, data):
        choice = np.random.choice((0, 1), size=data[0].shape[0], p=[1 - self.p, self.p])
        idx = np.nonzero(choice)[0]

        for id in idx:
            gt = data[-1][id]
            x0, y0 = gt[0], gt[1]
            gx = np.exp(-(self.x - x0)**2 / (2 * self.sigma**2))
            gy = np.exp(-(self.y - y0)**2 / (2 * self.sigma**2))
            g = np.outer(gx, gy)
            # g /= np.sum(g)
            # make tensor from numpy array and move to same device
            data[0][id, 0] = torch.from_numpy(g.T).to(data[0])
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, sigma={}, crop={})'.format(
            self.p, self.sigma, self.crop.tolist()
        )


class RandomRotate(nn.Module):

    def __init__(self, *, vert_only=False):
        super().__init__()
        self.vert_only = vert_only

    def _rot(self, gt, c, k):
        if k == 0:
            return gt
        elif k == 1:
            # swap axes
            gt_tmp = gt[:, [1, 0]]
            gt_tmp[:, 1] = c - gt_tmp[:, 1]
            return gt_tmp
        elif k == 2:
            return torch.tensor([c, c]) - gt
        else:
            # swap axes
            gt_tmp = gt[:, [1, 0]]
            gt_tmp[:, 0] = c - gt_tmp[:, 0]
            return gt_tmp

    def forward(self, data):
        assert data[0].shape[-2] == data[0].shape[-1], "Image must be square"
        # choices are k: number of 90 degree rotations
        # 0: 0 deg, 1: 90deg, 2: 180 deg, 3: 270 deg
        if self.vert_only:
            choice = np.random.choice((0, 1, 3))
        else:
            choice = np.random.choice((0, 1, 2, 3))

        # max img dimension
        c = data[0][0].shape[-1]

        # rotate data round last two (image spatial) dimensions
        # label stays the same
        # gt needs to be rotated
        data[0] = torch.rot90(data[0], choice, dims=[-2, -1])
        data[-1] = self._rot(data[-1], c, choice)

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(vert_only={})'.format(self.vert_only)


class RandomReverse(nn.Module):

    def __init__(self, *, p=0.5):
        super().__init__()
        self.p = p
        self.log = logger.getLogger("RandomReverse")

    def forward(self, data):
        flip = np.random.binomial(1, self.p)
        if flip:
            self.log.debug("Reversed Video")
            data[0] = torch.flip(data[0], [0])
            # np.array need to be copied to not have negative stride for data loader
            data[1] = np.flip(data[1], 0).copy()
            data[2] = np.flip(data[2], 0).copy()

            # if no unaries, return data as is
            if data[0].shape[1] not in [1, 4]:
                return data

            # otherwise iterate through unaries and use from timestamp before
            # iterate through timestamps
            for i in range(data[0].shape[0] - 1, 1, -1):
                data[0][i, -1] = data[0][i - 1, -1]

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
