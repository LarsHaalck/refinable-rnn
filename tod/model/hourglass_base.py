# adapted from https://github.com/princeton-vl/CornerNet-Lite
import torch
import torch.nn as nn


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


class upsample(nn.Module):
    def __init__(self, scale_factor):
        super(upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)


class merge(nn.Module):

    def forward(self, x, y):
        return x + y


class convolution(nn.Module):

    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(
            inp_dim,
            out_dim, (k, k),
            padding=(pad, pad),
            stride=(stride, stride),
            bias=not with_bn
        )
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class residual(nn.Module):

    def __init__(self, inp_dim, out_dim, k=3, stride=1):
        super(residual, self).__init__()
        p = (k - 1) // 2

        self.conv1 = nn.Conv2d(
            inp_dim, out_dim, (k, k), padding=(p, p), stride=(stride, stride), bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(p, p), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim,
                      (1, 1), stride=(stride,
                                      stride), bias=False), nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


class corner_pool(nn.Module):

    def __init__(self, dim, pool1, pool2):
        super(corner_pool, self).__init__()
        self._init_layers(dim, pool1, pool2)

    def _init_layers(self, dim, pool1, pool2):
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2


def _make_layer(inp_dim, out_dim, modules):
    layers = [residual(inp_dim, out_dim)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)


def _make_layer_revr(inp_dim, out_dim, modules):
    layers = [residual(inp_dim, inp_dim) for _ in range(modules - 1)]
    layers += [residual(inp_dim, out_dim)]
    return nn.Sequential(*layers)


def _make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)


def _make_unpool_layer(dim):
    return upsample(scale_factor=2)


def _make_merge_layer(dim):
    return merge()


class hg_module(nn.Module):

    def __init__(
        self,
        n,
        dims,
        modules,
        make_up_layer=_make_layer,
        make_pool_layer=_make_pool_layer,
        make_hg_layer=_make_layer,
        make_low_layer=_make_layer,
        make_hg_layer_revr=_make_layer_revr,
        make_unpool_layer=_make_unpool_layer,
        make_merge_layer=_make_merge_layer
    ):
        super(hg_module, self).__init__()

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.n = n
        self.up1 = make_up_layer(curr_dim, curr_dim, curr_mod)
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)
        self.low2 = hg_module(
            n - 1,
            dims[1:],
            modules[1:],
            make_up_layer=make_up_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
        self.up2 = make_unpool_layer(curr_dim)
        self.merg = make_merge_layer(curr_dim)

    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        merg = self.merg(up1, up2)
        return merg


class hg(nn.Module):

    def __init__(self, pre, hg_modules, cnvs, inters, cnvs_, inters_):
        super(hg, self).__init__()

        self.pre = pre
        self.hgs = hg_modules
        self.cnvs = cnvs

        self.inters = inters
        self.inters_ = inters_
        self.cnvs_ = cnvs_

    def forward(self, x):
        inter = self.pre(x)

        cnvs = []
        for ind, (hg_, cnv_) in enumerate(zip(self.hgs, self.cnvs)):
            hg = hg_(inter)
            cnv = cnv_(hg)
            cnvs.append(cnv)

            if ind < len(self.hgs) - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = nn.functional.relu_(inter)
                inter = self.inters[ind](inter)
        return cnvs


class saccade_module(nn.Module):

    def __init__(
        self,
        n,
        dims,
        modules,
        make_up_layer=_make_layer,
        make_pool_layer=_make_pool_layer,
        make_hg_layer=_make_layer,
        make_low_layer=_make_layer,
        make_hg_layer_revr=_make_layer_revr,
        make_unpool_layer=_make_unpool_layer,
        make_merge_layer=_make_merge_layer
    ):
        super(saccade_module, self).__init__()

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.n = n
        self.up1 = make_up_layer(curr_dim, curr_dim, curr_mod)
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)
        self.low2 = saccade_module(
            n - 1,
            dims[1:],
            modules[1:],
            make_up_layer=make_up_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
        self.up2 = make_unpool_layer(curr_dim)
        self.merg = make_merge_layer(curr_dim)

    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        if self.n > 1:
            low2, mergs = self.low2(low1)
        else:
            low2, mergs = self.low2(low1), []
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        merg = self.merg(up1, up2)
        mergs.append(merg)
        return merg, mergs
