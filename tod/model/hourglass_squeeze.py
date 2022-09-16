# adapted from https://github.com/princeton-vl/CornerNet-Lite
import torch
import torch.nn as nn
from .hourglass_base import hg_module, hg, convolution, residual

HG_SQUEEZE_URL = "https://drive.google.com/u/0/uc?id=1qM8BBYCLUBcZx_UmLT0qMXNTh-Yshp4X"\
    "&export=download&confirm=t"


class fire_module(nn.Module):

    def __init__(self, inp_dim, out_dim, sr=2, stride=1):
        super(fire_module, self).__init__()
        self.conv1 = nn.Conv2d(
            inp_dim, out_dim // sr, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_dim // sr)
        self.conv_1x1 = nn.Conv2d(
            out_dim // sr, out_dim // 2, kernel_size=1, stride=stride, bias=False
        )
        self.conv_3x3 = nn.Conv2d(
            out_dim // sr,
            out_dim // 2,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=out_dim // sr,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.skip = (stride == 1 and inp_dim == out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        conv2 = torch.cat((self.conv_1x1(bn1), self.conv_3x3(bn1)), 1)
        bn2 = self.bn2(conv2)
        if self.skip:
            return self.relu(bn2 + x)
        else:
            return self.relu(bn2)


def make_pool_layer(dim):
    return nn.Sequential()


def make_unpool_layer(dim):
    return nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)


def make_layer(inp_dim, out_dim, modules):
    layers = [fire_module(inp_dim, out_dim)]
    layers += [fire_module(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)


def make_layer_revr(inp_dim, out_dim, modules):
    layers = [fire_module(inp_dim, inp_dim) for _ in range(modules - 1)]
    layers += [fire_module(inp_dim, out_dim)]
    return nn.Sequential(*layers)


def make_hg_layer(inp_dim, out_dim, modules):
    layers = [fire_module(inp_dim, out_dim, stride=2)]
    layers += [fire_module(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)


class HourGlassSqueeze(nn.Module):

    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(1, 256, 256, with_bn=False), nn.Conv2d(256, dim, (1, 1))
        )

    def _merge_mod(self):
        return nn.Sequential(nn.Conv2d(256, 256, (1, 1), bias=False), nn.BatchNorm2d(256))

    def __init__(self, *, channels=3, pretrained=False):
        super().__init__()

        stacks = 2
        pre = nn.Sequential(
            convolution(7, 3, 128, stride=2), residual(128, 256, stride=2),
            residual(256, 256, stride=2)
        )
        hg_mods = nn.ModuleList(
            [
                hg_module(
                    4, [256, 256, 384, 384, 512], [2, 2, 2, 2, 4],
                    make_pool_layer=make_pool_layer,
                    make_unpool_layer=make_unpool_layer,
                    make_up_layer=make_layer,
                    make_low_layer=make_layer,
                    make_hg_layer_revr=make_layer_revr,
                    make_hg_layer=make_hg_layer
                ) for _ in range(stacks)
            ]
        )
        cnvs = nn.ModuleList([convolution(3, 256, 256) for _ in range(stacks)])
        inters = nn.ModuleList([residual(256, 256) for _ in range(stacks - 1)])
        cnvs_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])
        inters_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])

        self.hgs = hg(pre, hg_mods, cnvs, inters, cnvs_, inters_)

        if pretrained:
            ckp = torch.hub.load_state_dict_from_url(
                HG_SQUEEZE_URL, file_name="CornerNet_Squeeze_500000.pkl"
            )
            ckp = {
                k[len("module.hg."):]: v
                for k, v in ckp.items() if k.startswith("module.hg.")
            }
            self.hgs.load_state_dict(ckp)

        # replace first convolution according to channels
        if channels != 3:
            old_conv = self.hgs.pre[0].conv
            self.hgs.pre[0].conv = nn.Conv2d(
                channels,
                128,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias
            )

    def forward(self, xs):
        xs = self.hgs(xs)
        return xs
