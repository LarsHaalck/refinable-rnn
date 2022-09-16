# adapted from https://github.com/princeton-vl/CornerNet-Lite
import torch
import torch.nn as nn
from .hourglass_base import hg_module, hg, convolution, residual

HG_URL = "https://drive.google.com/u/0/uc?id=1e8At_iZWyXQgLlMwHkB83kN-AN85Uff1"\
    "&export=download&confirm=t"


def make_pool_layer(dim):
    return nn.Sequential()


def make_hg_layer(inp_dim, out_dim, modules):
    layers = [residual(inp_dim, out_dim, stride=2)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)


class HourGlass(nn.Module):

    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(3, 256, 256, with_bn=False), nn.Conv2d(256, dim, (1, 1))
        )

    def _merge_mod(self):
        return nn.Sequential(nn.Conv2d(256, 256, (1, 1), bias=False), nn.BatchNorm2d(256))

    def __init__(self, *, channels=3, pretrained=False):
        super().__init__()

        stacks = 2
        pre = nn.Sequential(
            convolution(7, 3, 128, stride=2), residual(128, 256, stride=2)
        )
        hg_mods = nn.ModuleList(
            [
                hg_module(
                    5, [256, 256, 384, 384, 384, 512], [2, 2, 2, 2, 2, 4],
                    make_pool_layer=make_pool_layer,
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
            # ckp = torch.load("/home/lars/Downloads/CornerNet_500000.pkl")
            ckp = torch.hub.load_state_dict_from_url(
                HG_URL, file_name="CornerNet_500000.pkl"
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
