from tod.model.base import BaseEncoder
from tod.utils.misc import pair, get_pseudo_dim
import tod.utils.logger as logger

from torch import Tensor, nn
import torch
from tod.model.hourglass import HourGlass as HourGlassNet
from tod.model.hourglass_squeeze import HourGlassSqueeze as HourGlassSqueezeNet
from torchvision.models import resnet50
from tod.model.type import ModelType


def get_encoder(
    *,
    image_size,
    channels: int = 3,
    freeze: bool = False,
    type: ModelType,
    pretrained: bool = False
):
    return Encoder(
        image_size=image_size,
        channels=channels,
        freeze=freeze,
        type=type,
        pretrained=pretrained
    )


class Encoder(BaseEncoder):

    def __init__(
        self,
        *,
        image_size,
        channels=3,
        type: ModelType,
        pretrained: bool = False,
        freeze: bool = False
    ):
        super().__init__()
        log = logger.getLogger("Encoder")
        self.image_height, self.image_width = pair(image_size)
        self.type = type
        self.channels = channels
        self.freeze = freeze
        log.info(self)

        net = nn.Sequential()
        if type in [ModelType.ResnetReg, ModelType.ResnetClass]:
            net = resnet50(pretrained=pretrained)
            net.conv1 = torch.nn.Conv2d(
                channels, 64, (7, 7), (2, 2), padding=(7 // 2, 7 // 2)
            )
            layerlist = list(net.children())[:-2]
            net = torch.nn.Sequential(*layerlist)

        if type == ModelType.HourGlass:
            net = HourGlassNet(channels=channels, pretrained=pretrained)
        elif type == ModelType.HourGlassSqueeze:
            net = HourGlassSqueezeNet(channels=channels, pretrained=pretrained)

        # set requires_grad to false, and set mode to eval on freeze
        if freeze:
            for param in net.parameters():
                param.requires_grad = False
            net.eval()

        self.encoder = net
        self.hidden_dim = get_pseudo_dim(
            [1, channels, self.image_height, self.image_width], self.encoder
        )[1:]
        log.debug("hidden_dim: {}".format(self.hidden_dim))

    def get_hidden_dim(self) -> Tensor:
        return self.hidden_dim

    def forward(self, data) -> Tensor:
        b, t, c, h, w = data.shape
        data = data.view(b*t, c, h, w)
        return self.encoder(data)

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
                '(image_size={}, channels={}, freeze={}, type={})'.format(
                        (self.image_height, self.image_width), self.channels,
                        self.freeze, self.type)
