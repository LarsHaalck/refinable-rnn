from tod.io.input_type import InputType
from tod.utils.misc import pair
from tod.model.encoder import get_encoder
from tod.model.embedding import GapEmbedding, Conv1dReshape
from tod.model.projector import ProjectorReg, Projector2dx4, Projector2dx8, Projector2x1d
from tod.model.type import ModelType, ModelMode
import tod.utils.logger as logger

from tod.transforms.video_transforms import \
    TransformGt, InverseTransformGt, \
    TransformGtClassification2x1d, InverseTransformGtClassification2x1d, \
    TransformGtClassification2d, InverseTransformGtClassification2d

import torch
from torch import nn
from tod.utils.misc import _neg_loss2d, _neg_loss2x1d
import pathlib
import sys

log = logger.getLogger("Definition")


def load_model_config(load_path: str, map_location=None):
    checkpoint = None
    if len(load_path) > 0:
        if pathlib.Path(load_path).exists():
            log.info("Loading checkpoint from {}".format(load_path))
            checkpoint = torch.load(load_path, map_location=map_location)
        else:
            log.warn("Loading path {} does not exist".format(load_path))
            sys.exit(-1)
    return checkpoint


class ModelInterface():

    def __init__(
        self, *, type: ModelType, input_type: InputType, crop, kernel_size: int,
        kernel_sigma: float, freeze_encoder: bool, model_mode: ModelMode
    ):
        channels = None
        if input_type == InputType.Unaries:
            channels = 1
        elif input_type == InputType.Images:
            channels = 3
        elif input_type == InputType.ImagesUnaries:
            channels = 4

        self.channels = channels
        self.shape = pair(crop)
        self.type = type

        encoder = get_encoder(
            image_size=self.shape,
            channels=self.channels,
            freeze=freeze_encoder,
            type=type,
            pretrained=True
        )
        hidden_dim = encoder.get_hidden_dim()

        transform = None
        inv_transform = None
        projector = None
        loss = None
        embedding = None

        if self.type in [ModelType.ResnetReg, ModelType.ResnetClass]:
            embedding = GapEmbedding(encoder_dim=hidden_dim, feature_dim=1024)
        else:
            # embedding = GapEmbedding(encoder_dim=hidden_dim, feature_dim=256)
            embedding = Conv1dReshape(encoder_dim=hidden_dim, feature_dim=1024)

        self.embedding = embedding

        if self.type == ModelType.ResnetReg:
            transform = TransformGt(crop)
            inv_transform = InverseTransformGt(crop)
            projector = ProjectorReg(hidden_dim=1024)
            loss = torch.nn.MSELoss()
        elif self.type == ModelType.ResnetClass:
            transform = TransformGtClassification2x1d(
                crop, kernel_size=kernel_size, kernel_sigma=kernel_sigma
            )
            inv_transform = InverseTransformGtClassification2x1d(crop)
            projector = Projector2x1d(hidden_dim=1024)
            loss = _neg_loss2x1d
        elif self.type == ModelType.HourGlass:
            transform = TransformGtClassification2d(
                crop, kernel_size=kernel_size, kernel_sigma=kernel_sigma
            )
            inv_transform = InverseTransformGtClassification2d(crop)
            projector = Projector2dx4()
            loss = _neg_loss2d
        elif self.type == ModelType.HourGlassSqueeze:
            transform = TransformGtClassification2d(
                crop, kernel_size=kernel_size, kernel_sigma=kernel_sigma
            )
            inv_transform = InverseTransformGtClassification2d(crop)
            projector = Projector2dx8()
            loss = _neg_loss2d

        self.transform = transform
        self.inv_transform = inv_transform
        self.projector = projector
        self.loss = loss

        # for resnet types the embedding is part of the encoder
        if self.type in [ModelType.ResnetClass, ModelType.ResnetReg]:
            self.encoder = nn.Sequential(encoder, embedding)
        else:
            self.encoder = encoder

        # resnet types:
        # shared: data -> encoder -> embedding -> hidden
        # lstm: hidden -> lstm -> add on hidden
        # projector: hidden -> output

        # hg types:
        # shared: data -> encoder -> hidden
        # lstm: hidden -> embedding -> lstm -> add on hidden
        # projector: hidden -> output

        # single:
        # shared -> projector

        # recurrent
        # shared (fixed) -> lstm -> projector (loaded + refined)
