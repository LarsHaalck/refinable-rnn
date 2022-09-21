from tod.model.base import PseudoNetwork
from tod.io.input_type import InputType
from tod.utils.misc import pair
from tod.model.encoder import get_encoder
from tod.model.embedding import GapEmbedding, Conv1dReshape
from tod.model.projector import ProjectorReg, Projector2dx4, Projector2dx8, \
        Projector2x1d, Reprojector
from tod.model.recurrent import RecurrentNet
from tod.model.type import ModelType
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


# shared (fixed) -> lstm -> projector (loaded + refined)
class ModelInterface():

    def __init__(
        self,
        *,
        type: ModelType,
        input_type: InputType,
        crop,
        kernel_size: int,
        kernel_sigma: float,
        freeze_encoder: bool,
        hg_across_spatial=True
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
        encoder_dim = encoder.get_hidden_dim()

        transform = None
        inv_transform = None
        projector = None
        loss = None
        embedding = None

        hidden_dim = 1024
        if self.type in [ModelType.ResnetReg, ModelType.ResnetClass]:
            embedding = GapEmbedding(encoder_dim=encoder_dim, feature_dim=hidden_dim)
            self.embedding = PseudoNetwork()
            self.reprojector = PseudoNetwork()
        else:

            if hg_across_spatial:
                self.embedding = Conv1dReshape(
                    encoder_dim=encoder_dim, feature_dim=hidden_dim
                )
                self.reprojector = Reprojector(
                    src_dim=hidden_dim, tgt_dim=[encoder_dim[0], 1, 1]
                )
            else:
                hidden_dim = 256
                self.embedding = GapEmbedding(
                    encoder_dim=hidden_dim, feature_dim=hidden_dim
                )
                self.reprojector = Reprojector(
                    src_dim=hidden_dim, tgt_dim=[1, encoder_dim[1], encoder_dim[2]]
                )

        if self.type == ModelType.ResnetReg:
            transform = TransformGt(crop)
            inv_transform = InverseTransformGt(crop)
            projector = ProjectorReg(hidden_dim=hidden_dim)
            loss = torch.nn.MSELoss()
        elif self.type == ModelType.ResnetClass:
            transform = TransformGtClassification2x1d(
                crop, kernel_size=kernel_size, kernel_sigma=kernel_sigma
            )
            inv_transform = InverseTransformGtClassification2x1d(crop)
            projector = Projector2x1d(hidden_dim=hidden_dim)
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
            if freeze_encoder:
                for param in embedding.parameters():
                    param.requires_grad = False
                embedding.eval()
            self.encoder = nn.Sequential(encoder, embedding)
        else:
            self.encoder = encoder

        self.recurrent = RecurrentNet(
            encoder=self.encoder,
            embedding=self.embedding,
            reprojector=self.reprojector,
            projector=self.projector,
            hidden_dim=hidden_dim,
            model_type=self.type
        )
