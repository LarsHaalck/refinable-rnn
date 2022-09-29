# model
from tod.model.encoder import Type as EncType
from tod.model.embedding import Type as EmbType
# data
from tod.io import VideoDataset, get_folders_from_fold_file, InputType
from tod.transforms import TransformGt, TransformGtClassification2d, RandomRotate, RandomDrop, InverseTransformGtClassification2d
# utils
import tod.utils.logger as logger
from tod.utils.device import getDevice
from tod.utils.vis import plot_points, plot_loss, show_single_item
from tod.utils.misc import _neg_loss2d
# misc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from model_def import load_model_config, Type as ModelType, get_model_from_dataset

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import pathlib
import sys

device = getDevice()

########################################################
# loader settings
########################################################
# {{{ num_worker/prefetch
num_workers = 2
prefetch_factor = 2
# }}}

########################################################
# training settings
########################################################
# {{{ params
batch_size = 1
lr = 1e-4
epochs = 5000
patience = 100
crop = 1024
kernel_size, kernel_sigma = 31, 1.
load_path = ""
logger.LOG_LEVEL = logger.INFO
log = logger.getLogger("Train")
input_type = InputType.ImagesUnaries
model_type = ModelType.HourGlassSqueeze
# }}}

########################################################
# datasets
########################################################
# {{{ datasets
# {{{ data transforms and folders
log.info("Training with device: {}".format(device))
length = 1
transform = nn.Sequential(
    # RandomNoise(p=1, mean=0, sigma=0.02, unary_only=True),
    RandomDrop(p_drop=0.2, p_unary=0.5),
    RandomRotate(vert_only=True),
    TransformGtClassification2d(crop, kernel_size=kernel_size, kernel_sigma=kernel_sigma)
)
inv_transform = InverseTransformGtClassification2d(crop)

test_set = VideoDataset(
    folders="/data/ant-ml/Ant13R4",
    config={
        "crop_size": crop,
        "input_type": InputType.ImagesUnaries,
        "video_length": length,
        "disjoint": True
    },
    transform=TransformGtClassification2d(
        crop, kernel_size=kernel_size, kernel_sigma=kernel_sigma
    )
)
log.info("Test Dataset: {}".format(test_set))
# }}}

# {{{ data selection
# for overfitting:
# train_set = torch.utils.data.Subset(dataset, np.arange(32))
# test_set = torch.utils.data.Subset(dataset, np.arange(32) + 32)

# data_length = len(dataset)
# train_length = int(0.9 * data_length)
# test_length = data_length - train_length
# train_set, test_set = random_split(dataset, (train_length, test_length))
# }}}

# {{{ data loading
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor
)

########################################################
# checkpoint loading
########################################################
checkpoint, model_config = load_model_config(load_path)

########################################################
# model and optimizer
########################################################
# {{{ model, optim
model_config = model_config or {
    "type": ModelType.SingleNet,
    "enc_type": EncType.Resnet50,
    "enc_pretrained": True,
    "emb_type": EmbType.Gap
}

model, _ = get_model_from_dataset(test_set, model_config)
model = model.to(device)
model.print_network(verbose=False)
stats = summary(model.encoder, (1, test_set.num_channels(), crop, crop), verbose=0)
log.info(str(stats))
# crit = torch.nn.MSELoss()
# crit = torch.nn.CrossEntropyLoss()
crit = _neg_loss2d
log.info("Criterion: {}".format(crit))
# }}}

########################################################
# checkpoint loading
########################################################
# {{{ model, optimizer loadint
start_epoch = 0
if checkpoint is not None:
    log.info("Loading model from checkpoint")
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint["epoch"]
# }}}

########################################################
# training loop
########################################################
# {{{ train loop
# best val loss and id
best_val_loss = np.inf
best_epoch = 0
epochs += start_epoch

for epoch in range(start_epoch, epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    # {{{ eval
    with torch.no_grad():
        model.eval()
        pos0 = torch.tensor([])
        pos1 = torch.tensor([])
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label, gt in test_loader:
            data = data.to(device)
            label = label.to(device)
            gt = gt.to(device)

            regs = model(data)
            loss = crit(regs, torch.squeeze(gt, dim=1))

            regs_point = inv_transform(regs[-1]).to("cpu")
            gt = inv_transform(gt).to("cpu").view(-1, 2)
            regs = regs[-1].to("cpu")
            regs_img = regs[0]
            # regs_img = torch.outer(regs[0, 1], regs[0, 0])

            fig2, ax2 = plt.subplots(1, 4)
            data = data.to("cpu")
            ax2[0].imshow(regs_img)
            ax2[0].scatter(gt[0, 0], gt[0, 1], color='red', marker='x', label="gt")
            ax2[0].scatter(
                regs_point[0, 0],
                regs_point[0, 1],
                color='yellow',
                marker='x',
                label="pred"
            )
            ax2[1].imshow(data[0, 0, :3].permute(1, 2, 0))
            ax2[1].scatter(gt[0, 0], gt[0, 1], color='red', marker='x', label="gt")
            ax2[1].scatter(
                regs_point[0, 0],
                regs_point[0, 1],
                color='yellow',
                marker='x',
                label="pred"
            )
            ax2[2].imshow(data[0, 0, -1])
            ax2[2].scatter(gt[0, 0], gt[0, 1], color='red', marker='x', label="gt")
            ax2[2].scatter(
                regs_point[0, 0],
                regs_point[0, 1],
                color='yellow',
                marker='x',
                label="pred"
            )
            ax2[3].imshow(
                data[0, 0, :3].permute(1, 2, 0) * (
                    (regs_img - torch.min(regs_img)) /
                    (torch.max(regs_img) - torch.min(regs_img))
                ).unsqueeze(-1)
            )
            ax2[3].scatter(gt[0, 0], gt[0, 1], color='red', marker='x', label="gt")
            ax2[3].scatter(
                regs_point[0, 0],
                regs_point[0, 1],
                color='yellow',
                marker='x',
                label="pred"
            )
            plt.show()
