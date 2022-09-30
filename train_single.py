# model
from tod.model.definition import ModelInterface, load_model_config
from tod.model.type import ModelType
# data
from tod.io import VideoDataset, get_folders_from_fold_file, InputType
# utils
import tod.utils.logger as logger
from tod.utils.device import getDevice
from tod.utils.vis import plot_loss, show_single_item
from tod.transforms import RandomDrop, RandomRotate
# misc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import pathlib
import sys
import argparse

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
# {{{ train params
batch_size = 12
lr = 1e-4
epochs = 5000
patience = 100
crop = 1024
kernel_size, kernel_sigma = 31, 3.
input_type = InputType.ImagesUnaries
model_type = ModelType.HourGlassSqueeze
# }}}

# {{{ argparse
parser = argparse.ArgumentParser(description='Some desc')
parser.add_argument(
    '-i',
    '--input',
    required=True,
    type=int,
    choices=range(0, 3),
    help='0: Unaries, 1: Images, 2: ImagesUnaries'
)
parser.add_argument(
    '-t',
    '--type',
    required=True,
    type=int,
    choices=range(0, 4),
    help='0: ResnetReg, 1: ResnetClass, 2: HGS, 3: HG'
)

args = parser.parse_args()
input_type = InputType(args.input)
model_type = ModelType(args.type)
# }}}

# {{{ load/save
store_path = (
    "/data/ant-ml-res/single_{}_{}_C{}_G{}_S{}_LR{}-".
    format(input_type, model_type, crop, kernel_size, kernel_sigma, lr) +
    datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
)
# empty load_path means "do not load anything"
load_path = ""
# }}}

# {{{ model
model_interface = ModelInterface(
    type=model_type,
    input_type=input_type,
    crop=crop,
    kernel_size=kernel_size,  # only used by hourglass
    kernel_sigma=kernel_sigma,  # only used by hourglass
    freeze_encoder=False,
)
# }}}

# create logging dir and file
store_path = pathlib.Path(store_path)
store_path.mkdir(exist_ok=False)
logger.LOG_LEVEL = logger.INFO
logger.LOG_FILE = store_path / "log.txt"
log = logger.getLogger("Train - Single")

########################################################
# datasets
########################################################
# {{{ datasets
# {{{ data transforms and folders
log.info("Training with device: {}".format(device))
length = 1
data_transform = nn.Sequential(
    # RandomNoise(p=1, mean=0, sigma=0.02, unary_only=True),
    RandomDrop(p_drop=0.2, p_unary=0.5),
    RandomRotate(vert_only=True),
    model_interface.transform
)
inv_transform = model_interface.inv_transform

train_datasets, test_datasets = get_folders_from_fold_file(
    csv_file="/data/ant-ml/dataset_folds.csv",
    path_prefix="/data/ant-ml",
    test_fold=2
)
log.info("Train sets: {}".format(train_datasets))
log.info("Test sets: {}".format(test_datasets))
dataset = VideoDataset(
    folders=train_datasets,
    config={
        "crop_size": crop,
        "input_type": input_type,
        "video_length": length,
        "disjoint": False
    },
    transform=data_transform,
)

test_set = VideoDataset(
    folders=test_datasets,
    config={
        "crop_size": crop,
        "input_type": input_type,
        "video_length": length,
        "disjoint": True
    },
    transform=model_interface.transform
)
log.info("Train Dataset: {}".format(dataset))
log.info("Test Dataset: {}".format(test_set))
# }}}

# {{{ data selection
# for overfitting:
# train_set = torch.utils.data.Subset(dataset, np.arange(32))
# test_set = torch.utils.data.Subset(dataset, np.arange(32) + 32)

train_set = dataset
# }}}

# {{{ data loading
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor
)

first_elem = train_set[0]
image_size = first_elem[0].shape[-2:]
log.info("Dataset element sizes {} {}".format(first_elem[0].shape, first_elem[1].shape))
log.info("Train size {} ({} batches)".format(len(train_set), len(train_loader)))
log.info("Test size {} ({} batches)".format(len(test_set), len(test_loader)))
# }}}
# }}}

########################################################
# checkpoint loading
########################################################
checkpoint = load_model_config(load_path)

########################################################
# model and optimizer
########################################################
# {{{ model, optim
encoder = model_interface.encoder.to(device)
projector = model_interface.projector.to(device)

stats = summary(encoder, (1, dataset.num_channels(), crop, crop), verbose=0)
log.info(str(stats))

crit = model_interface.loss
optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(projector.parameters()), lr=lr
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
log.info("Criterion: {}".format(crit))
# }}}

########################################################
# checkpoint loading
########################################################
# {{{ model, optimizer loadint
start_epoch = 0
if checkpoint is not None:
    log.info("Loading model from checkpoint")
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    projector.load_state_dict(checkpoint['projector_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint["epoch"]

# reset lr to supplied value
optimizer.defaults["lr"] = lr
for g in optimizer.param_groups:
    g['lr'] = lr
log.info("Optimizer: {}".format(optimizer))
# }}}

########################################################
# training loop
########################################################
# {{{ train loop
losses = []
val_losses = []

# best val loss and id
best_val_loss = np.inf
best_epoch = 0
epochs += start_epoch

for epoch in range(start_epoch, epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    tepoch = tqdm(total=len(train_loader), unit="batch")
    tepoch.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))
    encoder.train()
    projector.train()
    # {{{ train
    for data, label, gt in train_loader:
        tepoch.update(1)
        data = data.to(device)
        label = label.to(device)
        gt = gt.to(device)

        enc = encoder(data.squeeze(1))
        out = projector(enc)

        loss = crit(out, torch.squeeze(gt, dim=1))
        if torch.any(loss.isnan()):
            log.info("Loss is NaN!")
            sys.exit()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() / len(train_loader)
    # }}}

    # {{{ eval
    with torch.no_grad():
        encoder.eval()
        projector.eval()
        pos0 = torch.tensor([])
        pos1 = torch.tensor([])
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        plotted = False
        for data, label, gt in test_loader:
            data = data.to(device)
            label = label.to(device)
            gt = gt.to(device)

            enc = encoder(data.squeeze(1))
            out = projector(enc)
            loss = crit(out, torch.squeeze(gt, dim=1))

            # extract last item from hourglass stack
            if model_type in [ModelType.HourGlass, ModelType.HourGlassSqueeze]:
                out = out[-1]

            out = inv_transform(out).to("cpu")
            gt = inv_transform(gt).to("cpu")
            pos0 = torch.cat((pos0, out.view(-1, 2)))
            pos1 = torch.cat((pos1, gt.view(-1, 2)))

            epoch_val_loss += loss.item() / len(test_loader)
            if not plotted:
                z = 0
                fig = show_single_item(
                    (data[z].cpu().numpy(), gt[z].cpu().numpy()),
                    [out[z:z + 1].cpu().numpy()],
                    show=False
                )
                plt.savefig(store_path / ("single_{:04d}.jpg").format(epoch))
                plt.close(fig)
                plotted = True

        # {{{ plot points
        # if epoch % 1 == 0:
        #     fig = plt.figure(figsize=(20, 10))
        #     plot_points(net=pos0, gt=pos1, show=False)
        #     plt.savefig(store_path / ("points_{:04d}.jpg").format(epoch))
        #     plt.close(fig)
        # }}}

        scheduler.step(epoch_val_loss)
        tepoch.set_postfix_str(
            "train={:.4f}, val={:.4f}".format(epoch_loss, epoch_val_loss)
        )
        tepoch.close()

        # {{{ plot losses
        losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        fig = plt.figure(figsize=(20, 10))
        plot_loss(losses, val_losses, show=False)
        plt.savefig(store_path / "loss.jpg")
        np.savetxt(store_path / "loss.csv", np.c_[losses, val_losses])
        plt.close(fig)
        # }}}

        # {{{ early stopping
        # save better model if available
        if epoch - best_epoch > patience:
            log.info("Patience reached. Terminating traing")
            sys.exit(0)

        if epoch_val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = epoch_val_loss

            log.info(
                "Saving better model in epoch {} for val_loss={:4f}".format(
                    epoch + 1, epoch_val_loss
                )
            )
            torch.save(
                {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'projector_state_dict': projector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'val_loss': epoch_val_loss,
                }, store_path / "model.pt"
            )
        # }}}
    # }}}
# }}}
