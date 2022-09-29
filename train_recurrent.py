# model
from tod.model.definition import ModelInterface, load_model_config
from tod.model.type import ModelType
# data
from tod.io import VideoDataset, get_folders_from_fold_file, InputType
# utils
import tod.utils.logger as logger
from tod.utils.device import getDevice
from tod.utils.vis import plot_points, plot_loss, show_single_item
from tod.transforms import RandomDrop, RandomRotate
# misc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import pathlib
import sys
import argparse

paths = {
    InputType.Images: {
        ModelType.HourGlassSqueeze:
        "single_InputType.Images_ModelType.HourGlassSqueeze_C1024_G31_S3.0_LR0.0001-2022-09-19_17:27:28",
        ModelType.ResnetClass:
        "single_InputType.Images_ModelType.ResnetClass_C1024_G31_S3.0_LR0.0001-2022-09-19_17:28:32",
    },
    InputType.ImagesUnaries: {
        ModelType.HourGlass:
        "single_InputType.ImagesUnaries_ModelType.HourGlass_C1024_G31_S3.0_LR0.0001-2022-09-20_10:05:46",
        ModelType.HourGlassSqueeze:
        "single_InputType.ImagesUnaries_ModelType.HourGlassSqueeze_C1024_G31_S3.0_LR0.0001-2022-09-19_17:28:32",
        ModelType.ResnetClass:
        "single_InputType.ImagesUnaries_ModelType.ResnetClass_C1024_G31_S3.0_LR0.0001-2022-09-19_17:28:33",
        ModelType.ResnetReg:
        "single_InputType.ImagesUnaries_ModelType.ResnetReg_C1024_G31_S3.0_LR0.0001-2022-09-20_10:14:47",
    },
    InputType.Unaries: {
        ModelType.HourGlassSqueeze:
        "single_InputType.Unaries_ModelType.HourGlassSqueeze_C1024_G31_S3.0_LR0.0001-2022-09-19_17:28:32",
        ModelType.ResnetClass:
        "single_InputType.Unaries_ModelType.ResnetClass_C1024_G31_S3.0_LR0.0001-2022-09-19_17:28:32",
    }
}

device = getDevice()


def collect_single(*, data, gt=None):
    regs = []
    hn = recurrent.get_hidden((2. * gt) / crop - 1.) if gt is not None else None
    for i in range(data.shape[1]):
        img = data[:, i].to(device)
        curr_regs, hn = recurrent(img, hn)
        regs.append(curr_regs)
    return torch.stack(regs, dim=1)


########################################################
# loader settings
########################################################
# {{{ num_worker/prefetch
num_workers = 4
prefetch_factor = 4
# }}}

########################################################
# training settings
########################################################
# {{{ params
# batch_size = 4  # hg types
# batch_size = 16
lr = 1e-5
epochs = 5000
patience = 100
crop = 1024
kernel_size, kernel_sigma = 31, 3.
input_type = InputType.ImagesUnaries
model_type = ModelType.HourGlassSqueeze

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

# {{ load/save
store_path = (
    "/data/ant-ml-res/recurrent_{}_{}_C{}_G{}_S{}_LR{}-".
    format(input_type, model_type, crop, kernel_size, kernel_sigma, lr) +
    datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
)

# empty load_path means "do not load anything", None will fail
load_path = ""
load_path_enc = "/data/ant-ml-res/" + paths[input_type][model_type] + "/model.pt"

batch_size = 0
if model_type in [ModelType.ResnetClass, ModelType.ResnetReg]:
    batch_size = 16
else:
    batch_size = 4
# }}}

# {{{ model
model_interface = ModelInterface(
    type=model_type,
    input_type=input_type,
    crop=crop,
    kernel_size=kernel_size,  # only used by hourglass
    kernel_sigma=kernel_sigma,  # only used by hourglass
    freeze_encoder=True,
)
# }}}

# create logging dir and file
store_path = pathlib.Path(store_path)
store_path.mkdir(exist_ok=False)
logger.LOG_LEVEL = logger.INFO
logger.LOG_FILE = store_path / "log.txt"
log = logger.getLogger("Train - Recurrent")
# }}}

########################################################
# datasets
########################################################
# {{{ datasets
# {{{ data transforms and folders
log.info("Training with device: {}".format(device))
length = 32
data_transform = nn.Sequential(
    # RandomNoise(p=1, mean=0, sigma=0.02, unary_only=True),
    RandomDrop(p_drop=0.2, p_unary=None),
    RandomRotate(vert_only=True)
)
transform = model_interface.transform
inv_transform = model_interface.inv_transform

train_datasets, test_datasets = get_folders_from_fold_file(
    csv_file="/data/ant-ml/dataset_folds.csv", path_prefix="/data/ant-ml", test_fold=2
)
log.info("Train sets: {}".format(train_datasets))
log.info("Test sets: {}".format(test_datasets))
dataset = VideoDataset(
    folders=train_datasets,
    config={
        "crop_size": crop,
        "input_type": InputType.ImagesUnaries,
        "video_length": length,
        "disjoint": False
    },
    transform=data_transform,
)

test_set = VideoDataset(
    folders=test_datasets,
    config={
        "crop_size": crop,
        "input_type": InputType.ImagesUnaries,
        "video_length": length,
        "disjoint": True
    },
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
checkpoint_enc = load_model_config(load_path_enc)
checkpoint = load_model_config(load_path)

########################################################
# model and optimizer
########################################################
# {{{ model, optim
recurrent = model_interface.recurrent.to(device)

crit = model_interface.loss
optimizer = torch.optim.AdamW(recurrent.parameters(), lr=lr)
# }}}

########################################################
# checkpoint loading
########################################################
# {{{ model, optimizer loadint
if checkpoint_enc is not None:
    recurrent.encoder.load_state_dict(checkpoint_enc['encoder_state_dict'])
    recurrent.projector.load_state_dict(checkpoint_enc['projector_state_dict'])

start_epoch = 0
if checkpoint is not None:
    log.info("Loading model from checkpoint")
    recurrent.load_state_dict(checkpoint['recurrent_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint["epoch"]
# }}}
# reset lr to supplied value
optimizer.defaults["lr"] = lr
for g in optimizer.param_groups:
    g['lr'] = lr

log.info("Criterion: {}".format(crit))
log.info("Optimizer: {}".format(optimizer))

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
    recurrent.train()
    # {{{ train
    for data, label, gt in train_loader:
        tepoch.update(1)
        # do not copy data to device, as it could be very big
        label = label.to(device)
        gt = gt.to(device)

        regs = collect_single(
            data=data,
            gt=gt[:, 0, :],
        )

        gts = gt.shape
        # mimic shape and type of dataloader transform
        gt = transform([gt.view(-1, 2).to("cpu").numpy()])[0].to(device)
        gt = gt.view(gts[0], gts[1], *gt.shape[1:])

        loss = crit(regs, gt)
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
        recurrent.eval()
        pos0 = torch.tensor([])
        pos1 = torch.tensor([])
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        plotted = False
        for data, label, gt in test_loader:
            # do not copy data to device, as it could be very big
            label = label.to(device)
            gt = gt.to(device)

            regs = collect_single(
                data=data,
                gt=gt[:, 0, :],
            )

            gts = gt.shape
            # mimic shape and type of dataloader transform
            gt_trans = transform([gt.view(-1, 2).to("cpu").numpy()])[0].to(device)
            gt_trans = gt_trans.view(gts[0], gts[1], *gt_trans.shape[1:])
            loss = crit(regs, gt_trans)

            regs = inv_transform(regs).to("cpu").view(gts[0], gts[1], 2)
            gt = gt.to("cpu")

            pos0 = torch.cat((pos0, regs.view(-1, 2)))
            pos1 = torch.cat((pos1, gt.view(-1, 2)))
            epoch_val_loss += loss.item() / len(test_loader)
            if not plotted:
                z = 0
                fig = show_single_item(
                    (data[z].cpu().numpy(), gt[z].cpu().numpy()), [regs[z].cpu().numpy()],
                    show=False
                )
                plt.savefig(store_path / ("single_{:04d}.jpg").format(epoch))
                plt.close(fig)
                plotted = True

        # {{{ plot points
        if epoch % 1 == 0:
            fig = plt.figure(figsize=(20, 10))
            plot_points(net=pos0, gt=pos1, show=False)
            plt.savefig(store_path / ("points_{:04d}.jpg").format(epoch))
            plt.close(fig)
        # }}}

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
                    'recurrent_state_dict': recurrent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'val_loss': epoch_val_loss,
                }, store_path / "model.pt"
            )
        # }}}
    # }}}
# }}}
