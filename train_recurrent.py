# model
from tod.model.encoder import Type as EncType
from tod.model.embedding import Type as EmbType
# data
from tod.io import VideoDataset, get_folders_from_fold_file, InputType
from tod.transforms import TransformGtClassification2d, RandomRotate, RandomDrop,\
    InverseTransformGtClassification2d
from tod.transforms import TransformGtClassification2x1d,\
        InverseTransformGtClassification2x1d
# utils
import tod.utils.logger as logger
from tod.utils.device import getDevice
from tod.utils.vis import plot_points, plot_loss, show_single_item
from tod.utils.misc import _neg_loss2x1d
# misc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from model_def import load_model_config, Type as ModelType, get_model

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import pathlib
import sys

device = getDevice()


def collect_single(*, data, model, encoder, device, gt=None, crop_size=1024.):
    regs = []
    hn = model.get_hidden((2. * gt) / crop_size - 1.) if gt is not None else None
    for i in range(data.shape[1]):
        img = data[:, i].to(device)
        enc_feat = encoder(img)[-1]

        curr_regs, hn = model.forward_single(enc_feat, hn)

        regs.append(curr_regs.unsqueeze(1))
    return torch.cat(regs, dim=1)


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
batch_size = 12
lr = 1e-5
epochs = 5000
patience = 100
crop = 1024
kernel_size, kernel_sigma = 31, 3.
store_path = (
    "/data/ant-ml-res/re-recurrent-k{}s{}lr{}".format(kernel_size, kernel_sigma, lr) +
    datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
)
# load_path = "/data/ant-ml-res/recurrent-2022-06-20_10:55:54_best/model.pt"
load_path = ""
load_path_enc = "/data/ant-ml-res/test_re_re_single_classification_negloss_G31S3.0LR0.0001-2022-07-29_22:50:44/model.pt"
store_path = pathlib.Path(store_path)
store_path.mkdir(exist_ok=False)
logger.LOG_LEVEL = logger.INFO
logger.LOG_FILE = store_path / "log.txt"
log = logger.getLogger("Train")
# }}}

########################################################
# datasets
########################################################
# {{{ datasets
# {{{ data transforms and folders
log.info("Training with device: {}".format(device))
length = 32
transform = nn.Sequential(
    # RandomNoise(p=1, mean=0, sigma=0.02, unary_only=True),
    RandomDrop(p_drop=0.2, p_unary=None),
    RandomRotate(vert_only=True)
)
inv_transform2d = InverseTransformGtClassification2d(crop)
transform2x1d = TransformGtClassification2x1d(
    crop, kernel_size=kernel_size, kernel_sigma=kernel_sigma
)
inv_transform2x1d = InverseTransformGtClassification2x1d(crop)

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
    transform=transform,
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

# data_length = len(dataset)
# train_length = int(0.9 * data_length)
# test_length = data_length - train_length
# train_set, test_set = random_split(dataset, (train_length, test_length))
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
    shuffle=True,
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
checkpoint_enc, model_config_enc = load_model_config(load_path_enc)
# checkpoint_enc, model_config_enc = load_model_config(
#     load_path_enc, map_location=torch.device("cpu")
# )
checkpoint, model_config = load_model_config(load_path)

########################################################
# model and optimizer
########################################################
# {{{ model, optim
model_config = model_config or {
    "channels": dataset.num_channels(),
    "time_steps": 1,  # this is important!
    "image_size": dataset.image_size(),
    "type": ModelType.Recurrent,
    "enc_type": EncType.Resnet50,
    "enc_pretrained": True,
    "emb_type": EmbType.Gap,
    "enc_freeze": True,
    "hidden_dim": 1024,
}

# check if model_config works with encoder model_config
if model_config_enc:
    assert model_config_enc["enc_type"] == model_config["enc_type"
                                                        ], "Encoder types mismatch"

model, encoder = get_model(model_config)
model = model.to(device)
encoder = encoder.to(device)
model.print_network(verbose=False)
stats = summary(encoder, (1, dataset.num_channels(), crop, crop), verbose=0)
log.info(str(stats))
crit = _neg_loss2x1d
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# }}}

########################################################
# checkpoint loading
########################################################
# {{{ model, optimizer loadint
if checkpoint_enc is not None:
    ckp_dict = checkpoint_enc["model_state_dict"]
    ckp_dict = {
        k[len("encoder."):]: v
        for k, v in ckp_dict.items() if k.startswith("encoder.")
    }
    encoder.load_state_dict(ckp_dict)
    # encoder.to(torch.device("cpu"))
start_epoch = 0
if checkpoint is not None:
    log.info("Loading model from checkpoint")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint["epoch"]
# }}}
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
    model.train()
    # {{{ train
    for data, label, gt in train_loader:
        tepoch.update(1)
        # do not copy data to device, as it could be very big
        label = label.to(device)
        gt = gt.to(device)

        regs = collect_single(
            data=data,
            model=model,
            encoder=encoder,
            device=device,
            gt=gt[:, 0, :],
            crop_size=crop
        )
        gts = gt.shape
        gt = transform2x1d([gt.view(-1, 2).to("cpu").numpy()]
                           )[0].view(gts[0], gts[1], 2, 1024).to(device)
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
        model.eval()
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
                model=model,
                encoder=encoder,
                device=device,
                gt=gt[:, 0, :],
                crop_size=crop
            )
            gts = gt.shape
            gt_trans = transform2x1d([gt.view(-1, 2).to("cpu").numpy()]
                                     )[0].view(gts[0], gts[1], 2, 1024).to(device)
            loss = crit(regs, gt_trans)

            regs = inv_transform2x1d(regs).to("cpu")
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
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'val_loss': epoch_val_loss,
                    'model_config': model_config
                }, store_path / "model.pt"
            )
        # }}}
    # }}}
# }}}
