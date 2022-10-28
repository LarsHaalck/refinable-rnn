# model
from tod.model.definition import load_model_config, ModelType, ModelInterface
# data
from tod.io import VideoDataset, InputType
# utils
import tod.utils.logger as logger
from tod.utils.device import getDevice
# misc
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

import re
from tqdm import tqdm
from tod.utils.vis import plot_heatmap


def get_load_path_from_types(input_type, model_type):
    p1 = re.sub(r"InputType\.", "", str(input_type))
    p2 = re.sub(r"ModelType\.", "", str(model_type))
    p = "{}_{}".format(p1, p2)
    return "/data/ant-ml-res/single/{}/model.pt".format(p), p


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
crop = 1024
kernel_size, kernel_sigma = 31, 1.
logger.LOG_LEVEL = logger.INFO
log = logger.getLogger("Train")
input_type = InputType.ImagesUnaries
model_type = ModelType.HourGlassSqueeze

# empty load_path means "do not load anything"
load_path, key = get_load_path_from_types(input_type, model_type)
# }}}

# {{{ model
model_interface = ModelInterface(
    type=model_type,
    input_type=input_type,
    crop=crop,
    kernel_size=kernel_size,
    kernel_sigma=kernel_sigma,
    freeze_encoder=False,
)
# }}}

########################################################
# datasets
########################################################
# {{{ datasets
# {{{ data transforms and folders
log.info("Training with device: {}".format(device))
length = 1

test_set = VideoDataset(
    folders=["/data/eval_tod/Ant4R8", "/data/eval_tod/Ant6ZVF"],
    config={
        "crop_size": crop,
        "input_type": input_type,
        "video_length": length,
        "disjoint": True
    },
    transform=model_interface.transform
)

inv_transform = model_interface.inv_transform
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
checkpoint = load_model_config(load_path)

########################################################
# model and optimizer
########################################################
# {{{ model, optim
encoder = model_interface.encoder.to(device)
projector = model_interface.projector.to(device)
stats = summary(encoder, (1, test_set.num_channels(), crop, crop), verbose=0)
log.info(str(stats))
crit = torch.nn.MSELoss(reduction="none")
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
# }}}

########################################################
# loop
########################################################
# {{{ loop
encoder.eval()
projector.eval()

losses = torch.tensor([])
f = open(f'eval_single_{key}.csv', 'w')

with torch.no_grad():
    pos0 = torch.tensor([])
    pos1 = torch.tensor([])
    epoch_val_accuracy = 0
    epoch_val_loss = 0
    curr_it = 0
    for data, label, gt in tqdm(test_loader):
        data = data.to(device)
        label = label.to(device)
        gt = gt.to(device)

        enc = encoder(data.squeeze(1))
        out = projector(enc)

        gt = inv_transform(gt).to("cpu").view(-1, 2)

        if model_type == ModelType.ResnetClass:
            regs_point = [inv_transform(out).to("cpu")]
        elif model_type == ModelType.ResnetReg:
            regs_point = [inv_transform(out).to("cpu")]
        else:
            regs_point = [o.to("cpu") for o in inv_transform(out)]

        plot_heatmap(
            input_type=input_type,
            input=data,
            crop=crop,
            model_type=model_type,
            out=out,
            regs_point=regs_point,
            gt=gt,
            temp=10
        )
