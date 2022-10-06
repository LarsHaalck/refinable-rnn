# model
from tod.model.definition import ModelInterface, load_model_config

# data
from tod.io import VideoDataset, InputType
# utils
import tod.utils.logger as logger
from tod.utils.device import getDevice
# misc
import torch
import torch.nn as nn

import pathlib
import csv
from tqdm import tqdm
from scipy import interpolate as interp
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys

device = getDevice()


def show_results(pos_net_sgl, pos_net_fwd, pos_net_bi, pos_ltracker, pos_gt, grid):
    if torch.numel(pos_net_fwd) == 0:
        return

    pos_net_sgl = pos_net_sgl[start:curr_it]
    pos_net_fwd = pos_net_fwd[start:curr_it]
    pos_net_bi = pos_net_bi[start:curr_it]
    pos_ltracker = pos_ltracker[start:curr_it]
    pos_gt = pos_gt[start:curr_it]

    # if torch.any(gt == 1023):
    #     delind.append(i)
    # pos_net_sgl = np.delete(pos_net_sgl, delind, 0)
    # pos_net_fwd = np.delete(pos_net_fwd, delind, 0)
    # pos_net_bi = np.delete(pos_net_bi, delind, 0)
    # pos_gt = np.delete(pos_gt, delind, 0)
    # tracker_pred = np.delete(tracker_pred, delind, 0)

    pos_net_sgl = inv_transform(pos_net_sgl).numpy()
    pos_net_fwd = inv_transform(pos_net_fwd).numpy()
    pos_net_bi = inv_transform(pos_net_bi).numpy()
    pos_ltracker = inv_transform(pos_ltracker).numpy()
    pos_gt = inv_transform(pos_gt).numpy()
    time = np.arange(len(pos_gt))

    pos_interp = np.empty_like(pos_gt)
    pos_interp[:, 0] = interp.interp1d(
        time[grid], pos_gt[grid, 0], fill_value="extrapolate", kind="linear"
    )(time)
    pos_interp[:, 1] = interp.interp1d(
        time[grid], pos_gt[grid, 1], fill_value="extrapolate", kind="linear"
    )(time)

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(pos_net_sgl[:, 0], pos_net_sgl[:, 1], time, label="single")
    ax.plot(pos_net_fwd[:, 0], pos_net_fwd[:, 1], time, label="fwd")
    ax.plot(pos_net_bi[:, 0], pos_net_bi[:, 1], time, label="bi")
    ax.plot(pos_ltracker[:, 0], pos_ltracker[:, 1], time, label="ltracker")
    ax.plot(pos_gt[:, 0], pos_gt[:, 1], time, label="gt")
    ax.plot(pos_interp[:, 0], pos_interp[:, 1], time, label="interp")
    plt.legend()
    ax.set_xlim(0, crop)
    ax.set_ylim(0, crop)
    plt.show()

    dist_sgl = np.linalg.norm(pos_net_sgl - pos_gt, axis=1)
    dist_fwd = np.linalg.norm(pos_net_fwd - pos_gt, axis=1)
    dist_bi = np.linalg.norm(pos_net_bi - pos_gt, axis=1)
    dist_ltracker = np.linalg.norm(pos_ltracker - pos_gt, axis=1)
    dist_interp = np.linalg.norm(pos_interp - pos_gt, axis=1)
    ones = np.ones_like(dist_fwd)
    fig2, ax2 = plt.subplots()
    plt.scatter(0.8 * ones, dist_sgl, c=np.arange(len(dist_bi)), s=4)
    plt.scatter(1.8 * ones, dist_fwd, c=np.arange(len(dist_fwd)), s=4)
    plt.scatter(2.8 * ones, dist_bi, c=np.arange(len(dist_bi)), s=4)
    plt.scatter(3.8 * ones, dist_ltracker, c=np.arange(len(dist_bi)), s=4)
    plt.scatter(4.8 * ones, dist_interp, c=np.arange(len(dist_bi)), s=4)
    plt.violinplot([dist_sgl, dist_fwd, dist_bi, dist_ltracker, dist_interp])
    labels = ["single", "fwd", "bi", "ltracker", "interp"]
    ax2.set_xticks(np.arange(1, len(labels) + 1))
    ax2.set_xticklabels(labels)
    # ax2.set_xlim(0.25, len(labels) + 0.75)
    ax2.set_xlabel('Net architecture')
    plt.ylim(-100, crop)
    print(
        "med_sgl = [{}/{}]".format(
            np.median(dist_sgl), stats.median_abs_deviation(dist_sgl)
        )
    )
    print(
        "med_fwd = [{}/{}]".format(
            np.median(dist_fwd), stats.median_abs_deviation(dist_fwd)
        )
    )
    print(
        "med_bi = [{}/{}]".format(
            np.median(dist_bi), stats.median_abs_deviation(dist_bi)
        )
    )
    print(
        "med_ltracker = [{}/{}]".format(
            np.median(dist_ltracker), stats.median_abs_deviation(dist_ltracker)
        )
    )
    print(
        "med_int = [{}/{}]".format(
            np.median(dist_interp), stats.median_abs_deviation(dist_interp)
        )
    )
    print("mean_sgl = [{}/{}]".format(np.mean(dist_sgl), np.std(dist_sgl)))
    print("mean_fwd = [{}/{}]".format(np.mean(dist_fwd), np.std(dist_fwd)))
    print("mean_bi = [{}/{}]".format(np.mean(dist_bi), np.std(dist_bi)))
    print("mean_ltracker = [{}/{}]".format(np.mean(dist_ltracker), np.std(dist_ltracker)))
    print("mean_int = [{}/{}]".format(np.mean(dist_interp), np.std(dist_interp)))
    plt.show()

    plt.plot(dist_sgl, label="single")
    plt.plot(dist_fwd, label="fwd")
    # plt.plot(dist_bi, label="bi")
    plt.plot(dist_ltracker, label="ltracker")
    plt.legend()
    plt.show()

    np.savetxt("pos_net_fwd.csv", pos_net_fwd)
    np.savetxt("pos_net_bi.csv", pos_net_bi)
    np.savetxt("pos_gt.csv", pos_gt)
    np.savetxt("pos_interp.csv", pos_interp)


def signal_handler(sig, frame):
    show_results(pos_net_sgl, pos_net_fwd, pos_net_bi, pos_ltracker, pos_gt, grid)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
########################################################
# loader settings
########################################################
# {{{
num_workers = 2
prefetch_factor = 2
# }}}

########################################################
# generel settings
########################################################
# {{{ params
crop = 1024
kernel_size, kernel_sigma = 31, 3.
input_type = InputType.ImagesUnaries
model_type = ModelType.HourGlassSqueeze
# }}}

# {{ load/save
load_path = "enc/model.pt"
load_path_enc = "rec/model.pt"
logger.LOG_LEVEL = logger.INFO
log = logger.getLogger("Infer")
# }}}

########################################################
# datasets
########################################################
# {{{
# {{{ data prep
log.info("Training with device: {}".format(device))
transform = nn.Sequential(TransformGtClassification2x1d(crop))
inv_transform = InverseTransformGt(crop)  # InverseTransformGtClassification(crop)
vid_path = "/media/data/ant/Ant13R4/"

dataset = VideoDataset(
    folders=[vid_path],
    config={
        "crop_size": crop,
        "input_type": InputType.ImagesUnaries,
        "video_length": 0,
        "crop_center": True,
        "disjoint": True,
    },
    transform=transform,
)
# loader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=False,
#     num_workers=num_workers,
#     prefetch_factor=prefetch_factor
# )
tracker_pred = torch.zeros([len(dataset), 2]).float()
with open(
    pathlib.Path(
        "/media/data/ant/LTracker/continuous",
        pathlib.Path(vid_path).name + ".csv"
    )
) as f:
    csv_reader = csv.reader(f, delimiter=",")

    # skip first comment line
    r = 0
    # TOOD: replace this with inv_transform?
    for row in csv_reader:
        tracker_pred[
            r, 0] = 2. * ((float(row[0]) - (1920. - crop) // 2.) / float(crop)) - 1.
        tracker_pred[
            r, 1] = 2. * ((float(row[1]) - (1080. - crop) // 2.) / float(crop)) - 1.
        r += 1
log.info("Dataset: {}".format(dataset))
# }}}

first_elem = dataset[0]
image_size = first_elem[0].shape[-2:]
log.info("Dataset element sizes {} {}".format(first_elem[0].shape, first_elem[1].shape))
log.info("dataset size {}".format(len(dataset)))
# }}}
# }}}

########################################################
# checkpoint loading
########################################################
# {{{
checkpoint_enc, model_config_enc = load_model_config(load_path_enc)
checkpoint, model_config = load_model_config(load_path)

assert model_config_enc["enc_type"] == model_config["enc_type"], "Encoder types mismatch"
# }}}

########################################################
# model and optimizer
########################################################
# {{{
model, encoder = get_model(model_config)
model = model.to(device)
encoder = encoder.to(device)
model.print_network(verbose=False)
# }}}
checkpoint_single, model_config_single = load_model_config(load_path_enc)
single_net, _ = get_model_from_dataset(dataset, model_config_single)

########################################################
# checkpoint loading
########################################################
# {{{
if checkpoint_enc is not None:
    log.info("Loading encoder model from checkpoint")
    ckp_dict = checkpoint_enc["model_state_dict"]
    ckp_dict = {
        k[len("encoder."):]: v
        for k, v in ckp_dict.items() if k.startswith("encoder.")
    }
    encoder.load_state_dict(ckp_dict)
if checkpoint is not None:
    log.info("Loading recurrent model from checkpoint")
    model.load_state_dict(checkpoint['model_state_dict'])
if checkpoint_single is not None:
    log.info("Loading single_net model from checkpoint")
    single_net.load_state_dict(checkpoint_single['model_state_dict'])
    start_epoch = checkpoint_single["epoch"]
single_net.eval()
single_net_emb = single_net.emb.to(device)
single_net_x = single_net.mlp_x.to(device)
single_net_y = single_net.mlp_y.to(device)
# }}}

########################################################
# training loop
########################################################
# {{{
pos_net_sgl = torch.empty(len(dataset), 2)
pos_net_fwd = torch.empty(len(dataset), 2)
pos_net_bi = torch.empty(len(dataset), 2)
pos_gt = torch.empty(len(dataset), 2)
start, end = 0, len(dataset)
# nth = 120
curr_it = 0
last_flag = 0
flag = False
grid = []
clicks = 0
click_threshold = 0.1
delind = []
with torch.no_grad():
    model.eval()
    encoder.eval()
    hn = None
    for i in tqdm(range(start, end)):  # tqdm(range(len(dataset))):
        # if curr_it == 3000:
        #     break
        data, _, gt = dataset[i]
        data = data.to(device)
        gt = torch.tensor(gt).to(device)

        # if hn is None or curr_it % nth == 0:
        if hn is None or flag or i == end - 1:
            clicks += 1
            flag = False
            log.warning("Set new hidden state")

            # hn = model.get_hidden(torch.argmax(gt, dim=-1) / float(crop) - float(crop) / 2.)
            # hn = model.get_hidden(gt)
            # TODO: replace this with inv_transform
            hn = model.get_hidden((2. * torch.argmax(gt, dim=-1)) / float(crop) - 1.)

            # step back to last correction
            curr_hn = (hn[0].clone(), hn[1].clone())
            for k in tqdm(range(curr_it, last_flag, -1)):
                if k < 0:
                    continue
                prev_data, prev_label, _ = dataset[k]
                prev_data = prev_data.to(device)
                tmp, _, _ = dataset[k - 1]
                tmp = tmp.to(device)
                prev_data[:, -1] = tmp[:, -1]  # replace unary with the one before

                # null data for testing
                # prev_data[:, :3] = 0
                # prev_data[:, -1] = 0

                # TODO: maybe try to mask by hand? or is this equivalent to nulling?

                prev_regs, curr_hn = model.forward_single(encoder(prev_data), curr_hn)
                prev_regs = torch.argmax(prev_regs, dim=-1)
                prev_regs = prev_regs.to("cpu")
                prev_regs = (2. * (prev_regs / float(crop)) - 1.)

                # other metrics for averaging
                # alpha = (k - (curr_it - nth)) / nth
                # alpha = (k - last_flag) / (curr_it - last_flag)
                alpha = np.exp(-0.1 * (curr_it - k))
                if alpha < 0.2:
                    break

                pos_net_bi[k] = (1 -
                                 alpha) * pos_net_fwd[k] + alpha * prev_regs.view(-1, 2)
                # pos_net_bi[k] = prev_regs.view(-1, 2)

            grid.append(curr_it)
            last_flag = curr_it

        # data[:, :3] = 0
        # data[:, -1] = 0
        regs = encoder(data)
        regs_emb = single_net_emb(regs)
        regs_emb = regs_emb.view(1, -1)
        regs_x = single_net_x(regs_emb)
        regs_y = single_net_y(regs_emb)
        regs_single = torch.stack([regs_x, regs_y], dim=1)
        regs_single = torch.argmax(regs_single, dim=-1)
        regs_single = regs_single.to("cpu")

        # TOOD: replace this with inv_transform?
        regs_single = (2. * (regs_single / float(crop)) - 1.)

        regs, hn = model.forward_single(regs, hn)
        regs = torch.argmax(regs, dim=-1)
        regs = regs.to("cpu")

        # TOOD: replace this with inv_transform?
        regs = (2. * (regs / float(crop)) - 1.)
        gt = torch.argmax(gt, dim=-1)
        gt = gt.to("cpu")
        # fig = plt.figure()
        # plt.imshow(data[0, :3, :, :].permute(1, 2, 0).cpu())
        # plt.show()
        # plt.close(fig)

        # TOOD: replace this with inv_transform?
        gt = (2. * (gt / float(crop)) - 1.)
        pos_net_sgl[i] = regs_single.view(-1, 2)
        pos_net_fwd[i] = regs.view(-1, 2)
        pos_net_bi[i] = regs.view(-1, 2)
        pos_gt[i] = gt.view(-1, 2)
        if np.linalg.norm(pos_gt[i] - pos_net_fwd[i], axis=0) > click_threshold:
            flag = True
        curr_it += 1

        # fig = show_single_item(
        #     (data.cpu().numpy(), gt.cpu().numpy()), [regs.cpu().numpy()],
        #     show=False
        # )
        # plt.show()
        # plt.close(fig)

print(pos_gt.shape)
print("Video: ", pathlib.Path(vid_path).name)
print("Encoder: ", load_path_enc)
print("Recurrent: ", load_path)
print("#Clicks (including first): ", clicks)
print("#Frames: ", len(dataset))
print("Start frame: ", start)
print("End frame: ", end)
print("Left out frames: ", len(delind))
print("Threshold for click: ", click_threshold)
curr_it -= len(delind)
grid[-1] = curr_it - 1
show_results(pos_net_sgl, pos_net_fwd, pos_net_bi, tracker_pred, pos_gt, grid)
