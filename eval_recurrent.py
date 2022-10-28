# model
from tod.model.definition import ModelInterface, load_model_config
from tod.model.type import ModelType

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
import argparse
from pathlib import Path

device = getDevice()
abort = False

paths = {
    InputType.Images: {
        ModelType.HourGlassSqueeze: {
            True: "Recurrent_Images_HourGlassSqueeze_spatial",
            False: "Recurrent_Images_HourGlassSqueeze"
        },
        ModelType.ResnetClass: {
            True: "",
            False: "Recurrent_Images_ResnetClass",
        }
    },
    InputType.ImagesUnaries: {
        ModelType.HourGlass: {
            True: "",
            False: "Recurrent_ImagesUnaries_HourGlass"
        },
        ModelType.HourGlassSqueeze: {
            True: "Recurrent_ImagesUnaries_HourGlassSqueeze_spatial",
            False: "Recurrent_ImagesUnaries_HourGlassSqueeze"
        },
        ModelType.ResnetClass: {
            True: "",
            False: "Recurrent_ImagesUnaries_ResnetClass"
        },
        ModelType.ResnetReg: {
            True: "",
            False: "Recurrent_ImagesUnaries_ResnetReg"
        },
    },
    InputType.Unaries: {
        ModelType.HourGlassSqueeze: {
            True: "Recurrent_Unaries_HourGlassSqueeze_spatial",
            False: "Recurrent_Unaries_HourGlassSqueeze"
        },
        ModelType.ResnetClass: {
            True: "",
            False: "Recurrent_Unaries_ResnetClass"
        },
    }
}


def show_results(pos_net_sgl, pos_net_fwd, pos_net_bi, pos_gt, grid, curr_it):
    if torch.numel(pos_net_fwd) == 0:
        return

    pos_net_sgl = pos_net_sgl[start:curr_it]
    pos_net_fwd = pos_net_fwd[start:curr_it]
    pos_net_bi = pos_net_bi[start:curr_it]
    # pos_ltracker = pos_ltracker[start:curr_it]
    pos_gt = pos_gt[start:curr_it]

    time = np.arange(len(pos_gt))

    pos_interp = np.empty_like(pos_gt)
    pos_interp[:, 0] = interp.interp1d(
        time[grid], pos_gt[grid, 0], fill_value="extrapolate", kind="linear"
    )(time)
    pos_interp[:, 1] = interp.interp1d(
        time[grid], pos_gt[grid, 1], fill_value="extrapolate", kind="linear"
    )(time)

    gt = (pos_gt > crop).any(axis=1).nonzero()
    lt = (pos_gt < 0).any(axis=1).nonzero()
    delind = np.unique(np.sort(np.r_[gt, lt], axis=0))
    pos_net_sgl = np.delete(pos_net_sgl, delind, axis=0)
    pos_net_fwd = np.delete(pos_net_fwd, delind, axis=0)
    pos_net_bi = np.delete(pos_net_bi, delind, axis=0)
    pos_gt = np.delete(pos_gt, delind, axis=0)
    pos_interp = np.delete(pos_interp, delind, axis=0)
    # tracker_pred = np.delete(tracker_pred, delind, axis=0)

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(pos_net_sgl[:, 0], pos_net_sgl[:, 1], time, label="single")
    ax.plot(pos_net_fwd[:, 0], pos_net_fwd[:, 1], time, label="fwd")
    ax.plot(pos_net_bi[:, 0], pos_net_bi[:, 1], time, label="bi")
    # ax.plot(pos_ltracker[:, 0], pos_ltracker[:, 1], time, label="ltracker")
    ax.plot(pos_gt[:, 0], pos_gt[:, 1], time, label="gt")
    ax.plot(pos_interp[:, 0], pos_interp[:, 1], time, label="interp")
    ax.scatter(
        pos_gt[grid, 0],
        pos_gt[grid, 1],
        time[grid],
        label="clicks",
        marker="*",
        color='black',
        s=50,
        zorder=10
    )
    plt.legend()
    ax.set_xlim(0, crop)
    ax.set_ylim(0, crop)
    plt.show()

    dist_sgl = np.linalg.norm(pos_net_sgl - pos_gt, axis=1)
    dist_fwd = np.linalg.norm(pos_net_fwd - pos_gt, axis=1)
    dist_bi = np.linalg.norm(pos_net_bi - pos_gt, axis=1)
    # dist_ltracker = np.linalg.norm(pos_ltracker - pos_gt, axis=1)
    dist_interp = np.linalg.norm(torch.tensor(pos_interp) - pos_gt, axis=1)
    ones = np.ones_like(dist_fwd)

    _, ax2 = plt.subplots()
    plt.scatter(0.8 * ones, dist_sgl, c=np.arange(len(dist_bi)), s=4)
    plt.scatter(1.8 * ones, dist_fwd, c=np.arange(len(dist_fwd)), s=4)
    plt.scatter(2.8 * ones, dist_bi, c=np.arange(len(dist_bi)), s=4)
    # plt.scatter(3.8 * ones, dist_ltracker, c=np.arange(len(dist_bi)), s=4)
    plt.scatter(3.8 * ones, dist_interp, c=np.arange(len(dist_bi)), s=4)

    plt.violinplot([dist_sgl, dist_fwd, dist_bi])
    # labels = ["single", "fwd", "bi", "ltracker", "interp"]
    labels = ["single", "fwd", "bi", "interp"]
    ax2.set_xticks(np.arange(1, len(labels) + 1))
    ax2.set_xticklabels(labels)
    # ax2.set_xlim(0.25, len(labels) + 0.75)
    ax2.set_xlabel('Net architecture')
    plt.ylim(-100, crop)
    log.info(
        "med_sgl = [{}/{}]".format(
            np.median(dist_sgl), stats.median_abs_deviation(dist_sgl)
        )
    )
    log.info(
        "med_fwd = [{}/{}]".format(
            np.median(dist_fwd), stats.median_abs_deviation(dist_fwd)
        )
    )
    log.info(
        "med_bi = [{}/{}]".format(
            np.median(dist_bi), stats.median_abs_deviation(dist_bi)
        )
    )
    # log.info(
    #     "med_ltracker = [{}/{}]".format(
    #         np.median(dist_ltracker), stats.median_abs_deviation(dist_ltracker)
    #     )
    # )
    log.info(
        "med_int = [{}/{}]".format(
            np.median(dist_interp), stats.median_abs_deviation(dist_interp)
        )
    )
    log.info("mean_sgl = [{}/{}]".format(np.mean(dist_sgl), np.std(dist_sgl)))
    log.info("mean_fwd = [{}/{}]".format(np.mean(dist_fwd), np.std(dist_fwd)))
    log.info("mean_bi = [{}/{}]".format(np.mean(dist_bi), np.std(dist_bi)))
    # log.info("mean_ltracker = [{}/{}]".format(np.mean(dist_ltracker), np.std(dist_ltracker)))
    log.info("mean_int = [{}/{}]".format(np.mean(dist_interp), np.std(dist_interp)))
    plt.show()

    plt.plot(dist_sgl, label="single")
    plt.plot(dist_fwd, label="fwd")
    plt.plot(dist_bi, label="bi")
    [plt.axvline(g, c='black', linestyle=':') for g in grid]
    # plt.plot(dist_ltracker, label="ltracker")
    plt.legend()
    plt.show()

    vid = Path(vid_path).parts[-1]
    prefix = vid + "_" + str(input_type) + "_" + str(model_type) + "_" + str(
        spatial
    ) + "_" + str(mode)
    np.savetxt(
        f"/data/ant-ml-res/{prefix}_pos_net_sgl.csv", pos_net_sgl, header=str(clicks)
    )  # noqa
    np.savetxt(
        f"/data/ant-ml-res/{prefix}_pos_net_fwd.csv", pos_net_fwd, header=str(clicks)
    )  # noqa
    np.savetxt(
        f"/data/ant-ml-res/{prefix}_pos_net_bi.csv", pos_net_bi, header=str(clicks)
    )  # noqa
    np.savetxt(
        f"/data/ant-ml-res/{prefix}_pos_gt.csv", pos_gt, header=str(clicks)
    )  # noqa
    np.savetxt(
        f"/data/ant-ml-res/{prefix}_pos_interp.csv", pos_interp, header=str(clicks)
    )  # noqa


def signal_handler(sig, frame):
    global abort
    signal.signal(signal.SIGINT, orig_handler)
    # show_results(pos_net_sgl, pos_net_fwd, pos_net_bi, pos_gt, grid, curr_it)
    # sys.exit(0)
    abort = True


orig_handler = signal.getsignal(signal.SIGINT)
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
parser.add_argument(
    '-m',
    '--mode',
    required=True,
    type=int,
    choices=range(0, 3),
    help='0: threshold, 1: equi, 2: no'
)
parser.add_argument(
    '-s',
    '--spatial',
    action='store_const',
    const=True,
    default=False,
    help='HG across spatial or feature'
)
parser.add_argument('vidpath', type=pathlib.Path)

args = parser.parse_args()
input_type = InputType(args.input)
model_type = ModelType(args.type)
spatial = args.spatial
mode = args.mode

# {{ load/save
load_path = "/data/ant-ml-res/recurrent/" + paths[input_type][model_type][spatial
                                                                          ] + "/model.pt"
logger.LOG_LEVEL = logger.INFO
log = logger.getLogger("Infer")
# }}}

# {{{ model
model_interface = ModelInterface(
    type=model_type,
    input_type=input_type,
    crop=crop,
    kernel_size=kernel_size,  # only used by hourglass
    kernel_sigma=kernel_sigma,  # only used by hourglass
    freeze_encoder=True,
    hg_across_spatial=spatial,
)
# }}}

########################################################
# datasets
########################################################
# {{{ datasets
# {{{ data transforms and folders
log.info("Infering with device: {}".format(device))
transform = model_interface.transform
inv_transform = model_interface.inv_transform

vid_path = args.vidpath

dataset = VideoDataset(
    folders=[vid_path],
    config={
        "crop_size": crop,
        "input_type": input_type,
        "video_length": 0,
        "crop_center": True,
        "disjoint": True,
    },
)

log.info("Dataset: {}".format(dataset))
first_elem = dataset[0]
image_size = first_elem[0].shape[-2:]
log.info("Dataset element sizes {} {}".format(first_elem[0].shape, first_elem[1].shape))
log.info("dataset size {}".format(len(dataset)))

start, end = 0, len(dataset)
# start = 1982
# end = start + 50
# start = 1500
# end = start + 150
curr_it = start
# }}}

# {{{ tracker
# load tracker results
# tracker_pred = torch.zeros([len(dataset), 2]).float()
# with open(
#     pathlib.Path(
#         "/media/data/ant/LTracker/continuous",
#         pathlib.Path(vid_path).name + ".csv"
#     )
# ) as f:
#     csv_reader = csv.reader(f, delimiter=",")

#     # skip first comment line
#     r = 0
#     # TOOD: replace this with inv_transform?
#     for row in csv_reader:
#         tracker_pred[
#             r, 0] = 2. * ((float(row[0]) - (1920. - crop) // 2.) / float(crop)) - 1.
#         tracker_pred[
#             r, 1] = 2. * ((float(row[1]) - (1080. - crop) // 2.) / float(crop)) - 1.
#         r += 1
# }}}
# }}}

########################################################
# checkpoint loading
########################################################
checkpoint = load_model_config(load_path)

########################################################
# model and optimizer
########################################################
recurrent = model_interface.recurrent.to(device)

########################################################
# checkpoint loading
########################################################
# {{{
if checkpoint is not None:
    log.info("Loading recurrent model from checkpoint")
    recurrent.load_state_dict(checkpoint['recurrent_state_dict'])

recurrent.eval()
# }}}

########################################################
# training loop
########################################################
# {{{
pos_net_sgl = torch.empty(len(dataset), 2)
pos_net_fwd = torch.empty(len(dataset), 2)
pos_net_bi = torch.empty(len(dataset), 2)
pos_gt = torch.empty(len(dataset), 2)

last_flag = start
flag = False
grid = []
clicks = 0
click_threshold = 60
with torch.no_grad():
    hn = None
    bar = tqdm(total=(end - start))
    while curr_it < end:
        if abort:
            flag = True
            if last_flag == curr_it - 1:
                break

        data, _, gt = dataset[curr_it]
        data = data.to(device)
        gt = torch.tensor(gt).to(device)

        # make sure we have at least two points for linear interpolation
        if hn is None or flag or curr_it == end - 1:
            clicks += 1
            flag = False
            log.warning("Set new hidden state")

            hn = recurrent.get_hidden((2. * gt) / crop - 1.)

            # step back to last correction
            curr_hn = (hn[0].clone(), hn[1].clone())
            for k in tqdm(range(curr_it - 1, last_flag, -1)):
                if k < 1:
                    continue
                prev_data, prev_label, _ = dataset[k]
                prev_data = prev_data.to(device)

                # replace unary with the one before if input type contains unaries
                if input_type in [InputType.ImagesUnaries, InputType.Unaries]:
                    tmp, _, _ = dataset[k - 1]
                    tmp = tmp.to(device)
                    prev_data[:, -1] = tmp[:, -1]

                prev_regs, curr_hn = recurrent(prev_data, curr_hn)
                prev_regs = inv_transform(prev_regs).to("cpu").view(-1, 2)

                # other metrics for averaging
                # alpha = (k - (curr_it - nth)) / nth
                # alpha = (k - last_flag) / (curr_it - last_flag)
                alpha = np.exp(-0.2 * (curr_it - 1 - k))
                if alpha < 0.02:
                    break

                pos_net_bi[k] = (1 - alpha) * pos_net_fwd[k] + alpha * prev_regs

            grid.append(curr_it - start)
            last_flag = curr_it

        enc = recurrent.encoder(data)
        regs_single = recurrent.projector(enc)
        regs_single_inv = inv_transform(regs_single)

        regs, hn = recurrent(data, hn)
        regs_inv = inv_transform(regs)

        if model_type in [ModelType.HourGlass, ModelType.HourGlassSqueeze]:
            regs_single_inv = regs_single_inv[-1].to("cpu").view(-1, 2)
        else:
            regs_single_inv = regs_single_inv.to("cpu").view(-1, 2)

        regs_inv = regs_inv.to("cpu").view(-1, 2)

        gt = gt.to("cpu").view(-1, 2)

        # for heatmap gen
        # np.save(f"reccur_{curr_it}.npy", regs.to("cpu").numpy())
        # np.save(f"images_{curr_it}.npy", data.to("cpu").numpy())
        # np.save(f"single_{curr_it}.npy", regs_single[-1].to("cpu").numpy())
        # np.save(f"pos_single_{curr_it}.npy", regs_single_inv)
        # np.save(f"pos_recurrent_{curr_it}.npy", regs_inv)
        # np.save(f"pos_gt_{curr_it}.npy", gt.numpy())

        pos_net_sgl[curr_it] = regs_single_inv
        pos_net_fwd[curr_it] = regs_inv
        pos_net_bi[curr_it] = regs_inv
        pos_gt[curr_it] = gt

        delta = np.linalg.norm(gt.squeeze() - regs_inv.squeeze(), axis=0)
        if mode == 0:
            if delta > click_threshold and last_flag != curr_it:
                log.warning("delta flag {}".format(delta))
                flag = True
                continue
            if delta > click_threshold and last_flag == curr_it:
                log.warning("no improvement: {}".format(delta))
        elif mode == 1:
            if curr_it - last_flag == 50:
                log.warning("equidist flag")
                flag = True
                continue

        bar.update(1)
        curr_it += 1

log.info("Num gt: {}".format(pos_gt.shape))
log.info("Video: {}".format(pathlib.Path(vid_path).name))
log.info("Recurrent: {}".format(load_path))
log.info("#Clicks (including first): {}".format(clicks))
log.info("#Frames: {}".format(len(dataset)))
log.info("Threshold for click: {}".format(click_threshold))
show_results(pos_net_sgl, pos_net_fwd, pos_net_bi, pos_gt, grid, curr_it)
