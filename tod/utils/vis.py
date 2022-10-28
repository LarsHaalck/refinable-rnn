import matplotlib.pyplot as plt
import matplotlib.cm as cm
from einops import rearrange, reduce
import numpy as np
import torch
from tod.model.definition import ModelType
from tod.io import InputType
from torch.nn.functional import softmax
from torch import nn
from einops.layers.torch import Rearrange


def show_single_item(item, preds=[], show=True):
    is4d = (item[0].ndim == 4)
    imgs, unaries, gt = seperate_data(item)
    if not isinstance(preds, list):
        preds = [preds]
    # if pred is None:
    #     pred = gt.copy()

    hasImgs = imgs is not None
    hasUnaries = unaries is not None
    both = hasImgs and hasUnaries

    if hasImgs:
        shape = imgs.shape
    else:
        shape = unaries.shape

    # recalc original gt
    if gt.max() < 1:
        gt = inverse_transform(gt, shape)
        preds = [inverse_transform(pred, shape) for pred in preds]

    # shift gt for vis
    if is4d:
        if gt.shape[0] > 1:
            gt[:, 0] += get_multi_shift_width(item[0].shape)
            for i in range(len(preds)):
                preds[i][:, 0] += get_multi_shift_width(item[0].shape)
        else:
            gt[:, 0] += get_shift_width(item[0].shape)
            for i in range(len(preds)):
                preds[i][:, 0] += get_shift_width(item[0].shape)

    if gt.ndim == 1:
        gt = np.expand_dims(gt, axis=0)
        preds = [np.expand_dims(pred, axis=0) for pred in preds]

    fig = plt.figure(figsize=(20, 10))
    if both:
        plt.subplot(311)
    if imgs is not None:
        if is4d:
            imgs = rearrange(imgs, 't c h w -> h (t w) c')
        else:
            imgs = rearrange(imgs, 'c h w -> h w c')

        plt.imshow(imgs)
        plot_pred(preds)
        plot_gt(gt)
        set_grid(item[0].shape)
        plt.title("Image and Detection")

    if both:
        plt.subplot(312)
    if unaries is not None:
        # unaries = unaries.squeeze()
        if is4d:
            unaries = rearrange(unaries, 't ()  h w -> h (t w)')
        plt.imshow(unaries)
        plot_pred(preds)
        plot_gt(gt)
        set_grid(item[0].shape)
        plt.title("Unary and Detection")

    if both:
        plt.subplot(313)
        overlayed = imgs[..., 0] + 2 * unaries
        plt.imshow(overlayed)
        plot_pred(preds)
        plot_gt(gt)
        set_grid(item[0].shape)
        plt.title("Unary overlayed and Detection")

    plt.legend()
    if show:
        plt.show()
    else:
        return fig


def set_grid(shape):
    plt.xticks(get_multi_shift_width(shape))
    plt.yticks([])
    plt.grid()


def inverse_transform(data, shape):
    return (data + 1) / 2 * shape[-2:][::-1]


def get_shift_width(shape):
    return (shape[0] // 2) * shape[-1]


def get_multi_shift_width(shape):
    return np.arange(shape[0]) * shape[-1]


def plot_gt(gt):
    plt.scatter(gt[:, 0], gt[:, 1], color='red', marker='x', label="gt")


def plot_pred(preds):
    colors = ["yellow", "green", "blue"]
    for i in range(len(preds)):
        plt.scatter(
            preds[i][:, 0],
            preds[i][:, 1],
            color=colors[i],
            marker='x',
            label="pred" + str(i)
        )


def seperate_data(item):
    imgs = None
    unaries = None
    gt = None
    if len(item) == 2:
        if item[0].ndim == 4:
            if item[0].shape[1] == 3:
                imgs = item[0]
            elif item[0].shape[1] == 4:
                imgs = item[0][:, :3]
                unaries = item[0][:, [-1]]  # to keep dimension
            else:
                unaries = item[0]
            gt = item[1]
        else:
            if item[0].shape[0] == 3:
                imgs = item[0]
            elif item[0].shape[0] == 4:
                imgs = item[0][:3]
                unaries = item[0][-1]
            else:
                unaries = item[0]
            gt = item[1]
    if len(item) == 3:
        imgs = item[0]
        unaries = item[1]
        gt = item[2]

    return imgs, unaries, gt


MAX = None


def plot_points(*, net, gt, show=True):
    global MAX
    net = net.reshape(-1, 2)
    gt = gt.reshape(-1, 2)
    plt.scatter(net[:, 0], net[:, 1], color="C1", label="net")
    plt.scatter(gt[:, 0], gt[:, 1], color="C2", label="gt")
    plt.legend()
    norm = np.linalg.norm(net - gt, axis=1)
    if MAX is None:
        MAX = norm.max()
    norm = norm / MAX
    cmap = cm.get_cmap('viridis')
    for i in range(len(net)):
        plt.plot([net[i, 0], gt[i, 0]], [net[i, 1], gt[i, 1]], c=cmap(norm[i]))

    if show:
        plt.show()


def plot_loss(losses, val_losses, baseline=0, show=True):
    plt.plot(losses, label="loss")
    plt.plot(val_losses, label="val_los")
    if baseline > 0:
        plt.axhline(baseline, linestyle='--', color='black', label="baseline")
    plt.legend()

    if show:
        plt.show()


def plot_attention(
    att,
    depth: int,
    time_steps: int,
    multi: bool,
    cls: bool,
    time_mean: bool = True,
    title: str = ""
):
    sps = 3 if multi else 2
    for i in range(depth):  # depth
        if cls:
            curr_att = att[i][:time_steps, time_steps:]
        else:
            curr_att = att[i]
        plt.subplot(depth, sps, sps * i + 1)

        if multi:
            lines = plt.plot(curr_att[:, ::2].T)
            plt.plot(curr_att[:, 1::2].T, ":")
        else:
            lines = plt.plot(curr_att.T)

        plt.legend(iter(lines), ["att_" + str(i) for i in range(len(lines))])
        plt.suptitle(title)

        if not time_mean:
            return

        if multi:
            plt.subplot(depth, sps, sps * i + 2)
            curr_att = reduce(curr_att, 'a (t b x) -> b a t', 'sum', t=time_steps, b=2)
            plt.matshow(curr_att[0], fignum=False)

            plt.subplot(depth, sps, sps * i + 3)
            plt.matshow(curr_att[1], fignum=False)
        else:
            plt.subplot(depth, sps, sps * i + 2)
            curr_att = reduce(curr_att, 'a (t x) -> a t', 'sum', t=time_steps)
            plt.matshow(curr_att, fignum=False)


def plot_heatmap(*, input_type, input, crop, model_type, out, regs_point, gt, temp=10):
    if model_type == ModelType.ResnetClass:
        regs_img = torch.outer(
            softmax(out[0, 1] / temp, 0), softmax(out[0, 0] / temp, 0)
        ).to("cpu")
        regs_img /= regs_img.max()
        regs_img = [regs_img]
    elif model_type == ModelType.ResnetReg:
        regs_img = [torch.ones(crop, crop, 1)]
    else:
        sm = nn.Sequential(
            Rearrange("b h w -> b (h w)"),
            nn.Softmax(dim=-1),
            Rearrange("b (h w) -> b h w", h=crop, w=crop),
        )
        regs_img = [sm(o / temp).to("cpu").squeeze() for o in out]
        regs_img = [o / o.max() for o in regs_img]

    _, ax = plt.subplots(len(regs_point), 4)
    input = input.to("cpu")

    for i, img in enumerate(regs_img):
        pt = regs_point[i]
        if input_type in [InputType.Unaries, InputType.ImagesUnaries]:
            un_m = input[0, 0, [-1]].permute(1, 2, 0)
        else:
            un_m = torch.ones(*input[0, 0].shape[1:], 1)

        if input_type in [InputType.Images, InputType.ImagesUnaries]:
            im_m = input[0, 0, :3].permute(1, 2, 0)
        else:
            im_m = torch.ones(*input[0, 0].shape[1:], 1)

        cax = ax[i, 0] if len(regs_point) > 1 else ax[0]
        cax.imshow(img.squeeze(0), cmap=plt.get_cmap("YlOrBr"))
        cax.scatter(gt[0, 0], gt[0, 1], color='C9', marker='o', label="gt")
        cax.scatter(pt[0, 0], pt[0, 1], color='C3', marker='x', label="pred")
        cax.set_title("Heatmap")

        cax = ax[i, 1] if len(regs_point) > 1 else ax[1]
        cax.imshow(im_m)
        cax.scatter(gt[0, 0], gt[0, 1], color='C9', marker='o', label="gt")
        cax.scatter(pt[0, 0], pt[0, 1], color='C3', marker='x', label="pred")
        cax.set_title("Raw image (if given)")

        cax = ax[i, 2] if len(regs_point) > 1 else ax[2]
        cax.imshow(
            un_m,
            cmap=plt.get_cmap("binary") if un_m.mean() == 1 else plt.get_cmap("YlOrBr")
        )
        cax.scatter(gt[0, 0], gt[0, 1], color='C9', marker='o', label="gt")
        cax.scatter(pt[0, 0], pt[0, 1], color='C3', marker='x', label="pred")
        cax.set_title("Raw unary (if given)")

        img_m = img.unsqueeze(-1)
        img_m = (im_m * img_m + un_m * img_m)

        cax = ax[i, 3] if len(regs_point) > 1 else ax[3]
        cax.imshow(img_m)
        cax.scatter(gt[0, 0], gt[0, 1], color='C9', marker='o', label="gt")
        cax.scatter(pt[0, 0], pt[0, 1], color='C3', marker='x', label="pred")
        cax.set_title("Raw unary (if given)")
    plt.show()
