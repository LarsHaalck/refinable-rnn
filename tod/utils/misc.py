import torch
from torch.nn.functional import log_softmax
from torch import Tensor


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_pseudo_dim(input_size, model):
    out = model(torch.rand(input_size))
    # allows hourglass networks intermediate values to be process correctly
    if isinstance(out, list):
        return out[-1].shape
    else:
        return out.shape


def bounding_box(data):
    return (data.min(axis=0), data.max(axis=0))


def _neg_loss2x1d(pred, gt) -> Tensor:
    """
    Adapted from CornerNet code (https://github.com/princeton-vl/CornerNet)
    """
    pred_log = log_softmax(pred, dim=-1)

    pos_inds = gt.eq(1.)
    neg_inds = gt.lt(1.)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    pos_pred_log = pred_log[pos_inds]
    pos_pred = torch.exp(pos_pred_log)
    neg_pred = torch.exp(pred_log[neg_inds])

    pos_loss = pos_pred_log * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred + torch.finfo(torch.float32).tiny
                         ) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        return -1. * neg_loss
    return -1. * (pos_loss + neg_loss) / num_pos


def _neg_loss2d(preds, gt) -> Tensor:
    """
    Adapted from CornerNet code (https://github.com/princeton-vl/CornerNet)
    """
    if not isinstance(preds, list):
        preds = [preds]

    loss = torch.Tensor(0)
    for pred in preds:
        shape = pred.shape
        pred_log = log_softmax(pred.view(shape[0], -1), dim=-1).view(*shape)

        pos_inds = gt.eq(1.)
        neg_inds = gt.lt(1.)

        neg_weights = torch.pow(1 - gt[neg_inds], 4)

        pos_pred_log = pred_log[pos_inds]
        pos_pred = torch.exp(pos_pred_log)
        neg_pred = torch.exp(pred_log[neg_inds])

        pos_loss = pos_pred_log * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred + torch.finfo(torch.float32).tiny
                             ) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss += -1. * neg_loss
        else:
            loss += -1. * (pos_loss + neg_loss) / num_pos
    return loss
