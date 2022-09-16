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


# adapted from pytorch (without "self" on transformer module)
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0,
                                    float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# adapted from:
# https://github.com/huggingface/transformers/blob/63b90a51aaa982757235525492768106cf9ac50f/src/transformers/generation_logits_process.py#L171
def nucleus_sampling(
    *,
    scores: Tensor,
    top_p: float = 0.95,
    min_tokens_to_keep: int = 1,
    filter_value: float = -float("Inf")
) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because
        # we add the first one below)
        sorted_indices_to_remove[..., :min_tokens_to_keep - 1] = 0
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def bounding_box(data):
    return (data.min(axis=0), data.max(axis=0))


def _neg_loss2x1d(pred, gt):
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


def _neg_loss2d(preds, gt):
    b, w, h = preds[0].shape
    loss = 0
    for pred in preds:
        pred_log = log_softmax(pred.view(b, -1), dim=-1).view(b, w, h)

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
