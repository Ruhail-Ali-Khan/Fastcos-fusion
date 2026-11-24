import torch
import torch.nn.functional as F


def sigmoid(x):
    return torch.sigmoid(x)


def varifocal_loss(pred_logits, target_scores, alpha=0.75, gamma=2.0):
    """
    pred_logits:   [B, N, C]
    target_scores: [B, N, C]
    """

    pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])
    target_scores = target_scores.reshape(-1, target_scores.shape[-1])

    pred_prob = torch.sigmoid(pred_logits)

    pos_mask = target_scores > 0
    neg_mask = ~pos_mask

    loss = torch.zeros_like(pred_prob)

    # positive samples
    loss[pos_mask] = (
        target_scores[pos_mask]
        * (target_scores[pos_mask] - pred_prob[pos_mask]).abs().pow(gamma)
        * F.binary_cross_entropy_with_logits(
            pred_logits[pos_mask],
            target_scores[pos_mask],
            reduction="none"
        )
    )

    # negative samples
    loss[neg_mask] = (
        alpha
        * pred_prob[neg_mask].pow(gamma)
        * F.binary_cross_entropy_with_logits(
            pred_logits[neg_mask],
            target_scores[neg_mask],
            reduction="none"
        )
    )

    return loss.sum()



def iou_loss(pred_ltrb, target_ltrb):
    """
    IoU loss for l,t,r,b boxes
    """
    pl, pt, pr, pb = pred_ltrb.unbind(-1)
    tl, tt, tr, tb = target_ltrb.unbind(-1)

    inter_w = torch.min(pl, tl) + torch.min(pr, tr)
    inter_h = torch.min(pt, tt) + torch.min(pb, tb)
    inter = torch.clamp(inter_w, min=0) * torch.clamp(inter_h, min=0)

    area_p = (pl + pr) * (pt + pb)
    area_t = (tl + tr) * (tt + tb)
    union = area_p + area_t - inter + 1e-6

    iou = inter / union
    return 1.0 - iou


def fastcos_loss(
    cls_logits,
    reg_pred,
    cls_targets,
    reg_targets,
    reg_mask,
):
    """
    Args:
        cls_logits: [B, N, C]
        reg_pred:   [B, N, 4]
        cls_targets: [B, N, C]
        reg_targets: [B, N, 4]
        reg_mask: [B, N] bool
    """

    B, N, C = cls_logits.shape

    cls_loss = 0.0
    reg_loss = 0.0
    num_pos = 0

    for b in range(B):
        cls_loss += varifocal_loss(
            cls_logits[b],
            cls_targets[b],
        )

        if reg_mask[b].any():
            pred = reg_pred[b][reg_mask[b]]
            target = reg_targets[b][reg_mask[b]]

            # quality = max cls target per token
            quality = cls_targets[b][reg_mask[b]].max(dim=1).values.detach()

            reg_l = iou_loss(pred, target)
            reg_loss += (reg_l * quality).sum()
            num_pos += quality.numel()

    if num_pos > 0:
        reg_loss /= num_pos

    total_loss = cls_loss + reg_loss

    return {
    "loss": total_loss,
    "cls_loss": float(cls_loss),
    "reg_loss": float(reg_loss),
    }
