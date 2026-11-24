import torch


def compute_iou_ltrb(pred, target):
    """
    pred, target: (..., 4) in l,t,r,b format
    """
    pl, pt, pr, pb = pred.unbind(-1)
    tl, tt, tr, tb = target.unbind(-1)

    inter_w = torch.min(pl, tl) + torch.min(pr, tr)
    inter_h = torch.min(pt, tt) + torch.min(pb, tb)
    inter = torch.clamp(inter_w, min=0) * torch.clamp(inter_h, min=0)

    area_p = (pl + pr) * (pt + pb)
    area_t = (tl + tr) * (tt + tb)
    union = area_p + area_t - inter + 1e-6

    return inter / union


def generate_points(H, W, stride, device):
    """
    Generate grid center points
    """
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    points = torch.stack((x, y), dim=-1).reshape(-1, 2)
    return (points + 0.5) * stride


def fastcos_targets(
    gt_boxes,
    gt_labels,
    H,
    W,
    stride,
    num_classes,
    device,
):
    """
    Args:
        gt_boxes: [G, 4] (x1,y1,x2,y2)
        gt_labels: [G]
        H, W: patch grid size
        stride: patch stride (e.g. 16)
    Returns:
        cls_target: [N, C] quality-aware targets
        reg_target: [N, 4] l,t,r,b
        reg_mask:  [N] positive mask
    """

    N = H * W
    cls_target = torch.zeros((N, num_classes), device=device)
    reg_target = torch.zeros((N, 4), device=device)
    reg_mask = torch.zeros((N,), dtype=torch.bool, device=device)

    if gt_boxes.numel() == 0:
        return cls_target, reg_target, reg_mask

    points = generate_points(H, W, stride, device)
    px, py = points[:, 0], points[:, 1]

    x1, y1, x2, y2 = gt_boxes.T

    l = px[:, None] - x1[None]
    t = py[:, None] - y1[None]
    r = x2[None] - px[:, None]
    b = y2[None] - py[:, None]

    ltrb = torch.stack([l, t, r, b], dim=-1)  # [N, G, 4]

    inside = (ltrb.min(dim=-1).values > 0)

    areas = (x2 - x1) * (y2 - y1)
    areas = areas[None].repeat(N, 1)
    areas[~inside] = 1e8

    min_area, min_idx = areas.min(dim=1)
    pos_mask = min_area < 1e8

    if pos_mask.sum() == 0:
        return cls_target, reg_target, reg_mask

    assigned = min_idx[pos_mask]
    ltrb_pos = ltrb[pos_mask, assigned]

    reg_target[pos_mask] = ltrb_pos
    reg_mask[pos_mask] = True

    # Quality target = IoU of target box with itself → 1.0 initially
    # During training, prediction IoU refines this
    quality = torch.ones((assigned.shape[0],), device=device)

    cls_target[pos_mask, gt_labels[assigned]] = quality

    return cls_target, reg_target, reg_mask
