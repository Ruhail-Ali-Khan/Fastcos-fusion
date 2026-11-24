import torch
import torchvision.ops as ops


def generate_points(H, W, stride, device):
    """
    Token centers in image coordinates
    """
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    points = torch.stack((x, y), dim=-1).reshape(-1, 2)
    return (points + 0.5) * stride


def decode_boxes(
    cls_logits,
    reg_pred,
    H,
    W,
    stride,
    score_thresh=0.05,
    nms_thresh=0.6,
):
    """
    Args:
        cls_logits: [N, C]
        reg_pred:   [N, 4] (l,t,r,b)
    Returns:
        boxes:  [M, 4] in x1,y1,x2,y2
        scores: [M]
        labels: [M]
    """

    device = cls_logits.device
    N, C = cls_logits.shape

    # Quality-aware classification score
    cls_scores = torch.sigmoid(cls_logits)

    # Token centers
    points = generate_points(H, W, stride, device)

    boxes_all = []
    scores_all = []
    labels_all = []

    for c in range(C):
        scores = cls_scores[:, c]
        keep = scores > score_thresh
        if keep.sum() == 0:
            continue

        pts = points[keep]
        ltrb = reg_pred[keep]
        s = scores[keep]

        # Decode l,t,r,b → x1,y1,x2,y2
        x1 = pts[:, 0] - ltrb[:, 0]
        y1 = pts[:, 1] - ltrb[:, 1]
        x2 = pts[:, 0] + ltrb[:, 2]
        y2 = pts[:, 1] + ltrb[:, 3]

        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        keep_nms = ops.nms(boxes, s, nms_thresh)

        boxes_all.append(boxes[keep_nms])
        scores_all.append(s[keep_nms])
        labels_all.append(
            torch.full((keep_nms.numel(),), c, device=device, dtype=torch.long)
        )

    if len(boxes_all) == 0:
        return (
            torch.zeros((0, 4), device=device),
            torch.zeros((0,), device=device),
            torch.zeros((0,), device=device, dtype=torch.long),
        )

    return (
        torch.cat(boxes_all, dim=0),
        torch.cat(scores_all, dim=0),
        torch.cat(labels_all, dim=0),
    )
