import torch
from torch.utils.data import DataLoader
from data_n.nuscenes_dataset import NuScenes2D
from fusion.patch_embed import PatchEmbed
from fastcos.backbone import FastCOSBackbone
from fastcos.head import FastCOSHead
from fastcos.targets import fastcos_targets
from fastcos.loss import fastcos_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    dataset = NuScenes2D(
        root="D:/nuscenes",
        version="v1.0-trainval",
        img_h=640,
        img_w=960,
        use_lidar=True,
        max_scenes=85    
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    patch = PatchEmbed(3).to(DEVICE)
    backbone = FastCOSBackbone().to(DEVICE)
    head = FastCOSHead().to(DEVICE)

    params = list(patch.parameters()) + \
             list(backbone.parameters()) + \
             list(head.parameters())
    opt = torch.optim.AdamW(params, lr=2e-4)

    for epoch in range(12):
        for img, _, boxes, labels in loader:
            img = img.to(DEVICE)

            tokens, H, W = patch(img)
            feats = backbone(tokens, H, W)
            cls_logits, reg_pred = head(feats)

            cls_targets = []
            reg_targets = []
            masks = []

            for b in range(img.shape[0]):
                ct, rt, m = fastcos_targets(
                    boxes[b].to(DEVICE),
                    labels[b].to(DEVICE),
                    H, W,
                    stride=16,
                    num_classes=10,
                    device=DEVICE
                )
                cls_targets.append(ct)
                reg_targets.append(rt)
                masks.append(m)

            cls_t = torch.stack(cls_targets, dim=0)
            reg_t = torch.stack(reg_targets, dim=0)
            mask  = torch.stack(masks, dim=0)


            loss_dict = fastcos_loss(
                cls_logits,
                reg_pred,
                cls_t.unsqueeze(0),
                reg_t.unsqueeze(0),
                mask.unsqueeze(0)
            )


            opt.zero_grad()
            loss_dict["loss"].backward()
            opt.step()

        print(f"[Baseline] Epoch {epoch+1} loss:", loss_dict["loss"].item())

if __name__ == "__main__":
    train()
