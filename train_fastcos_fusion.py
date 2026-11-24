import torch
from torch.utils.data import DataLoader
from data_n.nuscenes_dataset import NuScenes2D
from fusion.patch_embed import PatchEmbed
from fusion.clft_fusion import CLFTFusionBlock
from fastcos.backbone import FastCOSBackbone
from fastcos.head import FastCOSHead
from fastcos.targets import fastcos_targets
from fastcos.loss import fastcos_loss
import os
os.makedirs("checkpoints", exist_ok=True)
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

    cam_patch = PatchEmbed(3).to(DEVICE)
    lid_patch = PatchEmbed(1).to(DEVICE)
    fusion = CLFTFusionBlock().to(DEVICE)
    backbone = FastCOSBackbone().to(DEVICE)
    head = FastCOSHead().to(DEVICE)

    params = (
        list(cam_patch.parameters()) +
        list(lid_patch.parameters()) +
        list(fusion.parameters()) +
        list(backbone.parameters()) +
        list(head.parameters())
    )

    opt = torch.optim.AdamW(params, lr=2e-4)

    for epoch in range(2):
        for img, lidar, boxes, labels in loader:
            img = img.to(DEVICE)
            lidar = lidar.to(DEVICE)

            cam_tok, H, W = cam_patch(img)
            lid_tok, _, _ = lid_patch(lidar)

            fused = fusion(cam_tok, lid_tok)
            feats = backbone(fused, H, W)
            cls_logits, reg_pred = head(feats)

            cls_t, reg_t, mask = fastcos_targets(
                boxes[0].to(DEVICE),
                labels[0].to(DEVICE),
                H, W,
                stride=16,
                num_classes=10,
                device=DEVICE
            )

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

        print(f"[Fusion] Epoch {epoch+1} loss:", loss_dict["loss"].item())
        torch.save(
            {
                "epoch": epoch,
                "cam_patch": cam_patch.state_dict(),
                "lid_patch": lid_patch.state_dict(),
                "fusion": fusion.state_dict(),
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
            },
            f"checkpoints/fastcos_fusion_epoch_{epoch}.pth"
        )

if __name__ == "__main__":
    train()
