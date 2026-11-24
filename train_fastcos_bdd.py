import os
import torch
from torch.utils.data import DataLoader

from data.nuscene_eval import BDD100KDetection
from fusion.patch_embed import PatchEmbed
from fastcos.backbone import FastCOSBackbone
from fastcos.head import FastCOSHead
from fastcos.targets import fastcos_targets
from fastcos.loss import fastcos_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10

def train():

    dataset = BDD100KDetection(
        root=r"bdd100k",
        split="train",
        img_h=720,
        img_w=1280
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    patch = PatchEmbed(3).to(DEVICE)
    backbone = FastCOSBackbone().to(DEVICE)
    head = FastCOSHead(num_classes=NUM_CLASSES).to(DEVICE)

    params = list(patch.parameters()) + \
             list(backbone.parameters()) + \
             list(head.parameters())

    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.05)

    os.makedirs("checkpoints_bdd", exist_ok=True)

    for epoch in range(12):

        patch.train()
        backbone.train()
        head.train()

        total_loss = 0

        for img, boxes, labels in loader:

            img = img.to(DEVICE)

            tokens, H, W = patch(img)
            feats = backbone(tokens, H, W)
            cls_logits, reg_pred = head(feats)

            cls_targets, reg_targets, masks = [], [], []

            for b in range(img.shape[0]):
                ct, rt, m = fastcos_targets(
                    boxes[b].to(DEVICE),
                    labels[b].to(DEVICE),
                    H, W,
                    stride=16,
                    num_classes=NUM_CLASSES,
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
                cls_t,
                reg_t,
                mask
            )

            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[BDD100K] Epoch {epoch+1} Loss: {avg_loss:.4f}")

        torch.save({
            "epoch": epoch + 1,
            "patch": patch.state_dict(),
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict()
        }, f"checkpoints_bdd/fastcos_epoch_{epoch+1}.pth")

    print("Training completed.")

if __name__ == "__main__":
    train()
