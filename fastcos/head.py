import torch
import torch.nn as nn
import torch.nn.functional as F


class FastCOSHead(nn.Module):
    """
    Fast-COS detection head:
    - Quality-aware classification (no centerness)
    - Anchor-free box regression
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()

        def make_tower():
            layers = []
            for _ in range(num_layers):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
            return nn.Sequential(*layers)

        self.cls_tower = make_tower()
        self.reg_tower = make_tower()

        self.cls_logits = nn.Linear(hidden_dim, num_classes)
        self.reg_pred = nn.Linear(hidden_dim, 4)

        # Initialize
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_logits.bias, 0.0)
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)

    def forward(self, tokens):
        """
        Args:
            tokens: [B, N, C]

        Returns:
            cls_logits: [B, N, num_classes]
            box_deltas: [B, N, 4] (l, t, r, b)
        """

        cls_feat = self.cls_tower(tokens)
        reg_feat = self.reg_tower(tokens)

        cls_logits = self.cls_logits(cls_feat)
        box_deltas = F.relu(self.reg_pred(reg_feat))  # distances must be positive

        return cls_logits, box_deltas
