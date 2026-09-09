# src/dino/head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """
    Projection head with L2-normalized bottleneck and weight-normalized prototype layer.
    Flow: in_dim (768) -> hidden_dim (2048) -> bottleneck_dim (256) -> out_dim (e.g. 65536)
    """
    def __init__(
        self,
        in_dim=768,
        out_dim=65536,       # 65536 for more images
        hidden_dim=2048,
        bottleneck_dim=256,
        nlayers=3
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        # Prototype projection layer (Standard linear layer)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, in_dim] -> [B, bottleneck_dim]
        x = self.mlp(x)

        # L2-normalize features across bottleneck dimension
        x = F.normalize(x, dim=-1, p=2)

        # Manual L2 normalization on prototype weights (replaces fragile weight_norm)
        norm_weight = F.normalize(self.last_layer.weight, dim=-1, p=2)

        # [B, bottleneck_dim] -> [B, out_dim]
        x = F.linear(x, norm_weight)
        return x


class FullDINOModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token, _ = self.backbone(x)
        return self.head(cls_token)