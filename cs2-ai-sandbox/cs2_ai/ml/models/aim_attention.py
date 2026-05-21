from __future__ import annotations

from cs2_ai.ml.utils.torch_utils import torch_available

AIM_HEAD_MODE_LEGACY = "legacy"
AIM_HEAD_MODE_MULTI_HEAD = "multi_head"

if torch_available():
    import torch
    from torch import nn


    class AimAttentionModel(nn.Module):
        def __init__(
            self,
            input_dim: int,
            model_dim: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            head_mode: str = AIM_HEAD_MODE_LEGACY,
            dropout: float = 0.1,
        ):
            super().__init__()
            if head_mode not in {AIM_HEAD_MODE_LEGACY, AIM_HEAD_MODE_MULTI_HEAD}:
                raise ValueError(f"Unsupported aim head_mode={head_mode!r}.")
            self.input_dim = int(input_dim)
            self.model_dim = int(model_dim)
            self.num_heads = int(num_heads)
            self.num_layers = int(num_layers)
            self.head_mode = str(head_mode)
            self.dropout = float(dropout)
            self.input_proj = nn.Linear(input_dim, model_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                batch_first=True,
                dropout=dropout,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.shared_head = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if self.head_mode == AIM_HEAD_MODE_MULTI_HEAD:
                self.aim_delta_head = nn.Linear(model_dim, 4)
                self.fire_head = nn.Linear(model_dim, 3)
                self.confidence_head = nn.Linear(model_dim, 1)
            else:
                self.aim_head = nn.Linear(model_dim, 2)
                self.shoot_head = nn.Linear(model_dim, 1)
                self.rightclick_head = nn.Linear(model_dim, 1)

        def forward(self, x):
            hidden = self.input_proj(x)
            encoded = self.encoder(hidden)
            pooled = self.shared_head(encoded[:, -1, :])
            if self.head_mode == AIM_HEAD_MODE_MULTI_HEAD:
                aim_delta = self.aim_delta_head(pooled)
                binary_logits = self.fire_head(pooled)
                confidence_logits = self.confidence_head(pooled)
                return aim_delta, binary_logits, confidence_logits
            aim_delta = self.aim_head(pooled)
            shoot_logits = self.shoot_head(pooled)
            rightclick_logits = self.rightclick_head(pooled)
            return aim_delta, shoot_logits, rightclick_logits
else:
    class AimAttentionModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available. Install torch to use AimAttentionModel.")
