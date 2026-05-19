from __future__ import annotations

from cs2_ai.ml.utils.torch_utils import torch_available

if torch_available():
    import torch
    from torch import nn


    class AimAttentionModel(nn.Module):
        def __init__(self, input_dim: int, model_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, model_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.aim_head = nn.Linear(model_dim, 2)
            self.shoot_head = nn.Linear(model_dim, 1)
            self.rightclick_head = nn.Linear(model_dim, 1)

        def forward(self, x):
            hidden = self.input_proj(x)
            encoded = self.encoder(hidden)
            pooled = encoded[:, -1, :]
            aim_delta = self.aim_head(pooled)
            shoot_logits = self.shoot_head(pooled)
            rightclick_logits = self.rightclick_head(pooled)
            return aim_delta, shoot_logits, rightclick_logits
else:
    class AimAttentionModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available. Install torch to use AimAttentionModel.")
