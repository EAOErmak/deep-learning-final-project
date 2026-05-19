from __future__ import annotations

from cs2_ai.ml.utils.torch_utils import torch_available

if torch_available():
    import torch
    from torch import nn


    class DecisionDQN(nn.Module):
        def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, x):
            if x.dim() == 3:
                x = x.mean(dim=1)
            return self.net(x)
else:
    class DecisionDQN:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available. Install torch to use DecisionDQN.")
