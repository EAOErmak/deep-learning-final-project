from __future__ import annotations

from cs2_ai.ml.utils.torch_utils import torch_available

if torch_available():
    import torch
    from torch import nn


    class MovementGRU(nn.Module):
        def __init__(
            self,
            input_dim: int,
            action_dim: int = 7,
            chunk_len: int = 8,
            hidden_dim: int = 256,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_dim = int(input_dim)
            self.action_dim = int(action_dim)
            self.chunk_len = int(chunk_len)
            self.hidden_dim = int(hidden_dim)
            self.num_layers = int(num_layers)
            self.dropout = float(dropout)
            gru_dropout = self.dropout if self.num_layers > 1 else 0.0
            self.gru = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=gru_dropout,
            )
            self.head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.chunk_len * self.action_dim),
            )

        def forward(self, x):
            if x.ndim != 3:
                raise ValueError(
                    f"MovementGRU expects input with shape [batch, seq_len, input_dim], got {tuple(x.shape)}."
                )
            if int(x.shape[-1]) != self.input_dim:
                raise ValueError(
                    f"MovementGRU expected input_dim={self.input_dim}, got last dimension={int(x.shape[-1])}."
                )
            output, _hidden = self.gru(x)
            last_hidden = output[:, -1, :]
            logits = self.head(last_hidden)
            return logits.view(x.size(0), self.chunk_len, self.action_dim)
else:
    class MovementGRU:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available. Install torch to use MovementGRU.")
