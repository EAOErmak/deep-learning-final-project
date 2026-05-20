from __future__ import annotations

from cs2_ai.ml.utils.torch_utils import torch_available

if torch_available():
    import torch
    from torch import nn


    class EnemyTrackerLSTM(nn.Module):
        """LSTM backbone for enemy tracking.

        Input shape:
            x: [batch, seq_len, input_dim]

        Output shapes:
            positions: [batch, output_enemies, 3]
            confidence: [batch, output_enemies]
        """

        def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, output_enemies: int = 5):
            super().__init__()
            self.output_enemies = output_enemies
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.position_head = nn.Linear(hidden_dim, output_enemies * 3)
            self.confidence_head = nn.Linear(hidden_dim, output_enemies)

        def forward(self, x):
            output, _ = self.lstm(x)
            positions = self.position_head(output).view(output.size(0), output.size(1), self.output_enemies, 3)
            confidence = self.confidence_head(output)
            return positions, confidence
else:
    class EnemyTrackerLSTM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available. Install torch to use EnemyTrackerLSTM.")

