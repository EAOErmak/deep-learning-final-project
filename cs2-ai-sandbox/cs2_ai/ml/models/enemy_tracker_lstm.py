from __future__ import annotations

from cs2_ai.ml.utils.torch_utils import torch_available

ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK = "each_tick"
ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK = "target_tick"

if torch_available():
    import torch
    from torch import nn


    class EnemyTrackerLSTM(nn.Module):
        """LSTM backbone for enemy tracking.

        Input shape:
            x: [batch, seq_len, input_dim]

        Output shapes by mode:
            output_mode="each_tick"
                positions: [batch, seq_len, output_enemies, 3]
                confidence: [batch, seq_len, output_enemies]
            output_mode="target_tick"
                positions: [batch, output_enemies, 3]
                confidence: [batch, output_enemies]
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            output_enemies: int = 5,
            dropout: float = 0.1,
            output_mode: str = ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK,
        ):
            super().__init__()
            if output_mode not in {ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK, ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK}:
                raise ValueError(
                    f"Unsupported EnemyTrackerLSTM output_mode={output_mode!r}. "
                    f"Expected {ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK!r} or {ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK!r}."
                )
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.num_layers = int(num_layers)
            self.output_enemies = int(output_enemies)
            self.dropout = float(dropout)
            self.output_mode = str(output_mode)
            lstm_dropout = float(dropout) if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.shared_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.position_head = nn.Linear(hidden_dim, output_enemies * 3)
            self.confidence_head = nn.Linear(hidden_dim, output_enemies)

        def forward(self, x):
            output, _ = self.lstm(x)
            if self.output_mode == ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK:
                hidden = self.shared_head(output[:, -1, :])
                positions = self.position_head(hidden).view(output.size(0), self.output_enemies, 3)
                confidence = self.confidence_head(hidden).view(output.size(0), self.output_enemies)
                return positions, confidence

            hidden = self.shared_head(output)
            positions = self.position_head(hidden).view(output.size(0), output.size(1), self.output_enemies, 3)
            confidence = self.confidence_head(hidden).view(output.size(0), output.size(1), self.output_enemies)
            return positions, confidence
else:
    class EnemyTrackerLSTM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available. Install torch to use EnemyTrackerLSTM.")
