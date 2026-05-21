# Movement GRU Default Refactor

## Что изменено

- `train_movement.py` теперь по умолчанию использует `--model movement_gru`.
- `target_mode` по умолчанию переключен на `action_chunk`.
- Добавлен явный inspection первого movement batch до начала training loop.
- Checkpoint metadata для movement training сохранен в более явном виде.
- Legacy-режимы `decision_dqn` и `next_tick_sequence` не удалены.

## Новые default параметры

- `--model movement_gru`
- `--target-mode action_chunk`
- `--chunk-len 8`

## Ожидаемые shapes

### `action_chunk`

- `features`: `[B, seq_len, feature_dim]`
- `targets`: `[B, chunk_len, action_dim]`
- `action_dim = 7`

Action order:

- `forward`
- `back`
- `left`
- `right`
- `walk`
- `crouch`
- `jump`

### `next_tick_sequence`

- `features`: `[B, seq_len, feature_dim]`
- `targets`: `[B, seq_len, action_dim]`
- `action_dim = 6`

Action order:

- `forward`
- `back`
- `left`
- `right`
- `walk`
- `crouch`

## Как запустить обучение

```powershell
python -m cs2_ai.ml.training.train_movement ^
  --dataset-dir dataset ^
  --model movement_gru ^
  --target-mode action_chunk ^
  --seq-len 64 ^
  --chunk-len 8 ^
  --batch-size 32 ^
  --epochs 3 ^
  --save-path checkpoints/movement_stream.pt
```

## Как проверить первый batch

Скрипт автоматически печатает inspection первого `train_loader` batch:

- `features shape`
- `features dtype`
- `targets shape`
- `targets dtype`
- `feature_dim`
- `action_names`
- `first sample_id`
- `first demo_name`
- target positive ratios по actions внутри batch

## Сохраненные legacy режимы

- `--model decision_dqn`
- `--target-mode next_tick_sequence`

Для legacy-режимов остаются warnings, но совместимость сохранена.

## Примечание по запуску

Полноценный train run требует наличие `dataset/clean_play_ticks`.
