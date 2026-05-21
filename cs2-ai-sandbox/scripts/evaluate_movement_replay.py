from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from cs2_ai.dataset.multi_demo_sequence_dataset import MultiDemoSequenceDataset
from cs2_ai.features.movement_features import MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE, movement_action_names_for_target_mode
from cs2_ai.ml.training.shape_assertions import assert_shape
from cs2_ai.ml.training.train_movement import (
    MOVEMENT_MODEL_DECISION_DQN,
    MOVEMENT_MODEL_GRU,
    MovementSequenceTorchDataset,
    build_model,
)
from cs2_ai.ml.utils.torch_utils import get_device, torch_available

if torch_available():
    import torch
else:
    torch = None

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'artifacts' / 'movement_replay_eval'


@dataclass(frozen=True, slots=True)
class ReplayEvalConfig:
    checkpoint_path: Path
    dataset_dir: Path
    demo_name: str
    round_number: int
    perspective_steamid: int
    threshold: float
    output_dir: Path
    max_windows: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Offline replay evaluator for movement checkpoints')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, default=PROJECT_ROOT / 'dataset')
    parser.add_argument('--demo', type=str, required=True)
    parser.add_argument('--round', dest='round_number', type=int, required=True)
    parser.add_argument('--perspective-steamid', type=int, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--max-windows', type=int, default=None)
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    checkpoint_stem = args.checkpoint.stem
    return DEFAULT_OUTPUT_ROOT / checkpoint_stem / f'{args.demo}_round{args.round_number}_p{args.perspective_steamid}'


def load_checkpoint(path: Path, device: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f'Movement checkpoint not found: {path}')
    checkpoint = torch.load(path, map_location=device)
    if 'model_state_dict' not in checkpoint:
        raise ValueError(f'Checkpoint {path} is missing model_state_dict.')
    return checkpoint


def build_eval_dataset(checkpoint: dict[str, object], config: ReplayEvalConfig) -> MovementSequenceTorchDataset:
    seq_len = int(checkpoint.get('seq_len', 64))
    stride = int(checkpoint.get('stride', 8))
    target_mode = str(checkpoint.get('target_mode', MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE))
    chunk_len = int(checkpoint.get('chunk_len', 8))
    base_dataset = MultiDemoSequenceDataset(
        dataset_dir=config.dataset_dir,
        subdir='clean_play_ticks',
        seq_len=seq_len,
        stride=stride,
        alive_only=True,
        include_demo_names={config.demo_name},
        show_progress=False,
    )
    return MovementSequenceTorchDataset(
        base_dataset,
        target_mode=target_mode,
        chunk_len=chunk_len,
    )


def filter_sample_indices(dataset: MovementSequenceTorchDataset, round_number: int, perspective_steamid: int) -> list[int]:
    indices: list[int] = []
    for idx in range(len(dataset)):
        meta = dataset.get_sample_metadata(idx)
        if int(meta['round_number']) != int(round_number):
            continue
        if int(meta['perspective_steamid']) != int(perspective_steamid):
            continue
        indices.append(idx)
    return indices


def build_model_from_checkpoint(checkpoint: dict[str, object], device: str):
    model_name = str(checkpoint.get('movement_model_name') or checkpoint.get('model_type') or MOVEMENT_MODEL_DECISION_DQN)
    if model_name == 'movement_gru':
        model_name = MOVEMENT_MODEL_GRU
    if model_name == 'decision_dqn_movement':
        model_name = MOVEMENT_MODEL_DECISION_DQN
    input_dim = int(checkpoint['input_dim'])
    action_names = checkpoint.get('action_names')
    if not isinstance(action_names, list) or not action_names:
        target_mode = str(checkpoint.get('target_mode', MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE))
        action_names = list(movement_action_names_for_target_mode(target_mode))
    action_dim = int(checkpoint.get('action_dim', len(action_names)))
    chunk_len = int(checkpoint.get('chunk_len', 8))
    hidden_dim = int(checkpoint.get('hidden_dim', 256))
    gru_num_layers = int(checkpoint.get('gru_num_layers', 2))
    gru_dropout = float(checkpoint.get('gru_dropout', 0.1))
    model = build_model(
        model_name=model_name,
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        target_len=chunk_len,
        gru_num_layers=gru_num_layers,
        gru_dropout=gru_dropout,
        device=device,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, model_name, list(action_names)


def normalize_logits(model_name: str, logits: 'torch.Tensor', expected_shape: tuple[int, int, int]) -> 'torch.Tensor':
    if model_name == MOVEMENT_MODEL_GRU:
        assert_shape(logits, expected_shape, 'movement replay logits')
        return logits
    if logits.ndim != 3:
        raise ValueError(f'Movement replay logits must be rank-3, got shape {tuple(logits.shape)}.')
    batch_size, target_len, action_dim = expected_shape
    if logits.shape[0] != batch_size or logits.shape[2] != action_dim:
        raise ValueError(
            f'Movement replay logits shape mismatch: expected batch/action {(batch_size, action_dim)}, got {tuple(logits.shape)}.'
        )
    if logits.shape[1] < target_len:
        raise ValueError(
            f'Movement replay logits sequence too short: expected at least {target_len}, got {int(logits.shape[1])}.'
        )
    sliced = logits[:, -target_len:, :]
    assert_shape(sliced, expected_shape, 'movement replay normalized logits')
    return sliced


def predict_chunk(
    model,
    model_name: str,
    features_np: np.ndarray,
    targets_np: np.ndarray,
    seq_len: int,
    input_dim: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    assert_shape(features_np, (seq_len, input_dim), 'movement replay features')
    assert_shape(targets_np, tuple(targets_np.shape), 'movement replay targets')
    features = torch.tensor(features_np[None, ...], dtype=torch.float32, device=device)
    targets = torch.tensor(targets_np[None, ...], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(features)
        normalized = normalize_logits(model_name, logits, tuple(targets.shape))
        probs = torch.sigmoid(normalized)
    return normalized.squeeze(0).cpu().numpy(), probs.squeeze(0).cpu().numpy()


def build_report_rows(
    dataset: MovementSequenceTorchDataset,
    sample_idx: int,
    action_names: list[str],
    threshold: float,
    prob_chunk: np.ndarray,
    target_chunk: np.ndarray,
) -> list[dict[str, object]]:
    meta = dataset.get_sample_metadata(sample_idx)
    target_ticks = dataset._resolve_target_ticks(meta)
    if target_ticks is None:
        raise ValueError(f'Failed to resolve target ticks for sample {meta["sample_id"]}.')
    pred_chunk = (prob_chunk >= threshold).astype(np.float32)
    rows: list[dict[str, object]] = []
    for chunk_offset, tick in enumerate(target_ticks):
        row: dict[str, object] = {
            'demo_name': str(meta['demo_name']),
            'round': int(meta['round_number']),
            'perspective_steamid': int(meta['perspective_steamid']),
            'window_sample_id': str(meta['sample_id']),
            'window_target_tick': int(meta['target_tick']),
            'window_last_input_tick': int(meta['tick_indices'][-1]),
            'tick': int(tick),
            'chunk_offset': int(chunk_offset),
        }
        for action_idx, action_name in enumerate(action_names):
            row[f'target_{action_name}'] = float(target_chunk[chunk_offset, action_idx])
            row[f'pred_{action_name}'] = float(pred_chunk[chunk_offset, action_idx])
            row[f'confidence_{action_name}'] = float(prob_chunk[chunk_offset, action_idx])
        rows.append(row)
    return rows


def summarize_report(report_df: pd.DataFrame, action_names: list[str]) -> dict[str, object]:
    per_action: dict[str, dict[str, float | int]] = {}
    for action_name in action_names:
        target = report_df[f'target_{action_name}'].to_numpy(dtype=np.float32)
        pred = report_df[f'pred_{action_name}'].to_numpy(dtype=np.float32)
        tp = int(np.sum((pred == 1.0) & (target == 1.0)))
        fp = int(np.sum((pred == 1.0) & (target == 0.0)))
        fn = int(np.sum((pred == 0.0) & (target == 1.0)))
        tn = int(np.sum((pred == 0.0) & (target == 0.0)))
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-8))
        per_action[action_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'target_positive_ratio': float(target.mean()) if len(target) else 0.0,
            'predicted_positive_ratio': float(pred.mean()) if len(pred) else 0.0,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }
    return {
        'rows': int(len(report_df)),
        'tick_min': int(report_df['tick'].min()) if not report_df.empty else None,
        'tick_max': int(report_df['tick'].max()) if not report_df.empty else None,
        'per_action': per_action,
    }


def save_time_plot(report_df: pd.DataFrame, action_names: list[str], output_path: Path) -> None:
    if plt is None:
        print('matplotlib is unavailable; skipping PNG plot export.')
        return
    plot_actions = [name for name in ('forward', 'left', 'right', 'jump') if name in action_names]
    if not plot_actions:
        print('No plot actions available in report; skipping PNG plot export.')
        return
    plot_df = (
        report_df.groupby('tick', as_index=False)
        .agg(
            {
                **{f'target_{name}': 'mean' for name in plot_actions},
                **{f'pred_{name}': 'mean' for name in plot_actions},
            }
        )
        .sort_values('tick')
    )
    fig, axes = plt.subplots(len(plot_actions), 1, figsize=(14, 2.8 * len(plot_actions)), sharex=True)
    if len(plot_actions) == 1:
        axes = [axes]
    for axis, action_name in zip(axes, plot_actions, strict=True):
        axis.step(plot_df['tick'], plot_df[f'target_{action_name}'], where='mid', label=f'target_{action_name}', linewidth=1.6)
        axis.step(plot_df['tick'], plot_df[f'pred_{action_name}'], where='mid', label=f'pred_{action_name}', linewidth=1.2, alpha=0.85)
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel(action_name)
        axis.grid(alpha=0.25)
        axis.legend(loc='upper right')
    axes[-1].set_xlabel('tick')
    fig.suptitle('Movement replay evaluation')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate(config: ReplayEvalConfig) -> int:
    if not torch_available():
        print('PyTorch is not available. Install torch to run movement replay evaluation.')
        return 1

    device = get_device()
    checkpoint = load_checkpoint(config.checkpoint_path, device)
    dataset = build_eval_dataset(checkpoint, config)
    filtered_indices = filter_sample_indices(dataset, config.round_number, config.perspective_steamid)
    if config.max_windows is not None:
        filtered_indices = filtered_indices[: config.max_windows]
    if not filtered_indices:
        print(
            f'No movement samples found for demo={config.demo_name} round={config.round_number} '
            f'perspective={config.perspective_steamid}.'
        )
        return 1

    model, model_name, action_names = build_model_from_checkpoint(checkpoint, device)
    seq_len = int(checkpoint.get('seq_len', 64))
    input_dim = int(checkpoint['input_dim'])
    report_rows: list[dict[str, object]] = []
    for dataset_idx in filtered_indices:
        features_np, target_np, _ = dataset[dataset_idx]
        logits_np, prob_np = predict_chunk(
            model,
            model_name,
            features_np,
            target_np,
            seq_len,
            input_dim,
            device,
        )
        assert_shape(logits_np, tuple(target_np.shape), 'movement replay logits numpy')
        assert_shape(prob_np, tuple(target_np.shape), 'movement replay probabilities')
        report_rows.extend(
            build_report_rows(
                dataset=dataset,
                sample_idx=dataset_idx,
                action_names=action_names,
                threshold=config.threshold,
                prob_chunk=prob_np,
                target_chunk=target_np,
            )
        )

    report_df = pd.DataFrame(report_rows)
    if report_df.empty:
        print('Movement replay report is empty after evaluation.')
        return 1

    summary = summarize_report(report_df, action_names)
    csv_path = config.output_dir / 'movement_replay_report.csv'
    summary_path = config.output_dir / 'movement_replay_summary.json'
    plot_path = config.output_dir / 'movement_replay_plot.png'
    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding='utf-8')
    save_time_plot(report_df, action_names, plot_path)

    print(f'Movement replay evaluation saved: {config.output_dir}')
    print(f'Windows evaluated: {len(filtered_indices)}')
    print(f'Rows exported: {len(report_df)}')
    print(f'CSV: {csv_path}')
    print(f'Summary: {summary_path}')
    if plt is not None:
        print(f'Plot: {plot_path}')
    print('Per-action summary:')
    for action_name in action_names:
        metrics = summary['per_action'][action_name]
        print(
            f'  {action_name}: '
            f'precision={metrics["precision"]:.4f} '
            f'recall={metrics["recall"]:.4f} '
            f'f1={metrics["f1"]:.4f} '
            f'tgt_pos={metrics["target_positive_ratio"]:.4f} '
            f'pred_pos={metrics["predicted_positive_ratio"]:.4f} '
            f'tp={metrics["tp"]} fp={metrics["fp"]} fn={metrics["fn"]} tn={metrics["tn"]}'
        )
    return 0


def main() -> int:
    args = parse_args()
    config = ReplayEvalConfig(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        demo_name=args.demo,
        round_number=args.round_number,
        perspective_steamid=args.perspective_steamid,
        threshold=float(args.threshold),
        output_dir=resolve_output_dir(args),
        max_windows=args.max_windows,
    )
    return evaluate(config)


if __name__ == '__main__':
    raise SystemExit(main())
