from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from torch.utils.data import DataLoader, Dataset

from cs2_ai.dataset.multi_demo_sequence_dataset import MultiDemoSequenceDataset, split_dataset_by_group
from cs2_ai.features.decision_features import DecisionFeatureExtractor
from cs2_ai.dataset.reward_builder import BasicRewardBuilder
from cs2_ai.modules.decision_maker import RuleBasedDecisionMaker
from cs2_ai.state.belief_state import BeliefState
from cs2_ai.schemas.module_outputs import EnemyTrackerOutput
from cs2_ai.schemas.game_state import GameStateSequence
from cs2_ai.ml.models.decision_dqn import DecisionDQN
from cs2_ai.ml.utils.tensorboard_utils import close_summary_writer, create_summary_writer, log_scalar_dict, tensorboard_available
from cs2_ai.ml.utils.torch_utils import build_dataloader_kwargs, configure_torch_runtime, get_device, set_seed, torch_available

if torch_available():
    import torch
    import torch.nn.functional as F
else:
    torch = None
    F = None

STRATEGIC_ACTION_TO_ID = {
    "buy": 0,
    "retake": 1,
    "defend_site": 2,
    "move_to_objective": 3,
    "defuse": 4,
    "plant": 5,
}

STRATEGIC_ACTION_FROM_ID = {v: k for k, v in STRATEGIC_ACTION_TO_ID.items()}


def get_base_dataset_and_index(dataset: Any, idx: int) -> tuple[Any, int]:
    curr_dataset = dataset
    curr_idx = idx
    while hasattr(curr_dataset, 'dataset') and hasattr(curr_dataset, 'indices'):
        curr_idx = curr_dataset.indices[curr_idx]
        curr_dataset = curr_dataset.dataset
    return curr_dataset, curr_idx


@dataclass(slots=True)
class RLTransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    sample_ids: list[str]
    demo_names: list[str]


class DecisionRLTransitionDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.feature_extractor = DecisionFeatureExtractor()
        self.reward_builder = BasicRewardBuilder()
        self.decision_maker = RuleBasedDecisionMaker()
        self.belief_updater = BeliefState()
        self.dummy_tracker_output = EnemyTrackerOutput(predictions=[])

    def __len__(self) -> int:
        return len(self.base_dataset)

    def get_sample_metadata(self, idx: int) -> dict[str, object]:
        ds, real_idx = get_base_dataset_and_index(self.base_dataset, idx)
        return ds.get_sample_metadata(real_idx)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, float, np.ndarray, float, dict[str, str]]:
        sequence_sample = self.base_dataset[idx]
        ds, real_idx = get_base_dataset_and_index(self.base_dataset, idx)
        sample_metadata = ds.get_sample_metadata(real_idx)
        
        states = sequence_sample.sequence.states
        target_tick = int(sample_metadata['target_tick'])
        target_state = ds.build_state_for_sample_tick(sample_metadata, target_tick)
        
        belief_state = self.belief_updater.update(states[-1], self.dummy_tracker_output)
        next_belief_state = self.belief_updater.update(target_state, self.dummy_tracker_output)
        
        state_features = self.feature_extractor.extract(sequence_sample.sequence, belief_state)
        
        next_sequence = GameStateSequence(
            perspective_steamid=sequence_sample.perspective_steamid,
            states=states[1:] + [target_state]
        )
        next_state_features = self.feature_extractor.extract(next_sequence, next_belief_state)
        
        decision_out = self.decision_maker.decide(target_state, next_belief_state)
        action_name = decision_out.strategic_action
        action_id = STRATEGIC_ACTION_TO_ID.get(action_name, 3)
        
        reward = self.reward_builder.compute_local_reward(states[-1], target_state)
        
        done = not target_state.self_player.is_alive or not target_state.round.round_in_progress
        done_val = 1.0 if done else 0.0
        
        meta = {
            'sample_id': str(sample_metadata['sample_id']),
            'demo_name': str(sample_metadata['demo_name']),
        }
        
        return state_features[-1], action_id, float(reward), next_state_features[-1], done_val, meta


class DecisionRLTrainer:
    def __init__(
        self,
        policy_net: torch.nn.Module,
        target_net: torch.nn.Module,
        device: str,
        learning_rate: float,
        gamma: float = 0.99,
        target_update_freq: int = 5,
        log_interval: int = 100,
    ):
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.log_interval = log_interval
        
        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def train_epoch(self, loader: DataLoader, epoch: int, writer: Any | None = None) -> dict[str, object]:
        self.policy_net.train()
        metrics = self._run_epoch(loader, training=True, epoch=epoch, writer=writer)
        if epoch % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Target network updated at epoch {epoch}")
        return metrics

    def eval_epoch(self, loader: DataLoader, epoch: int, writer: Any | None = None) -> dict[str, object]:
        self.policy_net.eval()
        with torch.no_grad():
            return self._run_epoch(loader, training=False, epoch=epoch, writer=writer)

    def _run_epoch(self, loader: DataLoader, training: bool, epoch: int, writer: Any | None = None) -> dict[str, object]:
        total_loss = 0.0
        total_reward = 0.0
        total_samples = 0
        total_batches = len(loader)
        phase_name = 'Train' if training else 'Val'
        
        seen_sample_ids: set[str] = set()
        seen_demo_sample_ids: dict[str, set[str]] = {}
        per_demo_sum: dict[str, float] = {}
        per_demo_count: dict[str, int] = {}
        
        for batch_idx, batch in enumerate(loader):
            batch = self._to_batch(batch)
            
            # Predict Q-values
            q_values = self.policy_net(batch.states)
            q_a = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
            
            # Compute targets
            with torch.no_grad():
                # Double DQN style
                next_actions = self.policy_net(batch.next_states).max(1)[1].unsqueeze(1)
                max_next_q = self.target_net(batch.next_states).gather(1, next_actions).squeeze(1)
                target_q = batch.rewards + self.gamma * max_next_q * (1.0 - batch.dones)
                
            loss = F.mse_loss(q_a, target_q)
            
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            batch_size = batch.states.size(0)
            total_samples += batch_size
            total_loss += float(loss.item() * batch_size)
            total_reward += float(batch.rewards.sum().item())
            
            if training and writer is not None:
                global_step = (epoch - 1) * total_batches + batch_idx
                writer.add_scalar('train/loss_step', loss.item(), global_step)
                writer.add_scalar('train/reward_step', batch.rewards.mean().item(), global_step)
                writer.add_scalar('train/q_value_mean_step', q_values.mean().item(), global_step)
                if batch_idx % 10 == 0:
                    writer.flush()
                    
            for sample_id, demo_name, sample_loss in zip(batch.sample_ids, batch.demo_names, [loss.item()] * batch_size):
                seen_sample_ids.add(sample_id)
                seen_demo_sample_ids.setdefault(demo_name, set()).add(sample_id)
                per_demo_sum[demo_name] = per_demo_sum.get(demo_name, 0.0) + float(sample_loss)
                per_demo_count[demo_name] = per_demo_count.get(demo_name, 0) + 1
                
            should_log_batch = (
                batch_idx == 0
                or (batch_idx + 1) % self.log_interval == 0
                or batch_idx == total_batches - 1
            )
            if should_log_batch:
                print(f'{phase_name} epoch {epoch} | Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item():.4f} | Seen: {len(seen_sample_ids)}')

        if total_samples == 0:
            return {
                'loss': 0.0,
                'reward': 0.0,
                'seen_sample_ids': set(),
                'per_demo_loss': {},
                'per_demo_seen_counts': {},
            }
            
        return {
            'loss': total_loss / total_samples,
            'reward': total_reward / total_samples,
            'seen_sample_ids': seen_sample_ids,
            'per_demo_loss': {name: per_demo_sum[name] / per_demo_count[name] for name in per_demo_sum},
            'per_demo_seen_counts': {name: len(seen_demo_sample_ids[name]) for name in seen_demo_sample_ids},
        }

    def _to_batch(self, batch_tuple) -> RLTransitionBatch:
        states, actions, rewards, next_states, dones, meta = batch_tuple
        return RLTransitionBatch(
            states=states.to(device=self.device, dtype=torch.float32),
            actions=actions.to(device=self.device, dtype=torch.long),
            rewards=rewards.to(device=self.device, dtype=torch.float32),
            next_states=next_states.to(device=self.device, dtype=torch.float32),
            dones=dones.to(device=self.device, dtype=torch.float32),
            sample_ids=meta['sample_id'],
            demo_names=meta['demo_name'],
        )


def build_coverage_summary(metrics: dict[str, object], expected_counts: dict[str, int]) -> dict[str, object]:
    seen = metrics.get('seen_sample_ids', set())
    total_expected = sum(expected_counts.values())
    overall_pct = (len(seen) / total_expected * 100.0) if total_expected > 0 else 0.0
    
    per_demo_info: dict[str, dict[str, object]] = {}
    per_demo_loss = metrics.get('per_demo_loss', {})
    per_demo_seen = metrics.get('per_demo_seen_counts', {})
    
    for name, expected in expected_counts.items():
        count_seen = per_demo_seen.get(name, 0)
        pct = (count_seen / expected * 100.0) if expected > 0 else 0.0
        loss = per_demo_loss.get(name, 0.0)
        per_demo_info[name] = {
            'seen': count_seen,
            'expected': expected,
            'pct': pct,
            'loss': loss,
        }
        
    return {
        'total_seen': len(seen),
        'total_expected': total_expected,
        'pct': overall_pct,
        'per_demo': per_demo_info,
    }


def print_coverage_summary(phase: str, coverage: dict[str, object]) -> None:
    print(f'{phase} coverage: {coverage["total_seen"]}/{coverage["total_expected"]} samples ({coverage["pct"]:.2f}%) | demos {len(coverage["per_demo"])}')
    for name, info in coverage['per_demo'].items():
        print(f'  {phase} demo {name}: {info["seen"]}/{info["expected"]} ({info["pct"]:.2f}%) | avg_loss={info["loss"]:.4f}')


def append_epoch_summary(log_path: Path, record: dict[str, object]) -> None:
    serializable = {}
    for k, v in record.items():
        if isinstance(v, dict):
            serializable[k] = {ik: iv for ik, iv in v.items() if ik != 'seen_sample_ids'}
        else:
            serializable[k] = v
            
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(serializable, ensure_ascii=False) + '\n')


def save_checkpoint(
    save_path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
    train_metrics: dict[str, object],
    val_metrics: dict[str, object],
    dataset_label: str,
    input_dim: int,
    demo_names: list[str],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'decision_dqn_offline',
        'input_dim': input_dim,
        'action_dim': 6,
        'seq_len': args.seq_len,
        'stride': args.stride,
        'dataset_source': dataset_label,
        'demo_names': demo_names,
        'demo_count': len(demo_names),
        'split_mode': args.split_mode,
        'train_metrics': {k: v for k, v in train_metrics.items() if k not in {'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'}},
        'val_metrics': {k: v for k, v in val_metrics.items() if k not in {'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'}},
        'feature_order': 'DecisionFeatureExtractor end step vector',
    }
    torch.save(checkpoint, save_path)


def load_checkpoint_if_available(policy_net: torch.nn.Module, target_net: torch.nn.Module, resume_from: Path | None, device: str) -> bool:
    if resume_from is None or not resume_from.exists():
        return False
    checkpoint = torch.load(resume_from, map_location=device)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])
    print(f'Resumed model weights from: {resume_from}')
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train an offline DecisionDQN model from transitions')
    parser.add_argument('--dataset-dir', type=Path, default=PROJECT_ROOT / 'dataset')
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target-update-freq', type=int, default=5)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--split-mode', type=str, choices=['random', 'demo', 'round'], default='demo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-samples-per-demo', type=int, default=None)
    parser.add_argument('--max-cached-demos', type=int, default=2)
    parser.add_argument('--save-path', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'decision_dqn.pt')
    parser.add_argument('--resume-from', type=Path, default=None)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--runs-dir', type=Path, default=PROJECT_ROOT / 'runs')
    parser.add_argument('--tensorboard-run-name', type=str, default=None)
    parser.add_argument('--disable-tensorboard', action='store_true')
    return parser.parse_args()


def main() -> int:
    if not torch_available():
        print("PyTorch is not available. Install torch to use train_decision_offline.py")
        return 1
        
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    runtime_info = configure_torch_runtime(device)
    
    print(f"Device: {device}")
    
    # Load MultiDemoSequenceDataset
    base_dataset = MultiDemoSequenceDataset(
        dataset_dir=args.dataset_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        max_samples_total=args.max_samples,
        max_samples_per_demo=args.max_samples_per_demo,
        max_cached_demos=args.max_cached_demos,
    )
    
    if len(base_dataset) == 0:
        print("Dataset is empty. Ensure you have parquet files in dataset/clean_play_ticks")
        return 1
        
    print(f"Total base samples: {len(base_dataset)}")
    
    # Split dataset
    train_subset, val_subset = split_dataset_by_group(
        base_dataset,
        val_split=args.val_split,
        seed=args.seed,
        mode=args.split_mode,
    )
    
    train_dataset = DecisionRLTransitionDataset(train_subset)
    val_dataset = DecisionRLTransitionDataset(val_subset)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    train_loader_kwargs = build_dataloader_kwargs(device, args.num_workers, is_training=True)
    val_loader_kwargs = build_dataloader_kwargs(device, args.num_workers, is_training=False)
    print(f'DataLoader workers: train={train_loader_kwargs["num_workers"]} val={val_loader_kwargs["num_workers"]}')
    print(f'CUDA tuning: matmul={runtime_info["matmul_precision"]} cudnn_benchmark={runtime_info["cudnn_benchmark"]} tf32={runtime_info["tf32"]}')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **val_loader_kwargs)
    
    # Expected sample counts per demo
    train_expected_counts = {}
    for idx in train_subset.indices:
        meta = base_dataset.get_sample_metadata(idx)
        name = str(meta['demo_name'])
        train_expected_counts[name] = train_expected_counts.get(name, 0) + 1
        
    val_expected_counts = {}
    for idx in val_subset.indices:
        meta = base_dataset.get_sample_metadata(idx)
        name = str(meta['demo_name'])
        val_expected_counts[name] = val_expected_counts.get(name, 0) + 1
        
    # Instantiate models
    feature_extractor = DecisionFeatureExtractor()
    input_dim = feature_extractor.feature_dim()
    action_dim = 6
    
    policy_net = DecisionDQN(input_dim=input_dim, action_dim=action_dim, hidden_dim=args.hidden_dim).to(device)
    target_net = DecisionDQN(input_dim=input_dim, action_dim=action_dim, hidden_dim=args.hidden_dim).to(device)
    load_checkpoint_if_available(policy_net, target_net, args.resume_from, device)
    
    trainer = DecisionRLTrainer(
        policy_net=policy_net,
        target_net=target_net,
        device=device,
        learning_rate=args.lr,
        gamma=args.gamma,
        target_update_freq=args.target_update_freq,
        log_interval=args.log_interval,
    )
    
    demo_names = base_dataset.get_demo_names()
    dataset_label = str(args.dataset_dir / 'clean_play_ticks')
    epoch_log_path = args.save_path.with_name(f'{args.save_path.stem}_epoch_metrics.jsonl')
    
    # Remove old epoch log if exists
    if epoch_log_path.exists():
        epoch_log_path.unlink()
        
    writer = None
    if not args.disable_tensorboard and tensorboard_available():
        writer, run_dir = create_summary_writer(
            runs_dir=args.runs_dir,
            run_name=args.tensorboard_run_name,
            default_prefix='decision',
            save_path=args.save_path,
            config={
                'args': vars(args),
                'device': device,
                'dataset_source': dataset_label,
                'demo_names': demo_names,
            },
        )
        print(f"TensorBoard run: {run_dir}")
        
    best_val_loss = math.inf
    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = trainer.train_epoch(train_loader, epoch=epoch, writer=writer)
            val_metrics = trainer.eval_epoch(val_loader, epoch=epoch, writer=writer) if len(val_dataset) > 0 else {
                'loss': train_metrics['loss'],
                'reward': train_metrics['reward'],
                'seen_sample_ids': set(),
                'per_demo_loss': {},
                'per_demo_seen_counts': {},
            }
            
            print(f'Epoch {epoch}/{args.epochs} | train_loss={train_metrics["loss"]:.4f} (reward={train_metrics["reward"]:.4f}) | val_loss={val_metrics["loss"]:.4f} (reward={val_metrics["reward"]:.4f})')
            
            train_coverage = build_coverage_summary(train_metrics, train_expected_counts)
            val_coverage = build_coverage_summary(val_metrics, val_expected_counts)
            print_coverage_summary('train', train_coverage)
            if val_expected_counts:
                print_coverage_summary('val', val_coverage)
                
            append_epoch_summary(epoch_log_path, {
                'epoch': epoch,
                'train': {'loss': train_metrics['loss'], 'reward': train_metrics['reward'], 'coverage': train_coverage},
                'val': {'loss': val_metrics['loss'], 'reward': val_metrics['reward'], 'coverage': val_coverage}
            })
            
            log_scalar_dict(writer, 'train', train_metrics, epoch, ignored_keys={'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'})
            log_scalar_dict(writer, 'val', val_metrics, epoch, ignored_keys={'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'})
            log_scalar_dict(writer, 'train_coverage', train_coverage, epoch, ignored_keys={'per_demo'})
            log_scalar_dict(writer, 'val_coverage', val_coverage, epoch, ignored_keys={'per_demo'})
            if writer is not None:
                writer.flush()
                
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(args.save_path, policy_net, args, train_metrics, val_metrics, dataset_label, input_dim, demo_names)
                print(f'  saved checkpoint -> {args.save_path}')
    finally:
        if writer is not None:
            close_summary_writer(writer)
            
    print("Training finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
