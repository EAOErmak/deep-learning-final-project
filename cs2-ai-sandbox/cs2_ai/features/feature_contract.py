from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


SCHEMA_VERSION = "v1"
NORMALIZATION_CONSTANTS = {
    "position_scale": 10000.0,
    "velocity_scale": 1000.0,
    "money_scale": 16000.0,
    "hp_scale": 100.0,
    "armor_scale": 100.0,
    "angle_scale": 180.0,
}


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    model_key: str
    version: str
    seq_len: int
    feature_names: tuple[str, ...]
    default_value: float
    normalization: dict[str, float]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def schema_hash(self) -> str:
        payload = {
            "model_key": self.model_key,
            "version": self.version,
            "seq_len": self.seq_len,
            "feature_names": list(self.feature_names),
            "default_value": self.default_value,
            "normalization": self.normalization,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def to_metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["schema_hash"] = self.schema_hash
        data["feature_dim"] = self.feature_dim
        return data


def pad_or_trim_sequence(frames: list[list[float]], seq_len: int, frame_dim: int, default_value: float = 0.0) -> list[list[float]]:
    padded = [list(frame) for frame in frames[-seq_len:]]
    while len(padded) < seq_len:
        padded.insert(0, [default_value] * frame_dim)
    return padded


def validate_checkpoint_schema(checkpoint: dict[str, object], expected_schema: FeatureSchema, checkpoint_path: str) -> None:
    metadata = checkpoint.get("feature_schema")
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} is missing feature_schema metadata.")
    schema_hash = metadata.get("schema_hash")
    if schema_hash != expected_schema.schema_hash:
        raise ValueError(
            f"Checkpoint {checkpoint_path} schema mismatch: expected {expected_schema.schema_hash}, got {schema_hash}."
        )
    if int(metadata.get("seq_len", -1)) != int(expected_schema.seq_len):
        raise ValueError(
            f"Checkpoint {checkpoint_path} seq_len mismatch: expected {expected_schema.seq_len}, got {metadata.get('seq_len')}."
        )
    if int(metadata.get("feature_dim", -1)) != int(expected_schema.feature_dim):
        raise ValueError(
            f"Checkpoint {checkpoint_path} feature_dim mismatch: expected {expected_schema.feature_dim}, got {metadata.get('feature_dim')}."
        )
