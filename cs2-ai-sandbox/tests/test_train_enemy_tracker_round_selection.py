from pathlib import Path

from cs2_ai.ml.training.train_enemy_tracker import select_stream_round_files


def test_select_stream_round_files_returns_single_deterministic_round_when_requested():
    round_files = [
        Path("data/rounds-dataset/demo_a/rounds/round_1.parquet"),
        Path("data/rounds-dataset/demo_a/rounds/round_2.parquet"),
        Path("data/rounds-dataset/demo_b/rounds/round_3.parquet"),
    ]

    selected = select_stream_round_files(
        round_files,
        seed=123,
        random_single_round=True,
        skip_trained_rounds=False,
        trained_round_uids=set(),
    )

    assert len(selected) == 1
    assert selected == select_stream_round_files(
        round_files,
        seed=123,
        random_single_round=True,
        skip_trained_rounds=False,
        trained_round_uids=set(),
    )


def test_select_stream_round_files_skips_trained_before_random_choice():
    round_files = [
        Path("data/rounds-dataset/demo_a/rounds/round_1.parquet"),
        Path("data/rounds-dataset/demo_b/rounds/round_2.parquet"),
    ]

    selected = select_stream_round_files(
        round_files,
        seed=999,
        random_single_round=True,
        skip_trained_rounds=True,
        trained_round_uids={"demo_a::round_1"},
    )

    assert selected == [Path("data/rounds-dataset/demo_b/rounds/round_2.parquet")]
