import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from cs2_ai.runtime.route_runtime import RouteRuntimeModel, snap_to_grid_xyz
from cs2_ai.features.movement_features import MovementFeatureExtractor
from cs2_ai.schemas.game_state import GameState, PlayerState, GameStateSequence

@pytest.fixture
def mock_checkpoint(tmp_path):
    ckpt_path = tmp_path / "mock_route.pt"
    # Create empty file
    ckpt_path.write_bytes(b"")
    return str(ckpt_path)

@patch("cs2_ai.runtime.route_runtime.torch.load")
@patch("cs2_ai.runtime.route_runtime.RouteGRURegressionModel")
def test_route_runtime_model_load(mock_model_class, mock_load, mock_checkpoint):
    mock_load.return_value = {
        "model_type": "route_gru_xyz",
        "route_output_mode": "xyz",
        "history_len": 8,
        "xyz_normalization": {"x": 4000.0, "y": 4000.0, "z": 512.0},
        "model_state_dict": {"gru.weight_ih_l0": MagicMock(shape=(768, 3))} # 256 * 3
    }
    
    mock_model_instance = MagicMock()
    # Return shape [1, 3] from model forward
    mock_model_instance.return_value = MagicMock()
    mock_model_instance.return_value.squeeze.return_value.cpu.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_model_class.return_value.to.return_value = mock_model_instance
    
    model = RouteRuntimeModel(mock_checkpoint)
    
    assert model.history_len == 8
    
    # Test predict_next_xyz
    history = [[10.0, 10.0, 0.0], [20.0, 20.0, 0.0]]
    current = [30.0, 30.0, 0.0]
    target = [100.0, 100.0, 0.0]
    
    pred = model.predict_next_xyz(history, current, target)
    
    assert len(pred) == 3
    # 0.1 * 4000 = 400, 0.2 * 4000 = 800, 0.3 * 512 = 153.6
    assert np.allclose(pred, [400.0, 800.0, 153.6])
    
    # Check history padding occurred (called with right shape)
    call_args = mock_model_instance.call_args[0]
    hist_tensor = call_args[0]
    assert hist_tensor.shape[1] == 8 # Sequence length

def test_snap_to_grid_xyz():
    pred = [123.45, -67.89, 45.0]
    snapped = snap_to_grid_xyz(pred)
    # The current snapping logic is disabled as requested, it returns the raw output
    assert snapped == pred

def test_movement_feature_extractor_route_target():
    extractor = MovementFeatureExtractor(seq_len=1)
    
    player = PlayerState(
        steamid=123, name="test", team_num=2,
        position=[100.0, 100.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        health=100.0, armor=100.0, has_helmet=False, is_alive=True, money=0.0,
        weapon="ak47", weapon_id=7, ammo=30.0, total_ammo=90.0,
        pitch=0.0, yaw=0.0, is_scoped=False, is_walking=False, is_airborne=False,
        duck_amount=0.0, ducking=False, shots_fired=0, flash_duration=0.0,
        spotted=False, last_place_name="Mid", in_bomb_zone=False, in_buy_zone=False,
        which_bomb_zone=0
    )
    
    from cs2_ai.schemas.game_state import PlayerInputState, RoundState, BombState
    
    input_state = PlayerInputState(
        forward=False, back=False, left=False, right=False, fire=False,
        rightclick=False, reload=False, use=False, zoom=False, walk=False,
        usercmd_mouse_dx=0.0, usercmd_mouse_dy=0.0, usercmd_forward_move=0.0, usercmd_left_move=0.0
    )
    round_state = RoundState(
        tick=1, round_number=1, round_start_time=0.0, round_in_progress=True,
        is_freeze_period=False, is_warmup_period=False, game_phase=2,
        round_win_status=0, round_win_reason=0, ct_losing_streak=0, t_losing_streak=0
    )
    bomb_state = BombState(is_bomb_planted=False, is_bomb_dropped=False, bomb_position=None)
    
    state = GameState(
        tick=1, perspective_steamid=123, self_player=player, self_input=input_state,
        teammates=[], enemies=[], round=round_state, bomb=bomb_state
    )
    sequence = GameStateSequence(perspective_steamid=123, states=[state])
    
    # With route_target=None
    features_none = extractor.extract(sequence, route_target=None)
    # With route_target=[0,0,0]
    features_zero = extractor.extract(sequence, route_target=[0.0, 0.0, 0.0])
    
    # Behavior must remain identical if no route target is passed
    assert np.allclose(features_none, features_zero)
    
    # With explicit route_target
    features_target = extractor.extract(sequence, route_target=[500.0, 500.0, 0.0])
    
    # Features should differ (specifically target_rel features)
    assert not np.allclose(features_none, features_target)
