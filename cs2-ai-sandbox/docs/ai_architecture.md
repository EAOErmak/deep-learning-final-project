# AI Architecture

## Purpose

This project uses offline Counter-Strike 2 demo datasets to prototype a modular AI stack without cheats, memory reading, reverse engineering, anti-cheat bypass, or online automation.

## Why `clean_play_ticks` Exists

`clean_play_ticks` is the main gameplay dataset layer for tactical modules.
It removes warmup and freeze periods so movement, aiming, enemy tracking, and decision logic can focus on active rounds.
It still keeps all players on each tick because multi-agent context matters.

## Why One Row Means One Player On One Tick

A single tick in the demo contains multiple player rows.
Each row describes one player state and one player input snapshot.
That layout is useful because:
- player policy targets stay attached to the player who produced them
- teammates and enemies remain available as context
- perspective-based samples can be built without losing round context

## Why Training Samples Use `perspective_player`

The same shared tick should produce multiple training views.
For one sample:
- `self` is one selected player
- teammates are same-side context
- enemies are opposite-side context
- target action is only the selected player's input

This is required for behavior cloning, sequence models, and RL transitions.

## Modules

### EnemyTracker
Input: `GameStateSequence`
Output: `EnemyTrackerOutput`

Uses visible/spotted information as model input.
Real enemy positions from demos are allowed as training targets, but hidden enemy positions are not passed directly into downstream decision features.

### BeliefState
Input: `GameState`, `EnemyTrackerOutput`
Output: `BeliefStateData`

Transforms enemy predictions into a simplified tactical belief representation.

### DecisionMaker
Input: `GameState`, `BeliefStateData`
Output: `DecisionOutput`

Produces high-level intent such as `buy`, `retake`, `defend_site`, or `fallback`.
It should not receive raw hidden enemy positions.

### Movement
Input: `GameState`, `BeliefStateData`, `DecisionOutput`
Output: `MovementOutput`

Handles locomotion intent, movement mode, and target movement direction.

### AimShoot
Input: `GameState`, `BeliefStateData`, `DecisionOutput`
Output: `AimShootOutput`

Handles aim deltas and shooting intent.

### Buy
Input: `GameState`
Output: `BuyOutput`

Works on freeze/buy-phase data and economy context.

### ActionCoordinator
Input: module outputs
Output: `ActionPlan`

Builds symbolic dry-run keyboard/mouse commands.
No real input execution happens here.

## AI/ML Stack

PyTorch is the main deep learning framework.
pandas and pyarrow are used for parquet dataset processing.
numpy is used for feature vectors and array targets.
scikit-learn can be used later for metrics, data splitting, and baseline utilities.
matplotlib, tqdm, and rich are intended for visualization, progress reporting, and readable console output.
gymnasium and stable-baselines3 are optional future tools for RL fine-tuning.
TensorFlow is not used in this project.

## Where LSTM, Attention, and RL Fit

- LSTM model: `cs2_ai/ml/models/enemy_tracker_lstm.py`
- Attention model: `cs2_ai/ml/models/aim_attention.py`
- RL / DQN model: `cs2_ai/ml/models/decision_dqn.py`

Rule-based modules remain active as the current default runtime path.
The ML layer is added in parallel so the project can transition gradually without breaking the dry-run architecture.

## Why DecisionMaker Does Not Receive Raw Enemy Positions

If decision logic sees exact hidden enemy positions from demos, it learns privileged information that would not exist in a realistic deployment.
That would break the intended separation between observation, tracking, and planning.

So the rule is:
- EnemyTracker may use real positions as offline target labels
- DecisionMaker should consume predictions and belief state instead

## Smoke Test

Run:

```powershell
python scripts/inspect_perspective_sample.py
python scripts/smoke_test_ai_architecture.py
```

If no parquet dataset exists, the scripts print a readable message telling you to run the parser/cleaner first.
If PyTorch is installed, the smoke test also verifies model imports, device selection, and dummy forward passes for the LSTM, attention, and DQN skeletons.
