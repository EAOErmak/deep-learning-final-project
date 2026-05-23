import argparse
import logging
import torch
from pathlib import Path

from cs2_ai.ml.training.train_route import RouteGRURegressionModel

logger = logging.getLogger(__name__)

class RouteRuntimeModel:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.history_len = 16
        self.xyz_normalization = {"x": 4000.0, "y": 4000.0, "z": 512.0}
        self.model = None
        self._load_checkpoint()

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Route checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Check metadata
        model_type = checkpoint.get("model_type", "")
        route_output_mode = checkpoint.get("route_output_mode", "")
        if "xyz" not in model_type and route_output_mode != "xyz":
            raise ValueError(f"Checkpoint {self.checkpoint_path} is not an xyz model. Found: {model_type}")
            
        self.history_len = checkpoint.get("history_len", 16)
        if "xyz_normalization" in checkpoint:
            self.xyz_normalization = checkpoint["xyz_normalization"]

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # Infer hidden_dim from gru.weight_ih_l0
        hidden_dim = 256
        if "gru.weight_ih_l0" in state_dict:
            hidden_dim = state_dict["gru.weight_ih_l0"].shape[0] // 3
            
        self.model = RouteGRURegressionModel(hidden_dim=hidden_dim).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        logger.info(f"Loaded RouteRuntimeModel from {self.checkpoint_path} | history_len={self.history_len} | hidden_dim={hidden_dim}")

    def _normalize_xyz(self, xyz: list[float]) -> list[float]:
        return [
            xyz[0] / self.xyz_normalization["x"],
            xyz[1] / self.xyz_normalization["y"],
            xyz[2] / self.xyz_normalization["z"],
        ]

    def _denormalize_xyz(self, norm_xyz: list[float]) -> list[float]:
        return [
            norm_xyz[0] * self.xyz_normalization["x"],
            norm_xyz[1] * self.xyz_normalization["y"],
            norm_xyz[2] * self.xyz_normalization["z"],
        ]

    def predict_next_xyz(self, history_xyz: list[list[float]], current_xyz: list[float], target_xyz: list[float]) -> list[float]:
        # Left-pad history
        padded_hist = history_xyz.copy()
        while len(padded_hist) < self.history_len:
            if len(padded_hist) > 0:
                padded_hist.insert(0, padded_hist[0])
            else:
                padded_hist.insert(0, current_xyz)
        
        if len(padded_hist) > self.history_len:
            padded_hist = padded_hist[-self.history_len:]
            
        norm_hist = [self._normalize_xyz(xyz) for xyz in padded_hist]
        norm_curr = self._normalize_xyz(current_xyz)
        norm_targ = self._normalize_xyz(target_xyz)
        
        hist_tensor = torch.tensor([norm_hist], dtype=torch.float32, device=self.device)
        curr_tensor = torch.tensor([norm_curr], dtype=torch.float32, device=self.device)
        targ_tensor = torch.tensor([norm_targ], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            pred_tensor = self.model(hist_tensor, curr_tensor, targ_tensor)
            
        pred_norm = pred_tensor.squeeze(0).cpu().tolist()
        return self._denormalize_xyz(pred_norm)

def snap_to_grid_xyz(pred_xyz: list[float]) -> list[float]:
    # Grid sizes: 25 xy, 16 z
    logger.warning("Route snapping unavailable; using raw predicted XYZ. Proper cell centering is not implemented in this helper.")
    return pred_xyz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-checkpoint", type=str, required=True)
    parser.add_argument("--current-x", type=float, default=0.0)
    parser.add_argument("--current-y", type=float, default=0.0)
    parser.add_argument("--current-z", type=float, default=0.0)
    parser.add_argument("--target-x", type=float, default=1000.0)
    parser.add_argument("--target-y", type=float, default=500.0)
    parser.add_argument("--target-z", type=float, default=0.0)
    parser.add_argument("--history", type=str, default="")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    model = RouteRuntimeModel(args.route_checkpoint)
    
    current_xyz = [args.current_x, args.current_y, args.current_z]
    target_xyz = [args.target_x, args.target_y, args.target_z]
    
    history_xyz = []
    if args.history:
        for point in args.history.split(";"):
            if point.strip():
                parts = point.split(",")
                history_xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
                
    pred = model.predict_next_xyz(history_xyz, current_xyz, target_xyz)
    snapped = snap_to_grid_xyz(pred)
    
    logger.info(f"Input current_xyz: {current_xyz}")
    logger.info(f"Input target_xyz: {target_xyz}")
    logger.info(f"Input history length: {len(history_xyz)}")
    logger.info(f"Predicted next_xyz: {pred}")
    logger.info(f"Snapped next_xyz: {snapped}")
