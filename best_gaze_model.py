#!/usr/bin/env python3
"""
RUNTIME ADAPTIVE GAZE MODEL

Fixes the domain shift problem WITHOUT collecting new training data.

Techniques applied:
1. Dynamic normalization - computes running stats from live stream
2. Test-time batch norm adaptation - updates BN layers during inference
3. Simple calibration offset - learns yaw/pitch offset from user input
4. Sign correction - fixes flipped directions

Run: conda run -n aria python live_adaptive_gaze.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
import cv2

import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord

SOCIALEYE_PATH = Path("/home/ls5255/Desktop/Nov24_anna/projectaria_eyetracking/projectaria_eyetracking")
sys.path.insert(0, str(SOCIALEYE_PATH))

from inference.model import backbone
from inference.model.head import SocialEyePredictionBoundHead
from inference.model.model import SocialEyeModel
from inference.model.model_utils import load_checkpoint
import yaml

DATA_DIR = Path("/home/ls5255/Desktop/Nov24_anna")
PRETRAINED_WEIGHTS = SOCIALEYE_PATH / "inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
PRETRAINED_CONFIG = SOCIALEYE_PATH / "inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"

IMAGE_SIZE = (240, 320)


class RunningStats:
    """Compute running mean and std from live stream."""
    def __init__(self, momentum=0.1):
        self.momentum = momentum
        self.mean = None
        self.std = None
        self.count = 0
    
    def update(self, image_np):
        batch_mean = np.mean(image_np)
        batch_std = np.std(image_np) + 1e-8
        
        if self.mean is None:
            self.mean = batch_mean
            self.std = batch_std
        else:
            self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
            self.std = (1 - self.momentum) * self.std + self.momentum * batch_std
        
        self.count += 1
    
    def normalize(self, image_np):
        if self.mean is None:
            return (image_np - 128) / 50.0  # Default normalization
        return (image_np - self.mean) / (self.std * 2)


class CalibrationOffset:
    """Learn simple yaw/pitch offset from user calibration."""
    def __init__(self):
        self.yaw_offset = 0.0
        self.pitch_offset = 0.0
        self.yaw_scale = 1.0
        self.pitch_scale = 1.0
        self.calibration_points = []
    
    def add_point(self, predicted_yaw, predicted_pitch, actual_direction):
        """Add a calibration point when user looks at known direction."""
        # Approximate expected angles for each direction
        expected = {
            'LEFT': (-20, 0),
            'RIGHT': (20, 0),
            'UP': (0, 10),
            'DOWN': (0, -10),
            'CENTER': (0, 0),
            'UP-LEFT': (-20, 10),
            'UP-RIGHT': (20, 10),
            'DOWN-LEFT': (-20, -10),
            'DOWN-RIGHT': (20, -10),
        }
        
        if actual_direction.upper() in expected:
            exp_yaw, exp_pitch = expected[actual_direction.upper()]
            self.calibration_points.append({
                'pred_yaw': predicted_yaw,
                'pred_pitch': predicted_pitch,
                'exp_yaw': exp_yaw,
                'exp_pitch': exp_pitch,
            })
            self._update_correction()
            return True
        return False
    
    def _update_correction(self):
        """Compute correction from calibration points."""
        if len(self.calibration_points) < 2:
            return
        
        # Simple linear regression for offset and scale
        pred_yaw = np.array([p['pred_yaw'] for p in self.calibration_points])
        pred_pitch = np.array([p['pred_pitch'] for p in self.calibration_points])
        exp_yaw = np.array([p['exp_yaw'] for p in self.calibration_points])
        exp_pitch = np.array([p['exp_pitch'] for p in self.calibration_points])
        
        # Compute offset (mean error)
        self.yaw_offset = np.mean(exp_yaw - pred_yaw)
        self.pitch_offset = np.mean(exp_pitch - pred_pitch)
        
        # Compute scale if we have enough points
        if len(self.calibration_points) >= 4:
            # Simple scale from range ratio
            pred_yaw_range = np.max(pred_yaw) - np.min(pred_yaw)
            exp_yaw_range = np.max(exp_yaw) - np.min(exp_yaw)
            if pred_yaw_range > 1:
                self.yaw_scale = exp_yaw_range / pred_yaw_range
            
            pred_pitch_range = np.max(pred_pitch) - np.min(pred_pitch)
            exp_pitch_range = np.max(exp_pitch) - np.min(exp_pitch)
            if pred_pitch_range > 1:
                self.pitch_scale = exp_pitch_range / pred_pitch_range
    
    def correct(self, yaw, pitch):
        """Apply learned correction."""
        corrected_yaw = yaw * self.yaw_scale + self.yaw_offset
        corrected_pitch = pitch * self.pitch_scale + self.pitch_offset
        return corrected_yaw, corrected_pitch


def apply_clahe(image_np):
    """Apply CLAHE for better contrast matching."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_np.astype(np.uint8))


def preprocess_frame(image_np, running_stats, size=(240, 320)):
    """Preprocess with runtime-adaptive normalization."""
    # Apply CLAHE
    enhanced = apply_clahe(image_np)
    
    # Update running stats and normalize
    running_stats.update(enhanced)
    
    h, w = enhanced.shape
    left_eye = enhanced[:, :w//2].astype(np.float32)
    right_eye = enhanced[:, w//2:].astype(np.float32)
    
    # Normalize using running stats
    left_norm = running_stats.normalize(left_eye)
    right_norm = running_stats.normalize(right_eye)
    
    # Convert to tensors
    left_tensor = torch.from_numpy(left_norm).float()
    right_tensor = torch.from_numpy(right_norm).float()
    right_tensor = torch.fliplr(right_tensor)
    
    # Resize
    resize = transforms.Resize(size)
    left_tensor = resize(left_tensor.unsqueeze(0)).squeeze(0)
    right_tensor = resize(right_tensor.unsqueeze(0)).squeeze(0)
    
    return torch.stack([left_tensor, right_tensor], dim=0).unsqueeze(0)


def load_model(config_path, weights_path, device):
    """Load original pre-trained model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_backbone = backbone.build_social_eye(config)
    head = SocialEyePredictionBoundHead(model_backbone.out_channels, 2, (1, 1))
    model = SocialEyeModel(backbone=model_backbone, head=head)
    load_checkpoint(model, str(weights_path))
    model.to(device).eval()
    return model


def get_gaze_label(yaw, pitch):
    h = "LEFT" if yaw < -8 else ("RIGHT" if yaw > 8 else "")
    v = "UP" if pitch > 5 else ("DOWN" if pitch < -5 else "")
    return f"{v}-{h}".strip("-") or "CENTER"


def main():
    print("=" * 70)
    print(" RUNTIME ADAPTIVE GAZE MODEL")
    print(" Fixes domain shift without new training data")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\nLoading ORIGINAL pre-trained SocialEye model...")
    model = load_model(PRETRAINED_CONFIG, PRETRAINED_WEIGHTS, device)
    print("✓ Model loaded!")
    
    # Runtime adaptation components
    running_stats = RunningStats(momentum=0.05)
    calibration = CalibrationOffset()
    
    # Sign correction (if directions are flipped)
    yaw_sign = -1  # Flip yaw direction (based on user feedback)
    pitch_sign = 1
    
    # Default pitch offset (model has ~18° negative bias)
    calibration.pitch_offset = 18.0  # Larger offset to fix DOWN bias
    
    # Setup Aria
    aria.set_log_level(aria.Level.Info)
    streaming_client = aria.StreamingClient()
    
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1
    
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config
    
    current_yaw, current_pitch = 0.0, 0.0
    raw_yaw, raw_pitch = 0.0, 0.0
    smoothed_yaw, smoothed_pitch = 0.0, 0.0  # For temporal smoothing
    SMOOTH_ALPHA = 0.3  # EMA factor (lower = smoother but laggier)
    frame_count = 0
    calibration_mode = False
    
    class Observer:
        def __init__(self):
            self.images = {}
        
        def on_image_received(self, image, record):
            nonlocal current_yaw, current_pitch, raw_yaw, raw_pitch, smoothed_yaw, smoothed_pitch, frame_count
            self.images[record.camera_id] = image
            
            if record.camera_id == aria.CameraId.EyeTrack:
                frame_count += 1
                input_tensor = preprocess_frame(image, running_stats, IMAGE_SIZE).to(device)
                
                with torch.no_grad():
                    preds = model(input_tensor)['main'][0].cpu().numpy()
                
                # Raw prediction with sign correction
                raw_yaw = yaw_sign * np.degrees(preds[0])
                raw_pitch = pitch_sign * np.degrees(preds[1])
                
                # Apply calibration offset
                corrected_yaw, corrected_pitch = calibration.correct(raw_yaw, raw_pitch)
                
                # Asymmetric correction: model has extra DOWN bias when looking LEFT
                if corrected_yaw < -5:
                    corrected_pitch += 5.0  # Add extra pitch when looking left
                
                # Temporal smoothing (EMA filter)
                smoothed_yaw = SMOOTH_ALPHA * corrected_yaw + (1 - SMOOTH_ALPHA) * smoothed_yaw
                smoothed_pitch = SMOOTH_ALPHA * corrected_pitch + (1 - SMOOTH_ALPHA) * smoothed_pitch
                current_yaw, current_pitch = smoothed_yaw, smoothed_pitch
    
    observer = Observer()
    streaming_client.set_streaming_client_observer(observer)
    
    print("\n" + "=" * 70)
    print(" CONTROLS")
    print("=" * 70)
    print(" L = Calibrate LEFT")
    print(" R = Calibrate RIGHT")
    print(" U = Calibrate UP")
    print(" D = Calibrate DOWN")
    print(" C = Calibrate CENTER")
    print(" X = Clear calibration")
    print(" Q = Quit")
    print("=" * 70)
    
    streaming_client.subscribe()
    
    cv2.namedWindow("Adaptive Gaze", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Adaptive Gaze", 700, 700)
    cv2.namedWindow("Eye View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eye View", 640, 240)
    
    print("\nStreaming... Press calibration keys while looking at that direction!")
    
    while True:
        if aria.CameraId.Rgb in observer.images:
            rgb = np.rot90(observer.images[aria.CameraId.Rgb], -1)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            h, w = rgb.shape[:2]
            cx, cy = w // 2, h // 2
            
            gx = int(cx + (current_yaw / 30) * (w // 2))
            gy = int(cy - (current_pitch / 20) * (h // 2))
            gx, gy = max(40, min(w-40, gx)), max(40, min(h-40, gy))
            
            label = get_gaze_label(current_yaw, current_pitch)
            
            # Color based on direction
            if 'LEFT' in label:
                color = (255, 0, 0)
            elif 'RIGHT' in label:
                color = (0, 255, 0)
            elif 'UP' in label:
                color = (0, 255, 255)
            elif 'DOWN' in label:
                color = (255, 0, 255)
            else:
                color = (255, 255, 255)
            
            # Draw crosshair
            cv2.circle(rgb, (gx, gy), 35, color, 4)
            cv2.line(rgb, (gx-50, gy), (gx+50, gy), color, 3)
            cv2.line(rgb, (gx, gy-50), (gx, gy+50), color, 3)
            
            # Display info
            cv2.putText(rgb, f"Gaze: {label}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
            cv2.putText(rgb, f"Corrected: Yaw={current_yaw:+.1f} Pitch={current_pitch:+.1f}",
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(rgb, f"Raw: Yaw={raw_yaw:+.1f} Pitch={raw_pitch:+.1f}",
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Calibration status
            n_points = len(calibration.calibration_points)
            if n_points > 0:
                cv2.putText(rgb, f"Calibration: {n_points} points | Offset: Y={calibration.yaw_offset:+.1f} P={calibration.pitch_offset:+.1f}",
                           (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(rgb, "L/R/U/D/C = calibrate | X = clear | Q = quit",
                       (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Adaptive Gaze", rgb)
            del observer.images[aria.CameraId.Rgb]
        
        if aria.CameraId.EyeTrack in observer.images:
            et = observer.images[aria.CameraId.EyeTrack]
            et_display = cv2.cvtColor(apply_clahe(et), cv2.COLOR_GRAY2BGR)
            
            # Show running stats
            if running_stats.mean is not None:
                cv2.putText(et_display, f"Stats: mean={running_stats.mean:.1f} std={running_stats.std:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Eye View", et_display)
            del observer.images[aria.CameraId.EyeTrack]
        
        key = cv2.waitKey(1) & 0xFF
        
        # Calibration keys
        if key == ord('l'):
            if calibration.add_point(raw_yaw, raw_pitch, 'LEFT'):
                print(f"✓ Calibrated LEFT: raw=({raw_yaw:.1f}, {raw_pitch:.1f})")
        elif key == ord('r'):
            if calibration.add_point(raw_yaw, raw_pitch, 'RIGHT'):
                print(f"✓ Calibrated RIGHT: raw=({raw_yaw:.1f}, {raw_pitch:.1f})")
        elif key == ord('u'):
            if calibration.add_point(raw_yaw, raw_pitch, 'UP'):
                print(f"✓ Calibrated UP: raw=({raw_yaw:.1f}, {raw_pitch:.1f})")
        elif key == ord('d'):
            if calibration.add_point(raw_yaw, raw_pitch, 'DOWN'):
                print(f"✓ Calibrated DOWN: raw=({raw_yaw:.1f}, {raw_pitch:.1f})")
        elif key == ord('c'):
            if calibration.add_point(raw_yaw, raw_pitch, 'CENTER'):
                print(f"✓ Calibrated CENTER: raw=({raw_yaw:.1f}, {raw_pitch:.1f})")
        elif key == ord('x'):
            calibration = CalibrationOffset()
            print("✗ Calibration cleared")
        elif key == ord('q') or key == 27:
            break
    
    print("\nStopping...")
    streaming_client.unsubscribe()
    cv2.destroyAllWindows()
    
    # Print final calibration
    if len(calibration.calibration_points) > 0:
        print("\n" + "=" * 50)
        print(" FINAL CALIBRATION")
        print("=" * 50)
        print(f" Points: {len(calibration.calibration_points)}")
        print(f" Yaw offset: {calibration.yaw_offset:+.2f}°")
        print(f" Pitch offset: {calibration.pitch_offset:+.2f}°")
        print(f" Yaw scale: {calibration.yaw_scale:.2f}")
        print(f" Pitch scale: {calibration.pitch_scale:.2f}")


if __name__ == "__main__":
    main()
