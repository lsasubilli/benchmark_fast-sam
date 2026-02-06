#!/usr/bin/env python3
"""
FastSAM Benchmark Script

Measures:
1. Model size in memory (parameters, MB)
2. FLOPs (floating point operations)
3. Inference speed (ms per frame, FPS)

Runs on live Aria eye tracking stream WITHOUT gaze CNN or visualization.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import cv2
import time
from collections import deque

# Add FastSAM path
FASTSAM_PATH = Path("/home/ls5255/FastSAM")
sys.path.insert(0, str(FASTSAM_PATH))

from fastsam import FastSAM, FastSAMPrompt
from PIL import Image

import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord

# FastSAM weights
FASTSAM_WEIGHTS = FASTSAM_PATH / "weights" / "FastSAM-x.pt"


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def estimate_flops(model, input_size=(1, 3, 1024, 1024)):
    """
    Estimate FLOPs using thop library if available, otherwise use manual estimation.
    """
    try:
        from thop import profile, clever_format
        
        # Create dummy input
        dummy_input = torch.randn(input_size).to(next(model.model.parameters()).device)
        
        flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return flops, flops_str
    except ImportError:
        print("  (thop not installed, using rough estimation)")
        # Rough estimation based on YOLOv8x-seg architecture
        # ~68.2M params, typically ~258 GFLOPs for 640x640
        # Scale for 1024x1024: (1024/640)^2 * 258 ≈ 660 GFLOPs
        estimated_gflops = 258 * (input_size[2] / 640) ** 2
        return estimated_gflops * 1e9, f"{estimated_gflops:.1f} GFLOPs (estimated)"
    except Exception as e:
        print(f"  FLOPs calculation error: {e}")
        return None, "N/A"


def benchmark_inference(model, image_np, device, num_warmup=5, num_iterations=50):
    """
    Benchmark pure inference time (no visualization).
    
    Returns: dict with timing statistics
    """
    # Convert to RGB PIL Image
    if len(image_np.shape) == 2:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image_np
    
    image_pil = Image.fromarray(image_rgb)
    
    # Warmup runs
    print(f"  Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            results = model(image_pil, device=device, retina_masks=True, 
                          imgsz=1024, conf=0.25, iou=0.9)
            if results:
                prompt = FastSAMPrompt(image_pil, results, device=device)
                masks = prompt.everything_prompt()
    
    # Synchronize CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark runs
    print(f"  Benchmarking ({num_iterations} iterations)...")
    times = []
    
    for i in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        with torch.no_grad():
            results = model(image_pil, device=device, retina_masks=True, 
                          imgsz=1024, conf=0.25, iou=0.9)
            if results:
                prompt = FastSAMPrompt(image_pil, results, device=device)
                masks = prompt.everything_prompt()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    
    times = np.array(times)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'fps': 1000 / np.mean(times),
        'times': times
    }


def benchmark_live_stream(model, device, duration_seconds=10):
    """
    Benchmark on live Aria stream.
    """
    print(f"\n  Running live stream benchmark for {duration_seconds} seconds...")
    
    # Aria Setup
    aria.set_log_level(aria.Level.Warning)  # Reduce logging
    streaming_client = aria.StreamingClient()
    
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.EyeTrack
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1
    
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config
    
    inference_times = []
    frame_count = 0
    start_time = None
    
    class BenchmarkObserver:
        def __init__(self):
            self.latest_image = None
        
        def on_image_received(self, image: np.array, record: ImageDataRecord):
            nonlocal frame_count, start_time, inference_times
            
            if record.camera_id == aria.CameraId.EyeTrack:
                if start_time is None:
                    start_time = time.time()
                
                # Convert image
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    image_rgb = image
                
                image_pil = Image.fromarray(image_rgb)
                
                # Time inference only
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                t_start = time.perf_counter()
                
                with torch.no_grad():
                    results = model(image_pil, device=device, retina_masks=True,
                                  imgsz=1024, conf=0.25, iou=0.9)
                    if results:
                        prompt = FastSAMPrompt(image_pil, results, device=device)
                        masks = prompt.everything_prompt()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                t_end = time.perf_counter()
                inference_times.append((t_end - t_start) * 1000)
                frame_count += 1
    
    observer = BenchmarkObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()
    
    # Run for specified duration
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        time.sleep(0.01)
    
    streaming_client.unsubscribe()
    
    if len(inference_times) > 0:
        times = np.array(inference_times)
        actual_duration = time.time() - start_time if start_time else duration_seconds
        
        return {
            'frames_processed': frame_count,
            'actual_duration': actual_duration,
            'effective_fps': frame_count / actual_duration,
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
            'inference_fps': 1000 / np.mean(times),
        }
    else:
        return None


def main():
    print("=" * 70)
    print(" FastSAM BENCHMARK")
    print(" Model Size, FLOPs, and Inference Speed")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load model
    print(f"\nLoading FastSAM from {FASTSAM_WEIGHTS}...")
    model = FastSAM(str(FASTSAM_WEIGHTS))
    print("✓ Model loaded!")
    
    # ==================== MODEL SIZE ====================
    print("\n" + "=" * 70)
    print(" MODEL SIZE")
    print("=" * 70)
    
    total_params, trainable_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    
    print(f"  Total parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Model size (memory):  {model_size_mb:.2f} MB")
    
    # ==================== FLOPs ====================
    print("\n" + "=" * 70)
    print(" FLOPs (Floating Point Operations)")
    print("=" * 70)
    
    flops, flops_str = estimate_flops(model, input_size=(1, 3, 1024, 1024))
    print(f"  FLOPs (1024x1024 input): {flops_str}")
    
    # Also estimate for smaller input
    flops_640, flops_640_str = estimate_flops(model, input_size=(1, 3, 640, 640))
    print(f"  FLOPs (640x640 input):   {flops_640_str}")
    
    # ==================== OFFLINE BENCHMARK ====================
    print("\n" + "=" * 70)
    print(" INFERENCE SPEED (Offline - Sample Image)")
    print("=" * 70)
    
    # Load a sample image
    sample_image_path = "/home/ls5255/et_images_jan23/frame_0009_camera_3.png"
    sample_image = cv2.imread(sample_image_path)
    
    if sample_image is not None:
        print(f"  Image: {sample_image_path}")
        print(f"  Size: {sample_image.shape[1]}x{sample_image.shape[0]}")
        
        stats = benchmark_inference(model, sample_image, device, 
                                   num_warmup=5, num_iterations=50)
        
        print(f"\n  Results (50 iterations):")
        print(f"  ─────────────────────────────────────")
        print(f"  Mean inference time:   {stats['mean_ms']:.2f} ms")
        print(f"  Std deviation:         {stats['std_ms']:.2f} ms")
        print(f"  Min time:              {stats['min_ms']:.2f} ms")
        print(f"  Max time:              {stats['max_ms']:.2f} ms")
        print(f"  Median time:           {stats['median_ms']:.2f} ms")
        print(f"  ─────────────────────────────────────")
        print(f"  FPS (theoretical):     {stats['fps']:.1f}")
    else:
        print(f"  ⚠ Could not load sample image: {sample_image_path}")
    
    # ==================== LIVE STREAM BENCHMARK ====================
    print("\n" + "=" * 70)
    print(" INFERENCE SPEED (Live Aria Stream)")
    print("=" * 70)
    
    try:
        live_stats = benchmark_live_stream(model, device, duration_seconds=10)
        
        if live_stats:
            print(f"\n  Results (live stream, {live_stats['actual_duration']:.1f}s):")
            print(f"  ─────────────────────────────────────")
            print(f"  Frames processed:      {live_stats['frames_processed']}")
            print(f"  Effective FPS:         {live_stats['effective_fps']:.1f}")
            print(f"  Mean inference time:   {live_stats['mean_ms']:.2f} ms")
            print(f"  Std deviation:         {live_stats['std_ms']:.2f} ms")
            print(f"  Min time:              {live_stats['min_ms']:.2f} ms")
            print(f"  Max time:              {live_stats['max_ms']:.2f} ms")
            print(f"  Median time:           {live_stats['median_ms']:.2f} ms")
            print(f"  ─────────────────────────────────────")
            print(f"  Pure inference FPS:    {live_stats['inference_fps']:.1f}")
        else:
            print("  ⚠ No frames received from stream")
    except Exception as e:
        print(f"  ⚠ Live stream benchmark failed: {e}")
        print("    (Make sure Aria streaming is started)")
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"  Model:              FastSAM-x (YOLOv8x-seg backbone)")
    print(f"  Parameters:         {total_params/1e6:.2f}M")
    print(f"  Model Size:         {model_size_mb:.2f} MB")
    print(f"  FLOPs (1024x1024):  {flops_str}")
    if sample_image is not None:
        print(f"  Inference Time:     {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
        print(f"  FPS:                {stats['fps']:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
