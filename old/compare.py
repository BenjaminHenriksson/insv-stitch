#!/usr/bin/env python3
"""Compare our stitcher output against Insta360 Studio ground truth.

Stitches a frame from raw .insv using stitch.py, extracts the matching
frame from a ground truth .mp4, and generates visual + metric comparisons.

Usage:
    python compare.py --raw DCIM/Camera01/VID_20260318_180009_00_005.insv \
                      --gt x5-ground-truth/VID_20260318_180009_00_005.mp4 \
                      --frame 0
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def extract_gt_frame(gt_path: str, frame_num: int, width: int, height: int) -> np.ndarray:
    """Extract a single frame from the ground truth video."""
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", gt_path,
        "-vf", f"select=eq(n\\,{frame_num})",
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    raw = proc.stdout
    expected = width * height * 3
    if len(raw) != expected:
        raise ValueError(f"GT frame extraction failed: got {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()


def stitch_frame(raw_path: str, frame_num: int, width: int, height: int,
                 pb_path: str = None, fov: float = 200.0,
                 yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                 no_optical_flow: bool = False) -> np.ndarray:
    """Stitch a single frame from raw .insv using our pipeline."""
    from stitch import CalibrationData, FramePipeline, find_pb_file, get_video_info

    # Find calibration
    if not pb_path:
        pb_path = find_pb_file(raw_path)
    print(f"  Calibration: {pb_path}")
    calib = CalibrationData.from_pb(pb_path)

    # Get stream dimensions
    stream_w, stream_h, fps, _ = get_video_info(raw_path)
    print(f"  Raw video: {stream_w}x{stream_h} @ {fps:.2f} fps")

    # Extract raw frames
    frame_back = _extract_raw_frame(raw_path, 0, frame_num, stream_w, stream_h)
    frame_front = _extract_raw_frame(raw_path, 1, frame_num, stream_w, stream_h)

    # Build orientation rotation if specified
    R_orientation = None
    if abs(yaw) > 0.01 or abs(pitch) > 0.01 or abs(roll) > 0.01:
        R_orientation = _euler_to_rotation(
            np.radians(yaw), np.radians(pitch), np.radians(roll)
        )
        print(f"  Orientation correction: yaw={yaw:.2f} pitch={pitch:.2f} roll={roll:.2f}")

    # Build pipeline
    pipeline = FramePipeline(
        calib, stream_w, width, height,
        effective_fov=fov,
        use_optical_flow=not no_optical_flow,
        R_orientation=R_orientation,
        verbose=True,
    )

    return pipeline.process(frame_back, frame_front)


def _extract_raw_frame(path: str, stream_idx: int, frame_num: int,
                       width: int, height: int) -> np.ndarray:
    """Extract a single frame from one stream of an .insv file."""
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", path,
        "-map", f"0:v:{stream_idx}",
        "-vf", f"select=eq(n\\,{frame_num})",
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    raw = proc.stdout
    expected = width * height * 3
    if len(raw) != expected:
        raise ValueError(
            f"Stream {stream_idx} frame {frame_num}: got {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()


def _euler_to_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Convert Euler angles (yaw, pitch, roll) to 3x3 rotation matrix.

    Convention: Rz(roll) @ Rx(pitch) @ Ry(yaw)
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])

    return Rz @ Rx @ Ry


def compute_metrics(ours: np.ndarray, gt: np.ndarray) -> dict:
    """Compute comparison metrics between our output and ground truth."""
    ours_f = ours.astype(np.float64)
    gt_f = gt.astype(np.float64)

    # Mean Absolute Error
    mae = np.mean(np.abs(ours_f - gt_f))

    # PSNR
    mse = np.mean((ours_f - gt_f) ** 2)
    if mse < 1e-10:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(255.0 ** 2 / mse)

    # Phase correlation: measures global (x, y) shift
    # Convert to grayscale float for phaseCorrelate
    gray_ours = cv2.cvtColor(ours, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float64)
    (dx, dy), response = cv2.phaseCorrelate(gray_ours, gray_gt)

    h, w = ours.shape[:2]
    yaw_offset_deg = dx / w * 360.0
    pitch_offset_deg = dy / h * 180.0

    return {
        "mae": mae,
        "psnr": psnr,
        "mse": mse,
        "phase_dx_px": dx,
        "phase_dy_px": dy,
        "phase_response": response,
        "yaw_offset_deg": yaw_offset_deg,
        "pitch_offset_deg": pitch_offset_deg,
    }


def generate_outputs(ours: np.ndarray, gt: np.ndarray, metrics: dict,
                     output_dir: str):
    """Generate all comparison output files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save individual frames
    cv2.imwrite(f"{output_dir}/ours.jpg", ours, [cv2.IMWRITE_JPEG_QUALITY, 98])
    cv2.imwrite(f"{output_dir}/gt.jpg", gt, [cv2.IMWRITE_JPEG_QUALITY, 98])

    # Side-by-side (scale to fit if needed)
    h = min(ours.shape[0], gt.shape[0])
    w_ours = int(ours.shape[1] * h / ours.shape[0])
    w_gt = int(gt.shape[1] * h / gt.shape[0])
    ours_resized = cv2.resize(ours, (w_ours, h))
    gt_resized = cv2.resize(gt, (w_gt, h))

    # Add labels
    label_h = 40
    ours_labeled = np.zeros((h + label_h, w_ours, 3), dtype=np.uint8)
    gt_labeled = np.zeros((h + label_h, w_gt, 3), dtype=np.uint8)
    ours_labeled[label_h:] = ours_resized
    gt_labeled[label_h:] = gt_resized
    cv2.putText(ours_labeled, "OURS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(gt_labeled, "GROUND TRUTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    side_by_side = np.hstack([ours_labeled, gt_labeled])
    cv2.imwrite(f"{output_dir}/side_by_side.jpg", side_by_side,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Difference map (amplified)
    diff = cv2.absdiff(ours, gt)
    diff_amplified = np.clip(diff.astype(np.float32) * 5, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/diff.jpg", diff_amplified, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Heatmap of difference
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_heatmap = cv2.applyColorMap(
        np.clip(diff_gray.astype(np.float32) * 3, 0, 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    cv2.imwrite(f"{output_dir}/diff_heatmap.jpg", diff_heatmap,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Metrics text file
    with open(f"{output_dir}/metrics.txt", "w") as f:
        f.write("Ground Truth Comparison Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"MAE (Mean Absolute Error):  {metrics['mae']:.2f}\n")
        f.write(f"PSNR:                       {metrics['psnr']:.2f} dB\n")
        f.write(f"MSE:                        {metrics['mse']:.2f}\n\n")
        f.write("Phase Correlation (global shift):\n")
        f.write(f"  dx = {metrics['phase_dx_px']:.2f} px  (yaw  = {metrics['yaw_offset_deg']:.3f} deg)\n")
        f.write(f"  dy = {metrics['phase_dy_px']:.2f} px  (pitch = {metrics['pitch_offset_deg']:.3f} deg)\n")
        f.write(f"  response = {metrics['phase_response']:.4f}\n")

    print(f"\nOutputs saved to {output_dir}/")
    print(f"  ours.jpg, gt.jpg, side_by_side.jpg, diff.jpg, diff_heatmap.jpg, metrics.txt")


def get_gt_resolution(gt_path: str) -> tuple:
    """Get width, height from ground truth video."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", gt_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return int(stream["width"]), int(stream["height"])


def main():
    parser = argparse.ArgumentParser(
        description="Compare stitcher output against Insta360 Studio ground truth",
    )
    parser.add_argument("--raw", "-r", required=True, help="Raw .insv input file")
    parser.add_argument("--gt", "-g", required=True, help="Ground truth .mp4 file")
    parser.add_argument("--frame", "-f", type=int, default=0, help="Frame number (default: 0)")
    parser.add_argument("--output", "-o", default="output/compare",
                        help="Output directory (default: output/compare)")
    parser.add_argument("--pb", help="Calibration .pb file (auto-detected if omitted)")
    parser.add_argument("--fov", type=float, default=200.0,
                        help="Effective FOV per lens (default: 200)")
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="Yaw correction in degrees")
    parser.add_argument("--pitch", type=float, default=0.0,
                        help="Pitch correction in degrees")
    parser.add_argument("--roll", type=float, default=0.0,
                        help="Roll correction in degrees")
    parser.add_argument("--no-optical-flow", action="store_true",
                        help="Disable optical flow alignment")
    parser.add_argument("--scale", type=float,
                        help="Scale factor (default: match GT resolution)")

    args = parser.parse_args()

    # Resolve paths
    raw_path = os.path.abspath(args.raw)
    gt_path = os.path.abspath(args.gt)

    if not os.path.exists(raw_path):
        print(f"Error: raw file not found: {raw_path}")
        sys.exit(1)
    if not os.path.exists(gt_path):
        print(f"Error: ground truth file not found: {gt_path}")
        sys.exit(1)

    # Get GT resolution to match
    gt_w, gt_h = get_gt_resolution(gt_path)
    print(f"Ground truth resolution: {gt_w}x{gt_h}")

    out_w = gt_w
    out_h = gt_h
    if args.scale:
        out_w = int(gt_w * args.scale)
        out_h = int(gt_h * args.scale)
        print(f"Scaled output: {out_w}x{out_h}")

    # Step 1: Extract GT frame
    print(f"\nExtracting GT frame {args.frame}...")
    gt_frame = extract_gt_frame(gt_path, args.frame, gt_w, gt_h)
    if args.scale:
        gt_frame = cv2.resize(gt_frame, (out_w, out_h))
    print(f"  GT frame: {gt_frame.shape}, mean={gt_frame.mean():.1f}")

    # Step 2: Stitch our frame
    print(f"\nStitching frame {args.frame}...")
    our_frame = stitch_frame(
        raw_path, args.frame, out_w, out_h,
        pb_path=args.pb, fov=args.fov,
        yaw=args.yaw, pitch=args.pitch, roll=args.roll,
        no_optical_flow=args.no_optical_flow,
    )
    print(f"  Our frame: {our_frame.shape}, mean={our_frame.mean():.1f}")

    # Step 3: Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(our_frame, gt_frame)
    print(f"  MAE:   {metrics['mae']:.2f}")
    print(f"  PSNR:  {metrics['psnr']:.2f} dB")
    print(f"  Phase: dx={metrics['phase_dx_px']:.1f}px (yaw={metrics['yaw_offset_deg']:.3f}deg), "
          f"dy={metrics['phase_dy_px']:.1f}px (pitch={metrics['pitch_offset_deg']:.3f}deg)")

    # Step 4: Generate outputs
    generate_outputs(our_frame, gt_frame, metrics, args.output)


if __name__ == "__main__":
    main()
