#!/usr/bin/env python3
"""Dual fisheye to equirectangular stitcher using OpenCV.

Uses per-lens camera calibration from .pb protobuf files for accurate
projection with distance-transform blending in the overlap zone.
Supports 360 cameras with dual fisheye lenses (tested with X5).
"""

import argparse
import base64
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CalibrationData
# ---------------------------------------------------------------------------

@dataclass
class LensCalibration:
    """Per-lens intrinsic and extrinsic calibration."""
    fx: float
    fy: float
    cx: float  # in per-lens (local) coordinates
    cy: float
    yaw_deg: float
    pitch_deg: float
    half_fov_deg: float
    rotation: np.ndarray  # Rodrigues vector (3,)
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


@dataclass
class CalibrationData:
    """Parsed calibration for both lenses."""
    lens_back: LensCalibration  # stream 0
    lens_front: LensCalibration  # stream 1
    ref_width: int = 10752
    ref_height: int = 5376

    @classmethod
    def from_pb(cls, pb_path: str) -> "CalibrationData":
        """Parse calibration from .pb protobuf file."""
        data = Path(pb_path).read_bytes()
        raw_text = data.decode("latin-1")

        # The detailed calibration is inside base64-encoded blocks within the protobuf.
        # Decode all base64 blocks and search within them.
        search_texts = [raw_text]
        b64_blocks = re.findall(r"[A-Za-z0-9+/]{50,}={0,2}", raw_text)
        for block in b64_blocks:
            try:
                padded = block + "=" * (4 - len(block) % 4) if len(block) % 4 else block
                decoded = base64.b64decode(padded).decode("utf-8", errors="ignore")
                search_texts.append(decoded)
            except Exception:
                pass

        full_text = "\n".join(search_texts)

        # Find the simple calibration first (for ref_w, ref_h)
        simple_match = re.search(r"n2_([\d._-]+)", full_text)
        if not simple_match:
            raise ValueError(f"No calibration found in {pb_path}")

        simple_parts = simple_match.group(1).split("_")
        ref_w = int(simple_parts[12])
        ref_h = int(simple_parts[13])

        # Find all detailed calibration strings: "2_<float>_<float>_..."
        # with enough fields to contain both lenses
        calib_matches = re.findall(
            r"2_[\d.]+_[\d.]+_[\d.]+_[\d._+-]+", full_text)

        # Pick the longest match (most fields = most detailed calibration)
        best_match = None
        best_fields = 0
        for m in calib_matches:
            fields = m.split("_")
            if len(fields) > best_fields and len(fields) >= 40:
                best_fields = len(fields)
                best_match = m

        if best_match:
            fields = best_match.split("_")
            print(f"  Using Tier calibration with {len(fields)} fields")

            # Split into two lens halves at the midpoint marker "2.000000"
            # First field is "2" (version prefix), then "2.000000" starts lens 1
            # Find the second "2.000000" which starts lens 2
            second_start = None
            for i in range(2, len(fields)):
                if fields[i] == "2.000000":
                    second_start = i
                    break

            if second_start:
                f1 = fields[1:second_start]     # lens 1 fields (starting with "2.000000")
                f2 = fields[second_start:]       # lens 2 fields (starting with "2.000000")
                lens_back = cls._parse_detailed_lens(f1, ref_w, is_back=True)
                lens_front = cls._parse_detailed_lens(f2, ref_w, is_back=False)
                return cls(lens_back=lens_back, lens_front=lens_front,
                           ref_width=ref_w, ref_height=ref_h)

        # Fallback to simple calibration
        print("Warning: Using simple calibration (no distortion coefficients)")
        lens_back = LensCalibration(
            fx=0, fy=0,
            cx=float(simple_parts[1]), cy=float(simple_parts[2]),
            yaw_deg=float(simple_parts[3]), pitch_deg=float(simple_parts[4]),
            half_fov_deg=float(simple_parts[5]),
            rotation=np.zeros(3),
        )
        lens_front = LensCalibration(
            fx=0, fy=0,
            cx=float(simple_parts[7]) - ref_w / 2, cy=float(simple_parts[8]),
            yaw_deg=float(simple_parts[9]), pitch_deg=float(simple_parts[10]),
            half_fov_deg=float(simple_parts[11]),
            rotation=np.zeros(3),
        )
        return cls(lens_back=lens_back, lens_front=lens_front,
                   ref_width=ref_w, ref_height=ref_h)

    @classmethod
    def _parse_detailed_lens(cls, fields: list, ref_w: int, is_back: bool) -> LensCalibration:
        """Parse detailed per-lens calibration fields.

        Field layout (fields list starts with "2.000000"):
        [0]  calib_version = 2.000000
        [1]  fx, [2] fy, [3] cx, [4] cy
        [5]  yaw, [6] pitch, [7] half_fov
        [8]  rot_x, [9] rot_y, [10] rot_z
        [11] k1, [12] k2, [13] k3, [14] k4  (Tier 3)
        [15] zero
        [16] p1, [17] p2
        [18-21] s1, s2, s3, s4
        [22-23] tauX, tauY
        [24] ref_w, [25] ref_h, [26] unknown

        Tier 2 (shorter): k1,k2,k3,p1,p2 (no k4)
        """
        f = [float(x) for x in fields]
        n = len(f)

        # fields[0] = 2.000000 (version)
        fx = f[1]
        fy = f[2]
        cx_ref = f[3]
        cy = f[4]
        yaw = f[5]
        pitch = f[6]
        half_fov = f[7]
        rot = np.array([f[8], f[9], f[10]])

        # Determine tier by field count (per lens, excluding trailing ref_w/h/unk)
        # Tier 2: ~18-19 fields, Tier 3: ~27 fields
        if n >= 24:
            # Tier 3: k1,k2,k3,k4,zero,p1,p2,s1,s2,s3,s4,tau1,tau2,...
            k1, k2, k3, k4 = f[11], f[12], f[13], f[14]
            # f[15] is zero
            p1, p2 = f[16], f[17]
        elif n >= 16:
            # Tier 2: k1,k2,k3,p1,p2,...
            k1, k2, k3 = f[11], f[12], f[13]
            k4 = 0.0
            p1, p2 = f[14], f[15]
        else:
            k1 = k2 = k3 = k4 = p1 = p2 = 0.0

        # Convert cx from reference frame to per-lens local coordinates
        cx_local = cx_ref if is_back else cx_ref - ref_w / 2

        return LensCalibration(
            fx=fx, fy=fy, cx=cx_local, cy=cy,
            yaw_deg=yaw, pitch_deg=pitch, half_fov_deg=half_fov,
            rotation=rot,
            k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2,
        )

    def scale_to(self, stream_width: int) -> "CalibrationData":
        """Scale calibration to match actual stream resolution."""
        per_lens_ref = self.ref_width / 2  # 5376
        scale = stream_width / per_lens_ref

        def scale_lens(lens: LensCalibration) -> LensCalibration:
            return LensCalibration(
                fx=lens.fx * scale, fy=lens.fy * scale,
                cx=lens.cx * scale, cy=lens.cy * scale,
                yaw_deg=lens.yaw_deg, pitch_deg=lens.pitch_deg,
                half_fov_deg=lens.half_fov_deg,
                rotation=lens.rotation.copy(),
                k1=lens.k1, k2=lens.k2, k3=lens.k3, k4=lens.k4,
                p1=lens.p1, p2=lens.p2,
            )

        return CalibrationData(
            lens_back=scale_lens(self.lens_back),
            lens_front=scale_lens(self.lens_front),
            ref_width=self.ref_width, ref_height=self.ref_height,
        )

    def print_summary(self):
        """Print calibration summary."""
        for name, lens in [("Back (stream 0)", self.lens_back),
                           ("Front (stream 1)", self.lens_front)]:
            print(f"  {name}:")
            print(f"    fx={lens.fx:.2f} fy={lens.fy:.2f} cx={lens.cx:.2f} cy={lens.cy:.2f}")
            print(f"    yaw={lens.yaw_deg:.3f}° pitch={lens.pitch_deg:.3f}° half_fov={lens.half_fov_deg:.3f}°")
            print(f"    rot={lens.rotation}")
            print(f"    k1={lens.k1:.6f} k2={lens.k2:.6f} k3={lens.k3:.6f} k4={lens.k4:.6f}")
            print(f"    p1={lens.p1:.6f} p2={lens.p2:.6f}")


# ---------------------------------------------------------------------------
# FisheyeRemapper
# ---------------------------------------------------------------------------

class FisheyeRemapper:
    """Builds remap tables for fisheye → equirectangular projection."""

    def __init__(self, calib: CalibrationData, input_size: int,
                 out_w: int, out_h: int, effective_fov_deg: float = 200.0,
                 R_orientation: np.ndarray = None):
        self.calib = calib
        self.input_size = input_size  # 3840 for video
        self.out_w = out_w
        self.out_h = out_h
        self.effective_half_fov = np.radians(effective_fov_deg / 2)
        self.R_orientation = R_orientation  # optional gyro/manual orientation correction

        # Scale calibration to input resolution
        self.scaled = calib.scale_to(input_size)

        # Build rotation matrices
        # Stream 0 (back lens) maps to equirect edges (lon=±π) → Ry(180°)
        # Stream 1 (front lens) maps to equirect center (lon=0) → identity
        self.R_back = self._build_rotation(self.scaled.lens_back, is_back=True)
        self.R_front = self._build_rotation(self.scaled.lens_front, is_back=False)

        # Build remap tables
        self._build_tables()

    def _build_rotation(self, lens: LensCalibration, is_back: bool) -> np.ndarray:
        """Build 3x3 rotation matrix for a lens."""
        # Base rotation: back lens faces -Z, front faces +Z
        if is_back:
            R_base = self._Ry(np.pi)  # 180° around Y
        else:
            R_base = np.eye(3)

        # Yaw/pitch correction (small angles)
        yaw_rad = np.radians(lens.yaw_deg)
        pitch_rad = np.radians(lens.pitch_deg)
        R_yaw = self._Ry(yaw_rad)
        R_pitch = self._Rx(pitch_rad)

        # Rodrigues rotation from calibration
        if np.linalg.norm(lens.rotation) > 1e-9:
            R_rod, _ = cv2.Rodrigues(lens.rotation)
        else:
            R_rod = np.eye(3)

        return R_rod @ R_pitch @ R_yaw @ R_base

    @staticmethod
    def _Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def _Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _build_tables(self):
        """Build remap tables for both lenses."""
        h, w = self.out_h, self.out_w

        # Equirectangular pixel grid
        x_eq = np.arange(w, dtype=np.float32)
        y_eq = np.arange(h, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_eq, y_eq)

        # Equirectangular → spherical
        lon = (x_grid / w) * 2 * np.pi - np.pi      # [-π, π]
        lat = (y_grid / h) * np.pi - np.pi / 2       # [-π/2, π/2]

        # Spherical → 3D unit vectors
        cos_lat = np.cos(lat)
        X = cos_lat * np.sin(lon)
        Y = -np.sin(lat)
        Z = cos_lat * np.cos(lon)

        # Stack into (H, W, 3) and reshape for matmul
        P = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)

        # Apply orientation correction (gyro/manual)
        # This rotates the viewing directions to compensate for camera tilt
        if self.R_orientation is not None:
            P = np.einsum("ij,hwj->hwi", self.R_orientation, P)

        # Process each lens
        self.map_back_x, self.map_back_y, self.mask_back = \
            self._project_lens(P, self.R_back, self.scaled.lens_back)
        self.map_front_x, self.map_front_y, self.mask_front = \
            self._project_lens(P, self.R_front, self.scaled.lens_front)

        # Overlap mask
        self.overlap_mask = self.mask_back & self.mask_front

    def _project_lens(self, P: np.ndarray, R: np.ndarray,
                      lens: LensCalibration):
        """Project 3D points to fisheye pixel coordinates for one lens.

        Returns: map_x, map_y (float32), valid_mask (bool)
        """
        h, w = P.shape[:2]

        # Transform to camera coordinates: Pc = R^T @ P
        R_inv = R.T
        Pc = np.einsum("ij,hwj->hwi", R_inv, P)

        Pc_x = Pc[..., 0]
        Pc_y = Pc[..., 1]
        Pc_z = Pc[..., 2]

        # Angle from optical axis
        r_xy = np.sqrt(Pc_x**2 + Pc_y**2)
        theta = np.arctan2(r_xy, Pc_z)
        # Negate Pc_y to convert from world Y-up to image Y-down (OpenCV convention)
        phi = np.arctan2(-Pc_y, Pc_x)

        # Valid mask: within effective FOV
        # Note: no Pc_z > 0 check. Fisheye covers beyond 180° (theta > 90°).
        valid = theta < self.effective_half_fov

        # Equidistant fisheye projection
        # f_equi derived from: at the image circle edge, r_pixels = f * theta_max
        # Use the lens center to estimate the inscribed circle radius
        radius = min(lens.cx, lens.cy,
                     self.input_size - lens.cx, self.input_size - lens.cy)
        f_equi = radius / self.effective_half_fov

        r_pix = f_equi * theta

        # Pixel coordinates
        map_x = (lens.cx + r_pix * np.cos(phi)).astype(np.float32)
        map_y = (lens.cy + r_pix * np.sin(phi)).astype(np.float32)

        # Invalidate out-of-bounds pixels
        oob = (map_x < 0) | (map_x >= self.input_size) | \
              (map_y < 0) | (map_y >= self.input_size)
        valid = valid & ~oob

        # Set invalid pixels to -1
        map_x[~valid] = -1
        map_y[~valid] = -1

        return map_x, map_y, valid

    def create_blend_mask(self) -> np.ndarray:
        """Create blend mask: 0=use back, 1=use front.

        - Back only: 0
        - Front only: 1
        - Overlap: blend weight based on distance from each lens's
          validity edge (distance transform)
        - Neither: 0
        """
        h, w = self.out_h, self.out_w
        mask = np.zeros((h, w), dtype=np.float32)

        back_only = self.mask_back & ~self.mask_front
        front_only = self.mask_front & ~self.mask_back
        overlap = self.mask_back & self.mask_front

        mask[front_only] = 1.0

        if overlap.any():
            # For each pixel in the overlap, compute distance to the
            # nearest non-valid pixel for each lens (distance transform).
            # Larger distance = more reliable (further from edge).
            dist_back = cv2.distanceTransform(
                self.mask_back.astype(np.uint8), cv2.DIST_L2, 5)
            dist_front = cv2.distanceTransform(
                self.mask_front.astype(np.uint8), cv2.DIST_L2, 5)

            total = dist_back + dist_front
            # blend=1 (use front) when dist_front >> dist_back
            blend = np.where(total > 0, dist_front / total, 0.5)
            mask[overlap] = blend[overlap]

        return mask







# ---------------------------------------------------------------------------
# FramePipeline
# ---------------------------------------------------------------------------

class FramePipeline:
    """Orchestrates per-frame stitching pipeline."""

    def __init__(self, calib: CalibrationData, input_size: int,
                 out_w: int = 7680, out_h: int = 3840,
                 effective_fov: float = 200.0,
                 use_optical_flow: bool = True, verbose: bool = False,
                 R_orientation: np.ndarray = None):
        self.verbose = verbose
        self.use_optical_flow = use_optical_flow

        if verbose:
            print(f"Building remap tables ({out_w}x{out_h}, FOV={effective_fov})...")
        t0 = time.time()
        self.remapper = FisheyeRemapper(
            calib, input_size, out_w, out_h, effective_fov,
            R_orientation=R_orientation,
        )
        if verbose:
            print(f"  Remap tables built in {time.time() - t0:.1f}s")
            print("Creating blend mask...")
        self.blend_mask = self.remapper.create_blend_mask()

        # Precompute overlap strip regions for optical flow / seam finding
        self._overlap_regions = self._find_overlap_regions()

        self.out_w = out_w
        self.out_h = out_h
        self.frame_count = 0

    def _find_overlap_regions(self):
        """Find the two seam strip regions from the blend mask transitions.

        The blend mask transitions from 0 (back) to 1 (front) at two locations.
        These transition zones are where we need optical flow + seam finding.
        """
        # Sample the blend mask at the equator (middle row)
        mid_row = self.blend_mask[self.blend_mask.shape[0] // 2]

        # Find columns where the mask is between 0.05 and 0.95 (transition zone)
        transition = (mid_row > 0.05) & (mid_row < 0.95)
        col_indices = np.where(transition)[0]
        if len(col_indices) == 0:
            return []

        # Split into separate regions at gaps
        diffs = np.diff(col_indices)
        gaps = np.where(diffs > 10)[0]

        regions = []
        prev = 0
        for g in gaps:
            regions.append(col_indices[prev:g + 1])
            prev = g + 1
        regions.append(col_indices[prev:])

        # Add margin around each transition strip
        margin = 30
        result = []
        for r in regions:
            if len(r) >= 3:
                x_start = max(0, r[0] - margin)
                x_end = min(self.blend_mask.shape[1], r[-1] + 1 + margin)
                result.append((x_start, x_end))

        return result

    def _flow_align_overlap(self, eq_back, eq_front):
        """Warp eq_front to align with eq_back using attenuated optical flow.

        Computes flow on the full image (for context), then attenuates it
        using the blend mask so it's only applied in the overlap zone.
        weight = 4 * mask * (1 - mask): peaks at 1.0 at the seam (mask=0.5),
        zero at single-lens regions (mask=0 or 1).
        """
        gray_back = cv2.cvtColor(eq_back, cv2.COLOR_BGR2GRAY)
        gray_front = cv2.cvtColor(eq_front, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray_back, gray_front, None,
            pyr_scale=0.5, levels=5, winsize=13,
            iterations=10, poly_n=7, poly_sigma=1.5, flags=0,
        )

        # Clamp extreme flow values
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        clamp = np.minimum(30.0 / (mag + 1e-6), 1.0)
        flow[..., 0] *= clamp
        flow[..., 1] *= clamp

        # Attenuate: apply flow only in the overlap zone
        weight = 4.0 * self.blend_mask * (1.0 - self.blend_mask)
        flow[..., 0] *= weight
        flow[..., 1] *= weight

        # Warp front to align with back
        h, w = eq_front.shape[:2]
        map_x = np.arange(w, dtype=np.float32)[None, :] + flow[..., 0]
        map_y = np.arange(h, dtype=np.float32)[:, None] + flow[..., 1]

        return cv2.remap(eq_front, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def _find_seam_mask(self, eq_back, eq_front_warped):
        """Find optimal seam through overlap using dynamic programming.

        Returns a mask: 0=back, 1=front, with narrow feather at the seam.
        """
        h, w = eq_back.shape[:2]
        mask = self.blend_mask.copy()

        for x_start, x_end in self._overlap_regions:
            strip_back = eq_back[:, x_start:x_end].astype(np.float32)
            strip_front = eq_front_warped[:, x_start:x_end].astype(np.float32)
            sw = strip_back.shape[1]

            # Cost = color difference + edge penalty
            # Color diff: penalize crossing areas where the two images disagree
            color_diff = np.sum(np.abs(strip_back - strip_front), axis=2)

            # Edge penalty: penalize crossing strong edges in either image
            # (wires, fence posts, any high-contrast feature)
            gray_b = cv2.cvtColor(strip_back.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            gray_f = cv2.cvtColor(strip_front.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            edges_b = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3)**2 + \
                      cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3)**2
            edges_f = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)**2 + \
                      cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)**2
            edge_cost = np.sqrt(edges_b + edges_f)

            # Normalize edge cost to same scale as color diff
            if edge_cost.max() > 0:
                edge_cost = edge_cost / edge_cost.max() * color_diff.max()

            cost = color_diff + edge_cost

            # DP: find minimum-cost vertical path from top to bottom
            # Vectorized: for each row, take min of (left, center, right) from previous row
            dp = np.zeros_like(cost)
            dp[0] = cost[0]
            path = np.zeros((h, sw), dtype=np.int32)

            for y in range(1, h):
                # Shift left, center, right
                center = dp[y - 1]
                left = np.full(sw, np.inf)
                left[1:] = dp[y - 1, :-1]
                right = np.full(sw, np.inf)
                right[:-1] = dp[y - 1, 1:]

                choices = np.stack([left, center, right], axis=0)
                best_idx = np.argmin(choices, axis=0)  # 0=left, 1=center, 2=right
                dp[y] = cost[y] + np.min(choices, axis=0)
                path[y] = np.arange(sw) + (best_idx - 1)  # offset: -1, 0, +1
                path[y] = np.clip(path[y], 0, sw - 1)

            # Backtrack to find the seam path
            seam = np.zeros(h, dtype=np.int32)
            seam[h - 1] = np.argmin(dp[h - 1])
            for y in range(h - 2, -1, -1):
                seam[y] = path[y + 1, seam[y + 1]]

            # Determine direction: check blend_mask at left and right edges of strip
            left_val = self.blend_mask[h // 2, x_start]
            right_val = self.blend_mask[h // 2, x_end - 1]

            # Build binary mask from seam path
            strip_mask = np.zeros((h, sw), dtype=np.float32)
            if left_val < right_val:
                # Left=back(0), right=front(1): right side of seam = front
                for y in range(h):
                    strip_mask[y, seam[y]:] = 1.0
            else:
                # Left=front(1), right=back(0): left side of seam = front
                for y in range(h):
                    strip_mask[y, :seam[y] + 1] = 1.0

            # Narrow feather: blur the binary mask
            strip_mask = cv2.GaussianBlur(strip_mask, (7, 7), 1.5)

            # Write into the full mask
            mask[:, x_start:x_end] = strip_mask

        return mask

    def process(self, frame_back: np.ndarray, frame_front: np.ndarray,
                debug_dir: str = None) -> np.ndarray:
        """Process one frame pair into equirectangular output."""
        t_start = time.time()
        timings = {}

        # 1. Remap fisheye → equirectangular
        t = time.time()
        eq_back = cv2.remap(frame_back,
                            self.remapper.map_back_x, self.remapper.map_back_y,
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
        eq_front = cv2.remap(frame_front,
                             self.remapper.map_front_x, self.remapper.map_front_y,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
        timings["remap"] = time.time() - t

        if debug_dir:
            cv2.imwrite(f"{debug_dir}/01_fisheye_back.jpg", frame_back)
            cv2.imwrite(f"{debug_dir}/02_fisheye_front.jpg", frame_front)
            cv2.imwrite(f"{debug_dir}/03_equirect_back.jpg", eq_back)
            cv2.imwrite(f"{debug_dir}/04_equirect_front.jpg", eq_front)

        # 2. Optical flow alignment in overlap zones
        if self.use_optical_flow:
            t = time.time()
            eq_front_warped = self._flow_align_overlap(eq_back, eq_front)
            timings["flow"] = time.time() - t

            if debug_dir:
                # Visualize flow effect: difference before vs after
                diff_before = cv2.absdiff(eq_back, eq_front)
                diff_after = cv2.absdiff(eq_back, eq_front_warped)
                cv2.imwrite(f"{debug_dir}/05_diff_before_flow.jpg", diff_before)
                cv2.imwrite(f"{debug_dir}/06_diff_after_flow.jpg", diff_after)

            # 3. Find optimal seam through overlap
            t = time.time()
            seam_mask = self._find_seam_mask(eq_back, eq_front_warped)
            timings["seam"] = time.time() - t

            if debug_dir:
                cv2.imwrite(f"{debug_dir}/07_seam_mask.png",
                            (seam_mask * 255).astype(np.uint8))

            # 4. Composite using seam mask
            t = time.time()
            m3 = seam_mask[..., np.newaxis]
            result = (eq_back.astype(np.float32) * (1 - m3) +
                      eq_front_warped.astype(np.float32) * m3)
            result = np.clip(result, 0, 255).astype(np.uint8)
            timings["composite"] = time.time() - t
        else:
            # Fallback: distance-transform blend (no flow)
            t = time.time()
            m3 = self.blend_mask[..., np.newaxis]
            result = (eq_back.astype(np.float32) * (1 - m3) +
                      eq_front.astype(np.float32) * m3)
            result = np.clip(result, 0, 255).astype(np.uint8)
            timings["blend"] = time.time() - t

        timings["total"] = time.time() - t_start
        self.frame_count += 1

        if debug_dir:
            cv2.imwrite(f"{debug_dir}/08_result.jpg", result,
                        [cv2.IMWRITE_JPEG_QUALITY, 98])

        if self.verbose:
            parts = " ".join(f"{k}={v*1000:.0f}ms" for k, v in timings.items())
            print(f"  Frame {self.frame_count}: {parts}")

        return result


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def get_video_info(path: str):
    """Get video stream info using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps_parts = stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1])
    nb_frames = int(stream.get("nb_frames", 0)) or None
    return width, height, fps, nb_frames


def create_decoder(input_path: str, stream_index: int, width: int, height: int):
    """Create ffmpeg decoder subprocess for one video stream."""
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", input_path,
        "-map", f"0:v:{stream_index}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def create_encoder(output_path: str, width: int, height: int, fps: float,
                   audio_source: str = None, cq: int = 18):
    """Create ffmpeg encoder subprocess."""
    cmd = [
        "ffmpeg", "-y", "-v", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
    ]

    if audio_source:
        cmd.extend(["-i", audio_source, "-map", "0:v", "-map", "1:a", "-c:a", "copy"])

    # Try NVENC, fall back to libx265
    cmd.extend([
        "-c:v", "hevc_nvenc", "-preset", "p4",
        "-cq", str(cq), "-b:v", "0",
        "-movflags", "+faststart",
        output_path,
    ])

    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def read_frame(pipe, width: int, height: int):
    """Read one BGR24 frame from ffmpeg pipe."""
    nbytes = width * height * 3
    raw = pipe.stdout.read(nbytes)
    if len(raw) < nbytes:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()


# ---------------------------------------------------------------------------
# Processing modes
# ---------------------------------------------------------------------------

def find_pb_file(input_path: str) -> str:
    """Auto-detect the .pb calibration file for an input file."""
    input_name = Path(input_path).name
    # .pb files are in MISC/Camera01/ with the same name + .pb
    base_dir = Path(input_path).parent.parent.parent  # up from DCIM/Camera01/
    pb_dir = base_dir / "MISC" / "Camera01"
    pb_path = pb_dir / f"{input_name}.pb"
    if pb_path.exists():
        return str(pb_path)

    # Try finding any .pb file
    if pb_dir.exists():
        pb_files = list(pb_dir.glob("*.pb"))
        if pb_files:
            print(f"Warning: Exact .pb not found, using {pb_files[0].name}")
            return str(pb_files[0])

    raise FileNotFoundError(f"No .pb calibration file found for {input_path}")


def process_single_frame(args):
    """Process a single frame and save debug images."""
    input_path = args.input
    frame_num = args.single_frame

    # Find calibration
    pb_path = args.pb or find_pb_file(input_path)
    print(f"Calibration: {pb_path}")
    calib = CalibrationData.from_pb(pb_path)
    calib.print_summary()

    # Determine input type
    ext = Path(input_path).suffix.lower()
    is_photo = ext == ".insp"

    # Output dimensions
    scale = args.scale or 1.0
    out_w = int(args.width * scale)
    out_h = out_w // 2

    # Debug output directory
    debug_dir = args.output or str(Path(input_path).parent.parent.parent / "debug")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Debug output: {debug_dir}")

    if is_photo:
        # Read .insp as JPEG (dual fisheye side-by-side)
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read {input_path}")
        h, w = img.shape[:2]
        half_w = w // 2
        frame_back = img[:, :half_w]
        frame_front = img[:, half_w:]
        input_size = half_w
    else:
        # Extract single frames from .insv
        width, height, fps, _ = get_video_info(input_path)
        input_size = width
        print(f"Video: {width}x{height} @ {fps:.2f} fps")

        for stream_idx, name in [(0, "back"), (1, "front")]:
            cmd = [
                "ffmpeg", "-v", "quiet",
                "-i", input_path,
                "-map", f"0:v:{stream_idx}",
                "-vf", f"select=eq(n\\,{frame_num})",
                "-frames:v", "1",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-",
            ]
            proc = subprocess.run(cmd, capture_output=True)
            raw = proc.stdout
            if len(raw) != height * width * 3:
                print(f"  WARNING: stream {stream_idx} ({name}) got {len(raw)} bytes, "
                      f"expected {height * width * 3}")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            print(f"  Stream {stream_idx} ({name}): mean={frame.mean():.1f}, "
                  f"shape={frame.shape}")
            if name == "back":
                frame_back = frame.copy()
            else:
                frame_front = frame.copy()

    # Scale calibration and build pipeline
    pipeline = FramePipeline(
        calib, input_size, out_w, out_h,
        effective_fov=args.fov,
        use_optical_flow=not args.no_optical_flow,
        verbose=True,
    )

    # Process
    print("Processing frame...")
    result = pipeline.process(frame_back, frame_front, debug_dir=debug_dir)

    print(f"\nDone! Debug images saved to: {debug_dir}/")
    print("  01-02: Raw fisheye input (back, front)")
    print("  03-04: Equirectangular projection (back, front)")
    print("  05-07: Validity masks (back, front, overlap)")
    print("  08:    Blend mask")
    print("  09-10: Exposure-compensated (if enabled)")
    print("  11:    Final blended result")
    print("  12:    Hard-cut result (for comparison)")


def process_video(args):
    """Process a full video file."""
    input_path = args.input

    # Find calibration
    pb_path = args.pb or find_pb_file(input_path)
    print(f"Calibration: {pb_path}")
    calib = CalibrationData.from_pb(pb_path)
    calib.print_summary()

    # Video info
    width, height, fps, nb_frames = get_video_info(input_path)
    print(f"Video: {width}x{height} @ {fps:.2f} fps, {nb_frames or '?'} frames")

    # Output dimensions
    scale = args.scale or 1.0
    out_w = int(args.width * scale)
    out_h = out_w // 2

    # Output path
    if args.output:
        output_path = args.output
    else:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent.parent.parent /
                          "converted" / "equirectangular" / f"{stem}_stitched.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build pipeline
    pipeline = FramePipeline(
        calib, width, out_w, out_h,
        effective_fov=args.fov,
        use_optical_flow=not args.no_optical_flow,
        verbose=args.verbose,
    )

    # Start decoders
    print("Starting decoders...")
    dec_back = create_decoder(input_path, 0, width, height)
    dec_front = create_decoder(input_path, 1, width, height)

    # Start encoder
    print(f"Output: {output_path} ({out_w}x{out_h})")
    enc = create_encoder(output_path, out_w, out_h, fps,
                         audio_source=input_path, cq=args.cq)

    # Process frames
    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            frame_back = read_frame(dec_back, width, height)
            frame_front = read_frame(dec_front, width, height)

            if frame_back is None or frame_front is None:
                break

            result = pipeline.process(frame_back, frame_front)

            enc.stdin.write(result.tobytes())
            frame_idx += 1

            if frame_idx % 30 == 0:
                elapsed = time.time() - t_start
                fps_actual = frame_idx / elapsed
                if nb_frames:
                    pct = frame_idx / nb_frames * 100
                    print(f"  Frame {frame_idx}/{nb_frames} ({pct:.0f}%) - {fps_actual:.1f} fps")
                else:
                    print(f"  Frame {frame_idx} - {fps_actual:.1f} fps")

    finally:
        dec_back.terminate()
        dec_front.terminate()
        enc.stdin.close()
        enc.wait()

    elapsed = time.time() - t_start
    print(f"\nDone! {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dual fisheye to equirectangular stitcher for 360 cameras",
    )
    parser.add_argument("--input", "-i", required=True, help="Input .insv or .insp file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--pb", help="Calibration .pb file (auto-detected if omitted)")
    parser.add_argument("--fov", type=float, default=200.0,
                        help="Effective FOV per lens in degrees (default: 200)")
    parser.add_argument("--width", type=int, default=7680,
                        help="Output width (default: 7680, height=width/2)")
    parser.add_argument("--scale", type=float,
                        help="Resolution scale factor (e.g. 0.5 for half-res)")
    parser.add_argument("--single-frame", type=int, metavar="N",
                        help="Process only frame N, save debug images")
    parser.add_argument("--no-optical-flow", action="store_true",
                        help="Disable optical flow alignment (use distance-transform blend)")
    parser.add_argument("--cq", type=int, default=18,
                        help="NVENC constant quality for video (default: 18)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-frame timing")

    args = parser.parse_args()

    if args.single_frame is not None:
        process_single_frame(args)
    elif Path(args.input).suffix.lower() == ".insp":
        # Photo mode - single frame
        args.single_frame = 0
        process_single_frame(args)
    else:
        process_video(args)


if __name__ == "__main__":
    main()
