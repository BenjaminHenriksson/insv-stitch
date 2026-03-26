"""
x5_pipeline.py: Insta360 X5 stitching pipeline for Linux.

MEI (Unified Omnidirectional) model with extended distortion:
  4 radial (k1-k4) + 2 tangential (p1-p2) + 4 thin prism (s1-s4) + 2 extra
Dual fisheye → equirectangular with IMU-based stabilization.
"""

import numpy as np
import cv2
import subprocess
import time
import re
import base64
from dataclasses import dataclass, field
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import uniform_filter1d

import telemetry_parser


# ============================================================
# 1. Data Structures
# ============================================================

@dataclass
class MEILensParams:
    """MEI unified spherical camera model with extended distortion."""
    xi: float               # Mirror parameter (2.0 for X5)
    fx: float               # Focal length x (video resolution)
    fy: float               # Focal length y (video resolution)
    cx: float               # Principal point x (video resolution, cx_fix applied)
    cy: float               # Principal point y (video resolution)
    # Radial distortion (up to r^8)
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0         # Extended: r^8 term
    # Tangential distortion
    p1: float = 0.0
    p2: float = 0.0
    # Thin prism distortion
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    # Extrinsics
    R_extrinsic: np.ndarray = field(default_factory=lambda: np.eye(3))
    t_extrinsic: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Sensor
    width: int = 3840
    height: int = 3840

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

    @property
    def D(self):
        """Legacy 5-element distortion for backward compat."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])


@dataclass
class PipelineMetadata:
    lens_params: list         # [MEILensParams front, MEILensParams back]
    imu_samples: list         # normalized_imu output
    fps: float
    frame_count: int
    frame_readout_time: float # ms
    width: int
    height: int
    offset: list


# ============================================================
# 2. Metadata Extraction
# ============================================================

def _find_pb_path(insv_path: str) -> str | None:
    """Find the .pb protobuf sidecar file for an .insv file."""
    insv = Path(insv_path)
    # Look in MISC/Camera01/ relative to the DCIM/Camera01/ path
    misc_dir = insv.parent.parent.parent / 'MISC' / 'Camera01'
    pb_name = insv.name + '.pb'
    pb_path = misc_dir / pb_name
    if pb_path.exists():
        return str(pb_path)
    return None


def _parse_extended_calibration(pb_path: str) -> list[float] | None:
    """
    Extract the extended 56-element calibration string from a .pb file.
    Returns the parsed float values, or None if not found.
    """
    with open(pb_path, 'rb') as f:
        data = f.read()

    text = data.decode('latin-1')

    # Find base64-encoded content
    b64_match = re.search(r'[A-Za-z0-9+/=]{100,}', text)
    if not b64_match:
        return None

    decoded = base64.b64decode(b64_match.group()).decode('latin-1')

    # Find calibration strings: "2_<numbers...>" with >40 elements
    for match in re.finditer(r'2_[\d]+\.[\d]+_', decoded):
        start = match.start()
        # Extract the full numeric string
        chunk = decoded[start:]
        parts = []
        for token in chunk.split('_'):
            try:
                parts.append(float(token))
            except ValueError:
                break
        if len(parts) >= 55:  # Extended format has 56 elements
            return parts

    return None


def _sensor_to_video(fx, fy, cx, cy, sensor_w=5376, video_w=3840, cx_fix=2.0):
    """Convert calibration values from sensor resolution to video resolution."""
    crop_info_dst = 5312  # window_crop_info dst_width
    scale = video_w / crop_info_dst
    # cx_fix=2 for X5: cx is halved in Gyroflow convention, we double it
    cx_v = cx * video_w / (sensor_w * cx_fix) * cx_fix  # = cx * video_w / sensor_w
    cy_v = cy * video_w / sensor_w
    fx_v = fx * scale
    fy_v = fy * scale
    return fx_v, fy_v, cx_v, cy_v


def extract_metadata(insv_path: str) -> PipelineMetadata:
    """
    Extract all metadata from .insv file.
    Prefers extended protobuf calibration (13-coeff model) if available,
    falls back to offset_v3 / Gyroflow profile.
    """
    tp = telemetry_parser.Parser(insv_path)
    telem = tp.telemetry()
    item = telem[0]

    meta = item['Default']['Metadata']
    lens_data = item['Lens']['Data']
    dim = meta['dimension']
    width, height = dim['x'], dim['y']
    offset = meta['offset']

    # Try to load extended calibration from protobuf sidecar
    pb_path = _find_pb_path(insv_path)
    ext = _parse_extended_calibration(pb_path) if pb_path else None

    if ext and len(ext) >= 55:
        # ---- Extended 56-element calibration (best available) ----
        # Format: [0]=num, per lens (27 elements):
        #   [+0]=xi, [+1]=fx, [+2]=fy, [+3]=cx, [+4]=cy,
        #   [+5]=yaw, [+6]=pitch, [+7]=roll,
        #   [+8]=tx, [+9]=ty, [+10]=tz,
        #   [+11..+23]=13 distortion coefficients,
        #   [+24]=w, [+25]=h, [+26]=type
        lenses = []
        for lens_idx in range(2):
            s = 1 + lens_idx * 27
            xi = ext[s]
            fx_s, fy_s = ext[s+1], ext[s+2]
            cx_s, cy_s = ext[s+3], ext[s+4]
            yaw, pitch, roll = ext[s+5], ext[s+6], ext[s+7]
            tx, ty, tz = ext[s+8], ext[s+9], ext[s+10]
            coeffs = ext[s+11:s+24]

            # Convert to video resolution
            if lens_idx == 1:
                cx_s -= 5376  # Remove concatenated offset
            fx_v, fy_v, cx_v, cy_v = _sensor_to_video(fx_s, fy_s, cx_s, cy_s)

            # Extrinsic rotation (small yaw/pitch corrections only)
            if lens_idx == 0:
                R = Rotation.from_euler('YX', [yaw, pitch], degrees=True).as_matrix()
            else:
                R_back = Rotation.from_euler('Y', 180, degrees=True)
                R_corr = Rotation.from_euler('YX', [yaw, pitch], degrees=True)
                R = (R_back * R_corr).as_matrix()

            lens = MEILensParams(
                xi=xi, fx=fx_v, fy=fy_v, cx=cx_v, cy=cy_v,
                k1=coeffs[0], k2=coeffs[1], k3=coeffs[2], k4=coeffs[3],
                p1=coeffs[5], p2=coeffs[6],
                s1=coeffs[7], s2=coeffs[8], s3=coeffs[9], s4=coeffs[10],
                R_extrinsic=R, t_extrinsic=np.array([tx, ty, tz]),
                width=width, height=height,
            )
            lenses.append(lens)

        print(f"  Using extended protobuf calibration (13 coefficients per lens)")
    else:
        # ---- Fallback: Gyroflow / offset_v3 (5-coeff model) ----
        fp = lens_data['fisheye_params']
        dc = fp['distortion_coeffs']
        cm = fp['camera_matrix']

        xi = dc[5]
        k1, k2, k3 = dc[0], dc[1], dc[2]
        p1, p2 = dc[3], dc[4]
        fx_0 = cm[0][0]
        fy_0 = cm[1][1]
        cx_0 = cm[0][2] * 2.0  # cx_fix=2 for X5
        cy_0 = cm[1][2]

        yaw_0, pitch_0 = offset[4], offset[5]
        R0 = Rotation.from_euler('YX', [yaw_0, pitch_0], degrees=True).as_matrix()
        lens_front = MEILensParams(
            xi=xi, fx=fx_0, fy=fy_0, cx=cx_0, cy=cy_0,
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            R_extrinsic=R0, width=width, height=height,
        )

        fx_ratio = offset[7] / offset[1]
        cx_1_offset = offset[8] - offset[14]
        yaw_1, pitch_1 = offset[10], offset[11]
        R_back = Rotation.from_euler('Y', 180, degrees=True)
        R1 = (R_back * Rotation.from_euler('YX', [yaw_1, pitch_1], degrees=True)).as_matrix()
        lens_back = MEILensParams(
            xi=xi, fx=fx_0 * fx_ratio, fy=fy_0 * fx_ratio,
            cx=cx_0 * (cx_1_offset / offset[2]),
            cy=cy_0 * (offset[9] / offset[3]),
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            R_extrinsic=R1, width=width, height=height,
        )
        lenses = [lens_front, lens_back]
        print(f"  Using Gyroflow/offset_v3 calibration (5 coefficients)")

    imu_samples = tp.normalized_imu()
    fps_val = float(meta.get('frame_rate', 30))
    frame_readout_time = lens_data.get('frame_readout_time', 21.24)

    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', insv_path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        frame_count = int(r.stdout.strip())
    except Exception:
        frame_count = 0

    return PipelineMetadata(
        lens_params=lenses,
        imu_samples=imu_samples, fps=fps_val,
        frame_count=frame_count,
        frame_readout_time=frame_readout_time,
        width=width, height=height, offset=offset,
    )


# ============================================================
# 3. IMU Stabilization
# ============================================================

# IMU-to-camera rotation matrix for X5.
# Optimized across multiple videos (003, 002) to avoid single-video overfitting.
# The IMU sensor is mounted at a non-trivial angle (~120° from identity) on the
# X5 PCB; this rotation maps normalized_imu() output to the stitching camera
# frame (X=right, Y=down, Z=forward).
IMU_TO_CAM = np.array([
    [-0.5678,  0.4608, -0.6821],
    [ 0.7642,  0.6031, -0.2287],
    [ 0.3060, -0.6511, -0.6946],
], dtype=np.float64)


def imu_to_camera(vec):
    """Transform a vector from IMU frame to camera image frame."""
    return IMU_TO_CAM @ np.asarray(vec)


def compute_gravity_orientation(accel_cam):
    """
    Compute orientation from gravity vector in camera Y-down frame.
    Returns rotation that levels the horizon.
    """
    g = accel_cam / np.linalg.norm(accel_cam)
    target = np.array([0.0, 1.0, 0.0])  # gravity = +Y when level

    axis = np.cross(g, target)
    s = np.linalg.norm(axis)
    c = np.dot(g, target)

    if s < 1e-6:
        return Rotation.identity()

    axis /= s
    angle = np.arccos(np.clip(c, -1, 1))
    return Rotation.from_rotvec(axis * angle)


def compute_stabilization_from_imu(
    imu_samples, fps, tau=1.5, smoothing_sec=0.3
):
    """
    Compute per-frame horizon-lock stabilization using complementary filter.

    Fuses gyroscope (fast/drifty) with accelerometer (slow/noisy but drift-free).
    Only applies accel correction when |accel| ≈ g (low dynamic acceleration).

    Returns: dict mapping frame_index → Rotation (correction to apply)
    """
    if not imu_samples:
        return {}

    G = 9.81
    ACCEL_THRESHOLD = 0.15  # Relative tolerance for |accel| ≈ g

    n = len(imu_samples)
    timestamps = np.array([s['timestamp_ms'] / 1000.0 for s in imu_samples])
    accels_imu = np.array([[s['accl'][0], s['accl'][1], s['accl'][2]]
                           for s in imu_samples])
    gyros_imu = np.array([[s['gyro'][0], s['gyro'][1], s['gyro'][2]]
                          for s in imu_samples])

    # Convert to camera frame
    accels_cam = (IMU_TO_CAM @ accels_imu.T).T
    gyros_cam = (IMU_TO_CAM @ gyros_imu.T).T

    # Smooth accelerometer to filter out dynamic forces.
    # A window of ~50 samples (~250ms at 200Hz) balances noise reduction
    # vs responsiveness.
    accel_mags = np.linalg.norm(accels_cam, axis=1)
    smooth_window = min(50, n)
    accels_smooth = np.zeros_like(accels_cam)
    for j in range(3):
        accels_smooth[:, j] = uniform_filter1d(accels_cam[:, j], smooth_window)

    # Compute leveling rotation at each IMU sample
    orientations = []
    for i in range(n):
        R_level = compute_gravity_orientation(accels_smooth[i])
        orientations.append((timestamps[i], R_level))

    # Map to per-frame corrections + per-scanline RS data
    frame_corrections = {}
    for frame_idx in range(int(timestamps[-1] * fps) + 2):
        t_frame = frame_idx / fps
        idx = np.searchsorted(timestamps, t_frame)
        idx = np.clip(idx, 0, len(timestamps) - 1)
        frame_corrections[frame_idx] = orientations[idx][1]

    return frame_corrections


def compute_rs_rotations(imu_samples, frame_num, fps, readout_time_ms,
                         n_scanlines=32):
    """
    Compute per-scanline orientation for rolling shutter correction.

    During the readout, each row was captured at a different time.
    The gyro angular velocity is integrated to compute the relative
    orientation at each scanline position.

    Returns: list of (scanline_frac, Rotation) pairs, where the Rotation
        is the TOTAL stabilization rotation at that scanline's capture time.
    """
    if not imu_samples:
        return None

    timestamps = np.array([s['timestamp_ms'] / 1000.0 for s in imu_samples])
    gyros_imu = np.array([[s['gyro'][0], s['gyro'][1], s['gyro'][2]]
                          for s in imu_samples])
    accels_imu = np.array([[s['accl'][0], s['accl'][1], s['accl'][2]]
                           for s in imu_samples])

    gyros_cam = (IMU_TO_CAM @ gyros_imu.T).T
    accels_cam = (IMU_TO_CAM @ accels_imu.T).T

    # Smoothed gravity for base orientation
    smooth_window = min(50, len(imu_samples))
    accels_smooth = np.zeros_like(accels_cam)
    for j in range(3):
        accels_smooth[:, j] = uniform_filter1d(accels_cam[:, j], smooth_window)

    readout_sec = readout_time_ms / 1000.0
    t_frame = frame_num / fps

    # Base orientation at frame center
    idx_center = np.searchsorted(timestamps, t_frame)
    idx_center = np.clip(idx_center, 0, len(timestamps) - 1)
    R_base = compute_gravity_orientation(accels_smooth[idx_center])

    # For each scanline, compute the gyro-integrated orientation delta
    # relative to the frame center
    rs_rots = []
    for i in range(n_scanlines):
        scanline_frac = i / (n_scanlines - 1)
        t_scanline = t_frame + (scanline_frac - 0.5) * readout_sec

        # Integrate gyro from t_frame to t_scanline
        idx_start = np.searchsorted(timestamps, min(t_frame, t_scanline))
        idx_end = np.searchsorted(timestamps, max(t_frame, t_scanline))
        idx_start = np.clip(idx_start, 0, len(timestamps) - 1)
        idx_end = np.clip(idx_end, 0, len(timestamps) - 1)

        if idx_start == idx_end:
            R_delta = Rotation.identity()
        else:
            # Integrate gyro angular velocity
            total_rotvec = np.zeros(3)
            for k in range(min(idx_start, idx_end), max(idx_start, idx_end)):
                dt = timestamps[k+1] - timestamps[k]
                gyro_rad = np.deg2rad(gyros_cam[k])
                total_rotvec += gyro_rad * dt

            if t_scanline < t_frame:
                total_rotvec = -total_rotvec

            R_delta = Rotation.from_rotvec(total_rotvec)

        # Total orientation at this scanline = base + gyro delta
        R_scanline = R_delta * R_base
        rs_rots.append((scanline_frac, R_scanline))

    return rs_rots


# ============================================================
# 4. Frame Decoding
# ============================================================

def decode_frame(insv_path, frame_num, track, width, height):
    """Decode a single frame from a specific video track."""
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', insv_path, '-map', f'0:{track}',
        '-vf', f'select=eq(n\\,{frame_num})',
        '-frames:v', '1',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    expected = width * height * 3
    if len(r.stdout) != expected:
        raise RuntimeError(
            f"Decode failed: got {len(r.stdout)} bytes, expected {expected}")
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


# ============================================================
# 5. MEI Forward Projection
# ============================================================

def mei_forward(X, Y, Z, xi, K, D_or_lens=None, lens=None):
    """
    Project 3D rays → pixel coordinates through MEI model.

    Supports both legacy 5-element D array and full MEILensParams with
    extended distortion (k1-k4, p1-p2, s1-s4).

    Returns: u, v (pixel coords), valid (bool mask)
    """
    norm = np.sqrt(X*X + Y*Y + Z*Z)
    norm = np.maximum(norm, 1e-10)
    Xs, Ys, Zs = X/norm, Y/norm, Z/norm

    denom = Zs + xi
    valid = denom > 1e-6

    x = np.where(valid, Xs / denom, 0.0)
    y = np.where(valid, Ys / denom, 0.0)

    r2 = x*x + y*y
    r4 = r2 * r2

    # Use MEILensParams if provided, else legacy D array
    if lens is not None or (isinstance(D_or_lens, MEILensParams)):
        L = lens if lens is not None else D_or_lens
        radial = 1.0 + L.k1*r2 + L.k2*r4 + L.k3*r2*r4 + L.k4*r4*r4
        xd = (x*radial + 2*L.p1*x*y + L.p2*(r2 + 2*x*x)
              + L.s1*r2 + L.s2*r4)
        yd = (y*radial + L.p1*(r2 + 2*y*y) + 2*L.p2*x*y
              + L.s3*r2 + L.s4*r4)
    else:
        D = D_or_lens
        k1, k2, p1, p2 = D[0], D[1], D[2], D[3]
        k3 = D[4] if len(D) > 4 else 0.0
        radial = 1.0 + k1*r2 + k2*r4 + k3*r2*r4
        xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y

    u = K[0, 0] * xd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]

    return u, v, valid


# ============================================================
# 6. Equirectangular Remap Builder
# ============================================================

def build_equirect_remap(lens, eq_w, eq_h, R_stabilization=None,
                         rs_rotations=None, depth_map=None):
    """
    Build remap tables: equirectangular pixels → fisheye source pixels.
    Camera convention: X=right, Y=down, Z=forward.

    Args:
        rs_rotations: Optional list of (scanline_frac, Rotation) pairs for
            rolling shutter correction.
        depth_map: Optional H×W float array of scene depth (meters). When
            provided, the lens translation (t_extrinsic) is used to compute
            parallax-corrected projections. At d=∞ the result is identical
            to the standard projection; at finite d, close objects shift
            to their geometrically correct position for this specific lens.
    """
    uu, vv = np.meshgrid(
        np.arange(eq_w, dtype=np.float64),
        np.arange(eq_h, dtype=np.float64))

    lon = (uu / eq_w) * 2 * np.pi - np.pi
    lat = np.pi / 2 - (vv / eq_h) * np.pi

    ray_x = np.cos(lat) * np.sin(lon)
    ray_y = -np.sin(lat)   # Y-down camera convention
    ray_z = np.cos(lat) * np.cos(lon)

    if rs_rotations is not None and len(rs_rotations) > 1:
        rays_stab = np.empty((eq_h, eq_w, 3), dtype=np.float64)
        fracs = np.array([f for f, _ in rs_rotations])
        rots = [R for _, R in rs_rotations]

        for row in range(eq_h):
            scanline_frac = row / eq_h
            idx = np.searchsorted(fracs, scanline_frac) - 1
            idx = np.clip(idx, 0, len(fracs) - 2)
            f0, f1 = fracs[idx], fracs[idx + 1]
            alpha = (scanline_frac - f0) / max(f1 - f0, 1e-9)
            alpha = np.clip(alpha, 0, 1)

            slerp = Slerp([0, 1], Rotation.concatenate([rots[idx], rots[idx + 1]]))
            R_row = slerp([alpha])[0]

            row_rays = np.stack([ray_x[row], ray_y[row], ray_z[row]], axis=-1)
            rays_stab[row] = (R_row.as_matrix() @ row_rays.T).T

        rays = rays_stab.reshape(-1, 3).T
    else:
        rays = np.stack([ray_x.ravel(), ray_y.ravel(), ray_z.ravel()], axis=0)
        if R_stabilization is not None:
            rays = R_stabilization.as_matrix() @ rays

    # Transform ray directions to lens-local frame
    rays_lens = lens.R_extrinsic.T @ rays

    Xl = rays_lens[0].reshape(eq_h, eq_w)
    Yl = rays_lens[1].reshape(eq_h, eq_w)
    Zl = rays_lens[2].reshape(eq_h, eq_w)

    # Translation-aware projection: account for lens offset from camera center.
    # For a 3D point at distance d along the ray:
    #   P_world = d * ray_dir
    #   P_lens = R.T @ (P_world - t_extrinsic)
    #          = d * ray_lens - t_local
    # MEI normalizes to unit sphere internally, so the ratio matters, not scale.
    # At d→∞ the t_local term vanishes → standard projection.
    if depth_map is not None and np.linalg.norm(lens.t_extrinsic) > 1e-6:
        # The protobuf t_extrinsic is the offset FROM this lens TO the
        # camera center (not the lens position). So the lens position
        # is at -t_extrinsic, and we ADD t_local to correct the ray.
        t_local = lens.R_extrinsic.T @ lens.t_extrinsic
        d = depth_map.astype(np.float64)
        Xl = d * Xl + t_local[0]
        Yl = d * Yl + t_local[1]
        Zl = d * Zl + t_local[2]

    u_src, v_src, valid = mei_forward(Xl, Yl, Zl, lens.xi, lens.K, lens=lens)

    # Bounds check against sensor
    valid = valid & (u_src >= 0) & (u_src < lens.width - 1)
    valid = valid & (v_src >= 0) & (v_src < lens.height - 1)

    # Circular fisheye mask
    margin = 0.05
    r_max = min(lens.width, lens.height) / 2.0 * (1.0 - margin)
    dist_from_center = np.sqrt(
        (u_src - lens.cx) ** 2 + (v_src - lens.cy) ** 2)
    valid = valid & (dist_from_center < r_max)

    map_x = np.where(valid, u_src, 0).astype(np.float32)
    map_y = np.where(valid, v_src, 0).astype(np.float32)

    return map_x, map_y, valid


def remap_fisheye(fisheye_bgr, map_x, map_y, valid):
    """Apply remap tables to a fisheye image."""
    result = cv2.remap(fisheye_bgr, map_x, map_y, cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result[~valid] = 0
    return result


# ============================================================
# 7. Blending
# ============================================================

def compute_blend_weights(valid_front, valid_back, eq_w, eq_h,
                          blend_width_deg=15.0):
    """
    Blend weights combining longitude preference with coverage depth.

    In the overlap region, each lens's weight is proportional to:
      longitude_preference × coverage_depth

    This naturally:
    - Assigns front/back hemisphere ownership via longitude
    - Favors the lens further from its fisheye edge (more central = less
      distortion, less parallax). No hardcoded feather distance needed
    - Smoothly transitions at coverage boundaries
    """
    # Longitude preference: front owns |lon| < 90°, back owns |lon| > 90°
    cols = np.arange(eq_w, dtype=np.float32)
    lon_deg = (cols / eq_w) * 360.0 - 180.0
    abs_lon = np.abs(lon_deg)

    lo = 90.0 - blend_width_deg
    hi = 90.0 + blend_width_deg

    front_pref = np.ones(eq_w, dtype=np.float32)
    ramp = (abs_lon >= lo) & (abs_lon <= hi)
    front_pref[ramp] = 1.0 - (abs_lon[ramp] - lo) / (hi - lo)
    front_pref[abs_lon > hi] = 0.0
    front_pref_2d = np.broadcast_to(front_pref[np.newaxis, :], (eq_h, eq_w))

    # Coverage depth: how far each pixel is from its lens's coverage edge
    front_dist = cv2.distanceTransform(
        valid_front.astype(np.uint8) * 255, cv2.DIST_L2, 5).astype(np.float32)
    back_dist = cv2.distanceTransform(
        valid_back.astype(np.uint8) * 255, cv2.DIST_L2, 5).astype(np.float32)

    # Combined weight: longitude preference × coverage depth
    # Coverage depth naturally fades to 0 at the edge. No hardcoded feather.
    w_front = front_pref_2d * front_dist
    w_back = (1.0 - front_pref_2d) * back_dist

    # Normalize
    total = w_front + w_back
    total = np.maximum(total, 1e-6)
    w_front = w_front / total
    w_back = w_back / total

    # Where only one lens is valid, use it exclusively
    only_front = valid_front & ~valid_back
    only_back = ~valid_front & valid_back
    w_front[only_front] = 1.0; w_back[only_front] = 0.0
    w_front[only_back] = 0.0; w_back[only_back] = 1.0

    # Zero where neither valid
    neither = ~valid_front & ~valid_back
    w_front[neither] = 0.0; w_back[neither] = 0.0

    return w_front, w_back


def compute_gain_compensation(front_overlap, back_overlap, overlap_mask):
    """Per-channel global gain to equalize exposure in overlap."""
    valid = overlap_mask
    gains = np.ones(3, dtype=np.float64)
    for c in range(3):
        mf = front_overlap[:, :, c][valid].mean()
        mb = back_overlap[:, :, c][valid].mean()
        if mb > 1:
            gains[c] = mf / mb
    return np.clip(gains, 0.5, 2.0)


def compute_spatial_gain(eq_front, eq_back, overlap_mask, blur_sigma=80):
    """
    Compute a spatially-varying per-channel gain field to equalize
    the back image to match the front image's appearance.

    Uses the overlap region to measure the per-pixel intensity ratio,
    then smooths it into a continuous gain field that extends beyond
    the overlap.
    """
    h, w = eq_front.shape[:2]
    gain_field = np.ones((h, w, 3), dtype=np.float32)

    for c in range(3):
        f = eq_front[:, :, c].astype(np.float32)
        b = eq_back[:, :, c].astype(np.float32)

        # Per-pixel ratio in overlap (where both have signal)
        valid = overlap_mask & (b > 10) & (f > 5)
        ratio = np.ones((h, w), dtype=np.float32)
        ratio[valid] = f[valid] / b[valid]

        # Clamp outliers
        ratio = np.clip(ratio, 0.3, 3.0)

        # Set non-overlap pixels to 1.0 before blurring
        ratio[~valid] = 0.0
        weight = valid.astype(np.float32)

        # Weighted Gaussian blur: blur(ratio * weight) / blur(weight)
        ratio_blurred = cv2.GaussianBlur(ratio * weight, (0, 0), blur_sigma)
        weight_blurred = cv2.GaussianBlur(weight, (0, 0), blur_sigma)
        weight_blurred = np.maximum(weight_blurred, 1e-6)

        gain_field[:, :, c] = ratio_blurred / weight_blurred

    # Clamp to reasonable range
    gain_field = np.clip(gain_field, 0.5, 2.0)
    return gain_field


def apply_gain(image, gains):
    """Apply per-channel gain (scalar or spatial field)."""
    result = image.astype(np.float32)
    if isinstance(gains, np.ndarray) and gains.ndim == 3:
        result *= gains
    else:
        for c in range(3):
            result[:, :, c] *= gains[c]
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# 8. Optical Flow (DIS)
# ============================================================

class FlowEngine:
    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.dis.setVariationalRefinementIterations(5)
        self.dis.setFinestScale(0)
        self.dis.setGradientDescentIterations(25)

    def compute(self, front_gray, back_gray, prev_flow=None):
        return self.dis.calc(front_gray, back_gray, prev_flow)


# ============================================================
# 9. Seam Finding (DP)
# ============================================================

def find_seam_dp(img_a, img_b):
    """Optimal vertical seam via dynamic programming."""
    h, w = img_a.shape[:2]
    diff = np.linalg.norm(
        img_a.astype(np.float32) - img_b.astype(np.float32), axis=2)
    ga = cv2.Sobel(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0)
    gb = cv2.Sobel(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0)
    cost = diff + 0.5 * np.abs(ga - gb)

    dp = cost.copy()
    for row in range(1, h):
        for col in range(w):
            lo, hi = max(0, col - 1), min(w, col + 2)
            dp[row, col] += dp[row - 1, lo:hi].min()

    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = dp[-1].argmin()
    for row in range(h - 2, -1, -1):
        c = seam[row + 1]
        lo, hi = max(0, c - 1), min(w, c + 2)
        seam[row] = lo + dp[row, lo:hi].argmin()

    mask = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        mask[row, :seam[row]] = 255
    return mask


# ============================================================
# 10. Multi-Band Blending
# ============================================================

def _gaussian_pyramid(img, levels):
    pyr = [img.astype(np.float64)]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _laplacian_pyramid(img, levels):
    gauss = _gaussian_pyramid(img, levels)
    lap = []
    for i in range(levels - 1):
        h, w = gauss[i].shape[:2]
        up = cv2.pyrUp(gauss[i + 1], dstsize=(w, h))
        lap.append(gauss[i] - up)
    lap.append(gauss[-1])
    return lap


def multiband_blend(img_a, img_b, seam_mask, levels=5, feather_px=10):
    """Laplacian pyramid multi-band blending."""
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    mask = seam_mask.astype(np.float64) / 255.0
    if feather_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather_px)
    if mask.ndim == 2:
        mask = np.stack([mask] * 3, axis=2)

    la = _laplacian_pyramid(a, levels)
    lb = _laplacian_pyramid(b, levels)
    gm = _gaussian_pyramid(mask, levels)

    blended = [l_a * g + l_b * (1.0 - g) for l_a, l_b, g in zip(la, lb, gm)]

    result = blended[-1]
    for i in range(len(blended) - 2, -1, -1):
        h, w = blended[i].shape[:2]
        result = cv2.pyrUp(result, dstsize=(w, h)) + blended[i]

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# 11. Pipeline Orchestrator
# ============================================================

def find_overlap_bands(valid_front, valid_back):
    """Find contiguous overlap column ranges."""
    overlap = valid_front & valid_back
    cols = np.any(overlap, axis=0)
    diff = np.diff(cols.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if cols[0]:
        starts = np.concatenate([[0], starts])
    if cols[-1]:
        ends = np.concatenate([ends, [len(cols)]])
    return overlap, list(zip(starts.tolist(), ends.tolist()))


class X5Pipeline:

    def __init__(self, insv_path, eq_width=3840,
                 enable_stabilization=True, enable_flow=True,
                 denoise=False):
        self.insv_path = insv_path
        self.eq_w = eq_width
        self.eq_h = eq_width // 2
        self.enable_flow = enable_flow
        self.denoise = denoise

        print("[1/4] Extracting metadata...")
        self.meta = extract_metadata(insv_path)
        print(f"  Video: {self.meta.width}x{self.meta.height} @ "
              f"{self.meta.fps}fps, {self.meta.frame_count} frames")

        print("[2/4] Computing IMU stabilization...")
        if enable_stabilization:
            self.stab_corrections = compute_stabilization_from_imu(
                self.meta.imu_samples, self.meta.fps)
            print(f"  {len(self.stab_corrections)} frame corrections computed")
        else:
            self.stab_corrections = {}

        print("[3/4] Building remap tables...")
        self._build_maps(frame_idx=0)

        if enable_flow:
            self.flow_engine = FlowEngine()

        print("[4/4] Ready.")
        print(f"  Output: {self.eq_w}x{self.eq_h}")

    def _build_maps(self, frame_idx=0, depth_map=None):
        """Build remap tables with stabilization + rolling shutter correction."""
        R_stab = self.stab_corrections.get(frame_idx, Rotation.identity())

        # Per-scanline RS correction from gyro data
        rs_rots = compute_rs_rotations(
            self.meta.imu_samples, frame_idx, self.meta.fps,
            self.meta.frame_readout_time, n_scanlines=32)

        self.maps = []
        for lens in self.meta.lens_params:
            mx, my, valid = build_equirect_remap(
                lens, self.eq_w, self.eq_h,
                R_stabilization=R_stab, rs_rotations=rs_rots,
                depth_map=depth_map)
            self.maps.append((mx, my, valid))

        self.w_front, self.w_back = compute_blend_weights(
            self.maps[0][2], self.maps[1][2], self.eq_w, self.eq_h)

        # Store stab/RS params for depth-corrected rebuild
        self._last_R_stab = R_stab
        self._last_rs_rots = rs_rots

    def _estimate_depth_and_rebuild(self, front_fish, back_fish):
        """
        Disparity-adaptive blending: sharpen blend weights near close objects.

        The standard longitude×depth blending produces ghosting when close
        objects straddle the stitch line (the two lenses see them from
        different positions). This method detects high-disparity regions
        via DIS flow in the stitch bands and sharpens the blend there,
        strongly favoring whichever lens has the pixel more centrally.

        The key insight: blending two misaligned views of a close object
        creates ghosting. Picking ONE view (the better one) eliminates
        the ghost at the cost of a sharper transition, which is less
        visible than the ghost.
        """
        eq_front = remap_fisheye(front_fish, *self.maps[0])
        eq_back = remap_fisheye(back_fish, *self.maps[1])
        valid_f, valid_b = self.maps[0][2], self.maps[1][2]
        overlap = valid_f & valid_b

        if not overlap.any():
            return

        # Compute flow in the stitch bands to detect parallax
        stitch_half_deg = 15.0
        parallax_map = np.zeros((self.eq_h, self.eq_w), dtype=np.float32)

        for stitch_lon in [90.0, 270.0]:
            col_center = int(stitch_lon / 360.0 * self.eq_w)
            hw = int(stitch_half_deg / 360.0 * self.eq_w)
            cs = max(0, col_center - hw)
            ce = min(self.eq_w, col_center + hw)

            band_ov = overlap[:, cs:ce]
            if not band_ov.any():
                continue

            fc = eq_front[:, cs:ce].copy()
            bc = eq_back[:, cs:ce].copy()
            fc[~valid_f[:, cs:ce]] = bc[~valid_f[:, cs:ce]]
            bc[~valid_b[:, cs:ce]] = fc[~valid_b[:, cs:ce]]

            fg = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
            bg = cv2.cvtColor(bc, cv2.COLOR_BGR2GRAY)

            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            dis.setFinestScale(0)
            flow = dis.calc(fg, bg, None)

            mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            mag[~band_ov] = 0
            parallax_map[:, cs:ce] = np.maximum(parallax_map[:, cs:ce], mag)

        # Smooth the parallax map
        parallax_map = cv2.GaussianBlur(parallax_map, (0, 0), 5)

        # Where parallax is high (close objects), sharpen the blend:
        # push weights toward 0 or 1 based on which lens dominates.
        # The threshold scales with resolution (more pixels = more parallax)
        threshold = self.eq_w / 960.0  # ~2px at 1920, ~8px at 7680
        sharpening = np.clip(parallax_map / max(threshold, 0.5), 0, 1)

        # Sharpen: w = w^(1+sharpening*3) then renormalize
        # This pushes weights toward 0/1 where parallax is high
        power = 1.0 + sharpening * 4.0
        w_f = self.w_front ** power
        w_b = self.w_back ** power
        total = np.maximum(w_f + w_b, 1e-6)
        self.w_front = w_f / total
        self.w_back = w_b / total

    def stitch_frame(self, frame_num=0):
        """Stitch a single frame with depth-aware translation correction."""
        t0 = time.time()

        if self.stab_corrections:
            self._build_maps(frame_num)

        front = decode_frame(self.insv_path, frame_num, 0,
                             self.meta.width, self.meta.height)
        back = decode_frame(self.insv_path, frame_num, 1,
                            self.meta.width, self.meta.height)

        eq_front = remap_fisheye(front, *self.maps[0])
        eq_back = remap_fisheye(back, *self.maps[1])

        valid_f, valid_b = self.maps[0][2], self.maps[1][2]

        # Color harmonization: SYMMETRIC correction at the seam.
        # Both lenses are adjusted toward their geometric mean in the overlap,
        # so neither side has an uncorrected color step at the transition.
        # The correction fades to zero away from the seam.
        overlap_mask = valid_f & valid_b
        eq_back_f = eq_back.astype(np.float32)
        eq_front_f = eq_front.astype(np.float32)
        if overlap_mask.any():
            gain_field = compute_spatial_gain(eq_front, eq_back, overlap_mask)
            # gain = front/back. To meet in the middle:
            #   front_adj = front / sqrt(gain) = front * sqrt(back/front)
            #   back_adj  = back  * sqrt(gain) = back  * sqrt(front/back)
            sqrt_gain = np.sqrt(np.maximum(gain_field, 0.01))
            inv_sqrt_gain = 1.0 / np.maximum(sqrt_gain, 0.01)

            # Correction weight: strongest at seam center, zero in primary regions
            correction_weight = 2.0 * np.minimum(self.w_front, self.w_back)
            correction_weight = correction_weight[:, :, np.newaxis]

            # Symmetrically correct both lenses, fading with distance from seam
            eq_front_f = eq_front_f * (1.0 - correction_weight) + \
                         eq_front_f * inv_sqrt_gain * correction_weight
            eq_back_f = eq_back_f * (1.0 - correction_weight) + \
                        eq_back_f * sqrt_gain * correction_weight

            eq_front = np.clip(eq_front_f, 0, 255).astype(np.uint8)
            eq_back = np.clip(eq_back_f, 0, 255).astype(np.uint8)

        # Base canvas: smooth weighted blend
        result = np.clip(
            eq_front.astype(np.float32) * self.w_front[:, :, np.newaxis] +
            eq_back.astype(np.float32) * self.w_back[:, :, np.newaxis],
            0, 255).astype(np.uint8)

        if self.enable_flow:
            # Apply flow + seam only in the stitch-line regions (~±90° lon).
            # Two stitch bands: one on the left side (~270° = col near 3/4),
            # one on the right side (~90° = col near 1/4).
            stitch_half_width_deg = 15.0
            stitch_bands = []
            for stitch_lon_deg in [90.0, 270.0]:
                col_center = int(stitch_lon_deg / 360.0 * self.eq_w)
                hw = int(stitch_half_width_deg / 360.0 * self.eq_w)
                col_s = max(0, col_center - hw)
                col_e = min(self.eq_w, col_center + hw)
                stitch_bands.append((col_s, col_e))

            for col_s, col_e in stitch_bands:
                band_w = col_e - col_s
                if band_w < 10:
                    continue

                # Check both lenses have coverage in this band
                band_valid_f = valid_f[:, col_s:col_e].any()
                band_valid_b = valid_b[:, col_s:col_e].any()
                if not (band_valid_f and band_valid_b):
                    continue

                s = col_s
                e = col_e

                band_overlap = valid_f[:, s:e] & valid_b[:, s:e]
                band_vf = valid_f[:, s:e]
                band_vb = valid_b[:, s:e]

                # For flow: use fully corrected versions of both lenses
                # (symmetric sqrt(gain) correction applied to both)
                if overlap_mask.any():
                    fc = np.clip(eq_front_f[:, s:e] * inv_sqrt_gain[:, s:e],
                                 0, 255).astype(np.uint8)
                    bc = np.clip(eq_back_f[:, s:e] * sqrt_gain[:, s:e],
                                 0, 255).astype(np.uint8)
                else:
                    fc = eq_front[:, s:e].copy()
                    bc = eq_back[:, s:e].copy()

                # Fill invalid pixels with the other lens's data before flow
                # so DIS doesn't try to match black regions to content.
                fc[~band_vf] = bc[~band_vf]
                bc[~band_vb] = fc[~band_vb]

                # Multi-scale DIS optical flow: compute at 1/4 resolution
                # for a smooth displacement field that captures coarse parallax
                # without mesh/texture-level noise.
                h, w = fc.shape[:2]
                flow_scale = 1
                h_s, w_s = h // flow_scale, w // flow_scale
                fc_small = cv2.resize(fc, (w_s, h_s))
                bc_small = cv2.resize(bc, (w_s, h_s))
                fg = cv2.cvtColor(fc_small, cv2.COLOR_BGR2GRAY)
                bg = cv2.cvtColor(bc_small, cv2.COLOR_BGR2GRAY)
                flow_small = self.flow_engine.compute(fg, bg)
                flow = cv2.resize(flow_small, (w, h)) * flow_scale

                # Clamp flow magnitude to physical limit
                mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
                max_flow = 40.0
                scale = np.where(mag > max_flow, max_flow / (mag + 1e-6), 1.0)
                flow[:, :, 0] *= scale
                flow[:, :, 1] *= scale

                # Zero flow outside the overlap (no correction needed there)
                flow[~band_overlap] = 0

                # Partial flow warp: each image moves halfway toward
                # the other to reduce parallax in the overlap.
                h, w = flow.shape[:2]
                ys, xs = np.mgrid[:h, :w].astype(np.float32)

                alpha = np.linspace(0, 1, w, dtype=np.float32).reshape(1, -1)
                alpha = np.broadcast_to(alpha, (h, w))

                fw = cv2.remap(fc, xs + alpha * flow[:, :, 0] * 0.5,
                               ys + alpha * flow[:, :, 1] * 0.5,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                bw_img = cv2.remap(bc, xs - (1 - alpha) * flow[:, :, 0] * 0.5,
                                   ys - (1 - alpha) * flow[:, :, 1] * 0.5,
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                # Seam finding + multi-band blending
                seam = find_seam_dp(fw, bw_img)
                blended = multiband_blend(fw, bw_img, seam,
                                          levels=5, feather_px=25)

                # Only write where both lenses are valid
                blend_mask = band_overlap[:, :, np.newaxis].astype(np.float32)
                orig = result[:, s:e].astype(np.float32)
                result[:, s:e] = np.clip(
                    blended * blend_mask + orig * (1 - blend_mask),
                    0, 255).astype(np.uint8)
        else:
            # Simple weighted blend (no flow)
            result = np.clip(
                eq_front.astype(np.float32) * self.w_front[:, :, np.newaxis] +
                eq_back.astype(np.float32) * self.w_back[:, :, np.newaxis],
                0, 255).astype(np.uint8)

        # Optional edge-preserving denoise (bilateral filter)
        if self.denoise:
            result = cv2.bilateralFilter(result, 9, 40, 40)

        dt = time.time() - t0
        print(f"Frame {frame_num}: {dt:.2f}s")
        return result

    def stitch_frame_to_file(self, frame_num, output_path):
        result = self.stitch_frame(frame_num)
        cv2.imwrite(output_path, result)
        print(f"Saved: {output_path}")
        return result

    def stitch_video(self, output_path, start=0, end=-1):
        """Process frames to video using ffmpeg pipe for output."""
        if end == -1:
            end = self.meta.frame_count

        # Use ffmpeg pipe for H.264 output (much better compression than mp4v)
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'warning',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{self.eq_w}x{self.eq_h}',
            '-r', str(self.meta.fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264', '-preset', 'medium',
            '-crf', '18', '-pix_fmt', 'yuv420p',
            output_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        t_total = time.time()
        for n in range(start, end):
            result = self.stitch_frame(n)
            proc.stdin.write(result.tobytes())

        proc.stdin.close()
        proc.wait()
        dt = time.time() - t_total
        n_frames = end - start
        print(f"Video saved: {output_path} "
              f"({n_frames} frames in {dt:.1f}s, {n_frames/dt:.1f} fps)")


# ============================================================
# 12. Ground Truth Comparison
# ============================================================

def extract_gt_frame(gt_path, frame_num=0):
    """Extract a frame from ground truth video."""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height', '-of', 'csv=p=0', gt_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    parts = [p for p in r.stdout.strip().split(',') if p]
    w, h = int(parts[0]), int(parts[1])

    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', gt_path,
           '-vf', f'select=eq(n\\,{frame_num})',
           '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(h, w, 3).copy()


def compare_psnr(our, gt):
    """Compute PSNR between two images (resizes gt to match)."""
    if our.shape != gt.shape:
        gt = cv2.resize(gt, (our.shape[1], our.shape[0]))
    mse = np.mean((our.astype(float) - gt.astype(float)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Insta360 X5 Stitching Pipeline')
    parser.add_argument('input', help='Path to .insv file')
    parser.add_argument('-o', '--output', default='output.jpg')
    parser.add_argument('-f', '--frame', type=int, default=0)
    parser.add_argument('-w', '--width', type=int, default=3840)
    parser.add_argument('--no-stab', action='store_true',
                        help='Disable IMU stabilization')
    parser.add_argument('--no-flow', action='store_true',
                        help='Disable optical flow stitching')
    parser.add_argument('--denoise', action='store_true',
                        help='Apply bilateral denoising (matches Insta360 Studio)')
    parser.add_argument('--video', action='store_true',
                        help='Process all frames to video')
    parser.add_argument('--start', type=int, default=0,
                        help='Start frame (for video mode)')
    parser.add_argument('--end', type=int, default=-1,
                        help='End frame (for video mode, -1 = all)')
    parser.add_argument('--gt', help='Ground truth video for comparison')
    args = parser.parse_args()

    pipeline = X5Pipeline(args.input, eq_width=args.width,
                          enable_stabilization=not args.no_stab,
                          enable_flow=not args.no_flow,
                          denoise=args.denoise)

    if args.video:
        pipeline.stitch_video(args.output, args.start, args.end)
    else:
        result = pipeline.stitch_frame_to_file(args.frame, args.output)
        if args.gt:
            gt = extract_gt_frame(args.gt, args.frame)
            psnr = compare_psnr(result, gt)
            print(f"PSNR vs ground truth: {psnr:.2f} dB")
