# Insta360 X5 Stitching Pipeline: Architecture and Documentation

## Overview

`x5_pipeline.py` is a from-scratch Linux implementation of Insta360 X5 dual-fisheye
stitching. It takes raw `.insv` files (two H.265 fisheye streams + embedded IMU
telemetry) and produces equirectangular 360° output.

**PSNR vs Insta360 Studio ground truth: ~22.5 dB at 1920×960, ~21.8 dB at 7680×3840**

---

## Architecture: Everything Is One Remap

Following the design principle from the Insta360 SDK and Qualcomm stabilization
patent: stabilization, rolling shutter correction, lens undistortion, and stitching
are fused into a **single backward-mapping** per output pixel. No double-resampling.

For every output equirectangular pixel `(u, v)`:

```
1. Convert (u, v) → 3D ray on unit sphere
2. Apply per-scanline stabilization rotation (rolling shutter corrected)
3. Transform ray into fisheye lens's local frame via R_extrinsic⁻¹
4. Project through MEI model with extended distortion → (src_x, src_y)
5. Sample the source fisheye image at (src_x, src_y)
```

---

## Pipeline Stages

### 1. Metadata Extraction (`extract_metadata`)

Extracts calibration and telemetry from the `.insv` file and its `.pb` sidecar:

- **Extended calibration** (preferred): 56-element string from the protobuf sidecar
  (`MISC/Camera01/*.insv.pb`), providing per-lens:
  - MEI mirror parameter (xi = 2.0)
  - Separate fx, fy, cx, cy at sensor resolution (5376×5376)
  - Per-lens yaw, pitch, roll (extrinsic rotation, small corrections)
  - Translation vector (tx, ty, tz, including the 32 mm inter-lens baseline)
  - 13 distortion coefficients: 4 radial (k1-k4) + 2 tangential (p1-p2) + 4 thin prism (s1-s4) + 2 extra

- **Fallback**: Gyroflow lens profile from telemetry-parser (5 coefficients, shared distortion)

- **IMU data**: `normalized_imu()` from telemetry-parser. Gyro (deg/s) and accel (m/s²) at ~200 Hz.

- **Video metadata**: frame rate, frame count, rolling shutter readout time (21.24ms)

**Resolution conversion**: Sensor coords (5376) → video coords (3840) via
`fx_video = fx_sensor × 3840/5312` and `cx_video = cx_sensor × 3840/5376`
(with cx_fix=2 for X5: cx is stored halved in Gyroflow convention).

### 2. IMU Stabilization (`compute_stabilization_from_imu`)

Gravity-based horizon locking:

1. Transform IMU accelerometer data to camera frame via `IMU_TO_CAM` matrix
2. Smooth accelerometer over ~50 samples (~250ms) to filter dynamic forces
3. Compute per-sample leveling rotation (Rodrigues rotation from gravity → [0,1,0])
4. Map to per-frame corrections by nearest-timestamp lookup

**IMU_TO_CAM matrix**: Transforms `normalized_imu()` output to the camera image frame
(X=right, Y=down, Z=forward). Currently derived via Wahba's method from GT-aligned
gravity vectors. This is a known limitation. See "Known Limitations" below.

### 3. Rolling Shutter Correction (`compute_rs_rotations`)

Per-scanline orientation from gyro integration:

1. For each of 32 evenly-spaced scanline positions across the readout (21.24ms):
   - Compute capture time: `t = t_frame + (scanline_frac - 0.5) × readout_time`
   - Integrate gyro angular velocity from frame center to scanline time
   - Compose with the base stabilization orientation
2. The remap builder interpolates between these 32 orientations per output row via SLERP

**Impact**: Corrects ~12-18px of displacement during typical handheld motion (+0.65 dB).

### 4. MEI Forward Projection (`mei_forward`)

Projects 3D rays to fisheye pixel coordinates through the extended MEI model:

```
1. Normalize ray to unit sphere: (Xs, Ys, Zs)
2. MEI mirror projection: x = Xs/(Zs + xi), y = Ys/(Zs + xi)
3. Extended distortion:
   radial = 1 + k1·r² + k2·r⁴ + k3·r⁶ + k4·r⁸
   xd = x·radial + 2·p1·x·y + p2·(r² + 2x²) + s1·r² + s2·r⁴
   yd = y·radial + p1·(r² + 2y²) + 2·p2·x·y + s3·r² + s4·r⁴
4. Camera matrix: u = fx·xd + cx, v = fy·yd + cy
```

xi = 2.0 for the X5 (hyperbolic mirror model, supporting >180° FOV).

### 5. Equirectangular Remap (`build_equirect_remap`)

Builds backward-mapping tables (map_x, map_y) for `cv2.remap()`:

1. For each output pixel → compute 3D ray (lon/lat → X,Y,Z with Y-down convention)
2. Apply per-scanline stabilization (RS-corrected, SLERP-interpolated)
3. Transform to lens-local frame via `R_extrinsic.T`
4. Project through MEI → source fisheye coordinates
5. Apply circular fisheye mask (5% margin from image circle edge)

### 6. Blending (`compute_blend_weights`)

**Longitude preference × coverage depth**, no hardcoded parameters:

```
w_front = longitude_pref(col) × distance_from_front_edge(row, col)
w_back  = (1 - longitude_pref(col)) × distance_from_back_edge(row, col)
normalize: w_front, w_back = w_front/(w_front+w_back), ...
```

- **Longitude preference**: Linear ramp from front-primary (|lon| < 75°) to
  back-primary (|lon| > 105°), transitioning at ±90°
- **Coverage depth**: `cv2.distanceTransform`. Pixels from the lens's
  validity boundary. A lens 24 px from its edge naturally gets less weight than
  one 162 px deep.

This handles:
- Hemisphere ownership (longitude)
- Coverage edge smoothness (no hard color steps at fisheye circle boundary)
- Close-object parallax reduction (favors the lens with more central coverage)

### 7. Color Harmonization

**Symmetric per-channel gain** in the blend zone:

1. Compute spatially-varying gain field from the overlap: `gain = front/back` per pixel, Gaussian-blurred (σ=80px)
2. Apply symmetrically: `front *= 1/√gain`, `back *= √gain`
3. Correction strength weighted by `2 × min(w_front, w_back)`. Full at seam center, zero in primary regions.

This preserves each hemisphere's natural exposure while smoothing the transition.

### 8. Optical Flow (Optional, `--flow`)

DIS optical flow for parallax correction in the stitch bands (±15° around ±90° longitude):

1. Extract stitch band crops from both gain-corrected lenses
2. Fill invalid pixels with the other lens's data
3. Compute DIS flow at full resolution
4. Partial warp: each image moves halfway (`×0.5`)
5. DP seam finding + 5-level Laplacian pyramid multi-band blending

**Impact**: Marginal improvement on close objects. The principled blending (coverage depth weighting) handles most parallax. Flow helps with fine structures like fence mesh at ~3m distance.

### 9. Denoising (Optional, `--denoise`)

Post-stitch bilateral filter (`cv2.bilateralFilter(d=9, sigmaColor=40, sigmaSpace=40)`):
- Preserves edges (ground texture sharpness matches GT)
- Smooths flat regions (sky noise matches GT)
- Chosen over NLM, which over-smooths texture

---

## Usage

```bash
# Single frame (all features)
uv run python3 x5_pipeline.py input.insv -o output.jpg -w 7680 --denoise

# Video
uv run python3 x5_pipeline.py input.insv -o output.mp4 -w 3840 --video

# Without stabilization
uv run python3 x5_pipeline.py input.insv -o output.jpg --no-stab

# With optical flow
uv run python3 x5_pipeline.py input.insv -o output.jpg --flow

# Compare to ground truth
uv run python3 x5_pipeline.py input.insv -o output.jpg --gt ground_truth.mp4
```

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| MEI model with xi=2.0 | Confirmed by Gyroflow source. Hyperbolic mirror handles >180° FOV |
| cx_fix=2 for X5 | Gyroflow convention halves cx; must double for actual optical center |
| Extended 13-coeff model | Protobuf sidecar has per-lens thin prism distortion that standard 5-coeff can't capture |
| Longitude × depth blending | Principled: no hardcoded feather distance, naturally favors central coverage |
| Symmetric gain correction | Prevents one-sided color step at the seam |
| Per-scanline RS correction | 12-18 px displacement during the 21 ms readout, same magnitude as parallax |
| Bilateral over NLM denoising | NLM destroys ground texture; bilateral preserves edges while matching GT noise floor |
| No per-channel CA correction | X5 has zero measurable chromatic aberration; applying CA scaling worsens PSNR |

---

## Known Limitations

1. **IMU_TO_CAM calibration**: Currently derived via Wahba's method from GT-aligned
   gravity vectors (2 videos). This is not fully principled. A self-calibrating AHRS
   filter (Madgwick/Mahony) or video-based orientation estimation would be better.

2. **Close-object parallax**: Objects <3m at the stitch line show ~18px ghosting from
   the 30mm inter-lens baseline. DIS optical flow partially corrects this but can't
   match Insta360's neural flow model (`ai_stitch_model_v2.ins`) on repetitive patterns
   like fence mesh.

3. **Per-frame ffmpeg decode**: Each frame spawns a separate ffmpeg process (~2s overhead).
   Pipe-based batch decoding would improve video throughput.

4. **Single-video IMU calibration**: The IMU_TO_CAM matrix was calibrated from video 003.
   Videos with very different camera dynamics may have reduced stabilization accuracy.

---

## File Dependencies

```
input.insv                          Raw dual-fisheye video
MISC/Camera01/input.insv.pb         Protobuf sidecar (extended calibration)
```

## Python Dependencies

```
numpy, opencv-contrib-python, scipy, telemetry-parser
```
