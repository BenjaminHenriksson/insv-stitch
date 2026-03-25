# Insta360 X5 Processing: Findings

## File Formats

### .insv (Video)
- MP4 container (ISO/IEC 14496-12), ftyp `avc1`
- **3 streams**: 2x HEVC 3840x3840 (one per lens) + 1x AAC 48kHz stereo
- Stream 0 = back lens, Stream 1 = front lens
- Extra data appended after MP4 as a binary footer (see [Footer Format](#insv-footer-format))

### .insp (Photo)
- JPEG, dual fisheye side-by-side: 11904x5952 (5952x5952 per lens)
- Left half = back lens, right half = front lens
- No footer/trailer observed (gyro data TBD)

### .pb (Calibration)
- Protobuf files in `MISC/Camera01/`, named `{original_filename}.pb`
- Only exist for videos (VID and LRV), not photos
- Contain base64-encoded calibration blocks with per-lens parameters

### .lrv (Low-Resolution Video)
- Low-res proxy of corresponding VID file
- Has its own `.pb` calibration file

## Calibration Format

Detailed calibration is base64-encoded inside the .pb protobuf. Once decoded, it's a `_`-delimited string with the format `2_<lens1_fields>_2.000000_<lens2_fields>`.

### Per-lens field layout (Tier 3, 27 fields)

| Index | Field | Example (back lens, ref res) |
|-------|-------|------------------------------|
| 0 | calib_version | 2.000000 |
| 1 | fx | 4271.090 |
| 2 | fy | 4272.200 |
| 3 | cx | 2680.960 |
| 4 | cy | 2680.490 |
| 5 | yaw (degrees) | -0.132 |
| 6 | pitch (degrees) | 0.434 |
| 7 | half_fov (degrees) | 89.816 |
| 8-10 | Rodrigues rotation (x,y,z) | 0, 0, 0 |
| 11-14 | k1, k2, k3, k4 | 0.2199, 1.6416, -1.4439, -2.6206 |
| 15 | zero | 0 |
| 16-17 | p1, p2 (tangential) | -0.000536, -0.001321 |
| 18-21 | s1, s2, s3, s4 (thin prism) | -0.00148, 0.00080, 0.00204, 0.00194 |
| 22-23 | tauX, tauY (tilted sensor) | 0.02581, 0.00293 |
| 24-25 | ref_width, ref_height | 10752, 5376 |
| 26 | unknown | 113 |

Reference resolution is the full dual-fisheye frame (10752x5376). Per-lens reference width = 10752/2 = 5376. Scale to actual stream resolution: `scale = stream_width / 5376`. For videos: `3840 / 5376 = 0.7143`.

cx for the front lens is stored in the full-frame coordinate system and must be converted: `cx_local = cx_ref - ref_width/2`.

### Distortion model: UNKNOWN

The k1-k4 coefficients do **not** work with the standard OpenCV fisheye theta polynomial:
```
theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
```
This diverges at theta > 40 degrees with the X5's coefficients. Testing with `cv2.fisheye.projectPoints` confirmed: the projection blows up beyond ~35 degrees, yet the lens covers 90 degrees per side.

The `cv2.fisheye.undistortPoints` inverse mapping shows that r=1915px (image edge) maps to only theta=34.73 degrees in the OpenCV fisheye model, far less than the actual 89.8 degree half-FOV.

**Current workaround**: Pure equidistant projection `r = f_equi * theta` with `f_equi = inscribed_radius / half_fov_rad`. This gives reasonable results but doesn't use the distortion coefficients. Optimal FOV parameter is ~200-204 degrees.

**Status**: Figuring out the correct model that maps these k1-k4 values to the full 90-degree FOV remains an open problem. The coefficients likely operate in a transformed coordinate space or use a non-standard polynomial form.

## .insv Footer Format: SOLVED

Confirmed via telemetry-parser source code (AdrianEddy/telemetry-parser).

### Structure (reading backwards from end of file)

```
[...MP4 data...][extra data section: extra_size bytes][padding:32][extra_size:4 LE][version:4 LE][magic:32 ASCII]
```

**Footer** (last 72 bytes): `[32 zero padding][extra_size: uint32 LE][version: uint32 LE][magic: 32 ASCII]`

### Records stored backwards

Records are stored from the end of the extra data section, growing backwards. Each record has a **6-byte trailer** after its data:

```
[record data: size bytes][format: 1 byte][id: 1 byte][size: 4 bytes LE]
```

The last record (closest to footer padding) is always **id=0 (Offsets)**, the index table. Parsing starts by reading the Offsets record first, at `file_end - 78 + 1`.

### Offsets record (id=0)

Contains 10-byte entries mapping record IDs to positions within the extra section:
```
[id: 1 byte][format: 1 byte][size: 4 bytes LE][offset: 4 bytes LE]
```

- `offset` is relative to `extra_start` (= `file_size - extra_size`)
- `format`: 0=binary, 1=protobuf

### Record types

| ID | Name | Format | Content |
|----|------|--------|---------|
| 0 | Offsets | binary | Index table (always last record) |
| 1 | Metadata | protobuf | Camera info, calibration, gyro config |
| 2 | Thumbnail | binary | H.264 video frame |
| 3 | Gyro | binary | IMU data (accel + gyro) |
| 4 | Exposure | binary | Shutter speed per frame |
| 9 | AAAData | binary | Auto-exposure data |
| 10 | Anchors | binary | Highlights |
| 11 | AAASimulation | binary | Unknown |
| 22 | (large) | binary | Unknown (8.3MB for 0.57s clip) |
| 28 | (unknown) | binary | Unknown |
| 29 | (unknown) | binary | Small (54 bytes) |

### VID_005 offsets table (31 entries, 310 bytes)

```
id= 1 fmt=1 size=3995      offset=10456326   (metadata protobuf)
id= 2 fmt=0 size=1228840   offset=9227480    (thumbnail)
id= 3 fmt=0 size=16640     offset=9096408    (IMU: 832 records × 20 bytes)
id= 4 fmt=0 size=416       offset=8965336    (exposure)
id= 9 fmt=0 size=912       offset=8834264    (AAA data)
id=22 fmt=0 size=8306836   offset=314584     (unknown, large)
```

## IMU Data: SOLVED via telemetry-parser

The `telemetry-parser` Python package (v0.3.0, `uv pip install telemetry-parser`) successfully parses X5 IMU data.

### Usage
```python
from telemetry_parser.telemetry_parser import Parser
p = Parser("/absolute/path/to/file.insv")
print(p.camera, p.model)  # "Insta360", "Insta360 X5"
imu = p.normalized_imu()  # list of dicts with timestamp_ms, accl, gyro, magn
meta = p.telemetry()[0]['Default']['Metadata']
```

### IMU configuration (from protobuf metadata)
- **acc_range**: 32 (±32g accelerometer)
- **gyro_range**: 2000 (±2000 deg/s gyroscope)
- **gyro_calib**: 6 floats = gyroscope bias offsets (rad/s): `[7.4e-5, 0.00257, 0.01828, -0.00440, 0.00235, 0.00577]`
- **Sample rate**: ~1000 Hz (996 Hz measured for VID_005)
- **first_frame_timestamp**: ms offset from recording start to first video frame

### Normalized IMU output
Each sample: `{timestamp_ms, accl: (x,y,z), gyro: (x,y,z), magn: None}`
- Accel in m/s², gyro in deg/s
- Timestamps relative to first video frame (negative = before recording)
- Axis mapping for X5: "yzX" (applied internally by telemetry-parser)

### IMU → equirectangular axis mapping

Determined empirically by comparing brute-force rotation's implied gravity with IMU readings, validated on VID_004 (stationary, gravity maps to (0.023, -0.999, 0.036) ≈ world down):

```
equirect_x = -imu_z
equirect_y = -imu_y
equirect_z = -imu_x
```

Or: `g_equirect = (-accel_imu[2], -accel_imu[1], -accel_imu[0])` after telemetry-parser's "yzX" remap.

### Gravity-derived orientation per file

| File | Samples | Accel at t=0 (m/s²) | |a| | Pitch | Roll |
|------|---------|---------------------|-----|-------|------|
| VID_002 | 167,392 | (4.25, 8.12, -1.15) | 9.24 | 27.4° | 98.1° |
| VID_003 | 3,328 | (5.25, 7.55, 0.15) | 9.20 | 34.8° | 88.9° |
| VID_004 | 852,752 | (-0.40, 10.11, -0.24) | 10.12 | -2.3° | 91.4° |
| VID_005 | 832 | (10.76, 6.02, 6.54) | 13.95 | 50.4° | 42.6° |

VID_004 is the most level (pitch≈0°, roll≈90° = camera held vertically in selfie mode).
VID_005 was being actively moved (|a|=13.95 >> 9.81, high gyro rates).

## Orientation / Gyroscope

### Insta360 Studio behavior
- Applies gyroscope-based orientation correction during stitching
- Levels the horizon and sets the "forward" direction
- This rotation is NOT present in raw .insv files
- The correction is a 3D rotation (yaw + pitch + roll) applied to the equirectangular sphere

### Empirical orientation measurement
Using NCC-based search (sweeping yaw/pitch/roll, comparing against GT at half resolution):

**VID_005** (camera held by hand, significantly tilted):
- yaw = 202.48 degrees
- pitch = 48.63 degrees
- roll = -36.64 degrees
- NCC = 0.87, PSNR = 18.72 dB, MAE = 18.32

The ~180 degrees of yaw is the baseline lens swap (GT centers the back lens, not the front). The remaining ~22 degrees yaw + 49 degrees pitch + 37 degrees roll is the gyroscope correction for camera tilt.

### Implementation
`R_orientation` rotation matrix applied in `FisheyeRemapper._build_tables()`, rotates all 3D viewing directions before fisheye projection:
```python
P = np.einsum("ij,hwj->hwi", R_orientation, P)
```

## Stitching Pipeline

### Current architecture (stitch.py)
1. **Remap**: fisheye → equirectangular using per-lens calibration
2. **Optical flow** (optional): Farneback in overlap strips, attenuated by `4 * mask * (1 - mask)`
3. **Seam finding** (optional): edge-aware DP with color_diff + Sobel edge cost
4. **Blending**: distance-transform or seam mask composite

### Blending comparison (VID_005 frame 0, half-res, with orientation correction)

| Method | MAE vs GT | PSNR | Notes |
|--------|-----------|------|-------|
| Distance-transform blend (no flow) | 18.32 | 18.72 | Best overall |
| Flow + hard seam (7px feather) | 31.18 | 12.44 | Black streak, visible seam lines |
| Flow + smooth distance-transform | 18.38 | 18.68 | Flow doesn't help |
| Multi-band (Laplacian pyramid) | 18.34 | 18.71 | Negligible difference |
| Strip-based flow + multiband | 18.32 | 18.71 | Flow is faster (375ms vs 2631ms) but no quality gain |

### Key observations
- The **distance-transform blend** is the best simple approach: smooth gradient hides parallax
- **Optical flow** (Farneback) doesn't improve quality because:
  - Sub-pixel parallax at distance is below Farneback's resolution
  - The smooth blend already masks moderate parallax
  - Flow introduces artifacts where it's inaccurate
- **Seam finding + narrow feather** creates visible discontinuities
- Our output has **higher sharpness** than GT at the seam (Laplacian variance 1215 vs 857), indicating ghost/double-edges from parallax mixing

### Remaining quality gap vs GT
- **Distant thin structures** (powerlines, fences): slight geometric misalignment → doubled edges at seam. Root cause is likely the distortion model (k1-k4 not applied).
- **Close objects** (hand, body at seam): parallax ghosting, but GT handles this cleanly
- **Overall color**: mean pixel values match well (143.0 vs 143.2)

## Ground Truth

Files in `x5-ground-truth/` are Insta360 Studio exports:
- 1 photo: 11904x5952 JPEG
- 4 videos: 7680x3840 HEVC, 29.97fps
- All contain spherical mapping metadata (equirectangular projection)
- Encoded with `Lavf60.3.100`
- File names match raw files (same timestamp scheme)

## Open Questions

1. **Distortion model**: SOLVED. Unified Camera Model (UCM) with xi=2.0. MAE improved 18.32→16.80.
2. **Photo orientation**: .insp files have footers too (confirmed). telemetry-parser TBD.
3. **Gravity → rotation matrix**: Axis mapping solved. Gravity gives pitch/roll; yaw needs GT or magnetometer. For fast-moving cameras, gyro integration from a stationary window is needed.
4. **Sensor fusion**: Proper complementary/Kalman filter for accel+gyro fusion would improve orientation for moving cameras.
5. **GT stitching approach**: What blend/seam strategy does Insta360 Studio use to achieve clean stitching at distance?
