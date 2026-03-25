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

## .insv Footer Format

The footer is appended after the MP4 data. Structure (reading from end of file):

```
[...MP4 data...][...extra data sections...][entry table][total_size:4 LE][count:4 LE][magic:32 ASCII]
```

### Magic
Last 32 bytes of file: ASCII string `8db42d694ccc418790edff439fe026bf`

### Top-level footer
- Bytes -36 to -33: entry count (uint32 LE). Observed value: 3
- Bytes -40 to -37: total extra data size (uint32 LE, e.g. 10,460,715 for VID_005)

### Entry table
10-byte entries with `d8 cc` marker, found by scanning backwards from the footer:

```
[id: 2 bytes LE] [size: 4 bytes LE] [marker: d8 cc] [counter: 2 bytes LE]
```

Data for each entry is located at: `entry_file_position - size`

### Observed entries (VID_005, 0.57 seconds)

| ID | Size | Description |
|----|------|-------------|
| 0x02 | 1,228,840 | Unknown (large, possibly thumbnail/preview) |
| 0x03 | 16,640 | Raw IMU data (gyro + accelerometer) |
| 0x04 | 416 | Protobuf: lens protector configurations |
| 0x09 | 912 | Protobuf: unknown repeating structure |
| 0x0b | 5,073 | Raw 16-bit data (constant, possibly calibration offsets) |
| 0x16 | 8,306,836 | Unknown (large) |
| 0x1c | 12,793 | Unknown (zeros observed at start) |
| 0x1d | 54 | Small index/pointer table |

### Entry 0x04: Protobuf lens configurations
Contains named lens protector profiles with associated calibration adjustments:
- `bare`: no protector
- `ProtectorA`, `ProtectorS`, `ProtectorAS`: physical lens protectors
- `InvisibleDiveWater`, `InvisibleDiveAir`: underwater housing modes

Each profile includes FOV and calibration offset doubles.

## IMU Data (Entry 0x03)

### Format
- **Encoding**: Raw 16-bit unsigned integers (uint16 LE)
- **NOT** the documented double-precision format from older Insta360 models
- 8,320 uint16 values for a 0.57-second clip = ~14,596 values/second
- Values cluster around 30,600-31,400 (slowly varying, consistent with stationary camera)

### Unknown parameters
- Record stride (how many uint16 values per sample), no clear channel separation found
- IMU zero-offset and sensitivity (counts per rad/s, counts per m/s^2)
- Which values are gyroscope vs accelerometer
- Timestamp encoding (if any; may be implicit at fixed sample rate)

### Entry 0x0b: Related IMU data?
- 2,536 uint16 values, constant at 35,703 (0x8B77)
- Possibly static calibration offsets or data from the other lens's IMU

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

1. **Distortion model**: What is the correct projection model for the X5's k1-k4 coefficients?
2. **IMU decoding**: What is the record structure, sample rate, and calibration of the raw 16-bit IMU data?
3. **Photo orientation**: Photos don't have a matching .pb file. Is the gyro/orientation data embedded in the JPEG EXIF?
4. **Per-frame orientation**: For video, does orientation change per frame (requires gyro integration) or is it constant?
5. **GT stitching approach**: What blend/seam strategy does Insta360 Studio use to achieve clean stitching at distance?
