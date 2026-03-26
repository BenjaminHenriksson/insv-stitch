# Insta360 X5 Stitching Pipeline

Linux stitcher for raw Insta360 X5 footage. Reads `.insv` files (two H.265 fisheye streams, IMU samples, a protobuf calibration sidecar) and produces stabilized equirectangular stills or video. No Insta360 Studio required.

PSNR against Studio's own output: 22.5 to 22.9 dB at 7680×3840.

Insta360 Studio is closed source and Windows/macOS only. Reproducing its output on Linux meant working out the `.insv` container, the protobuf-encoded MEI calibration, the IMU axis convention, and the stitching and blending math. The result is a single-file pipeline of about 1,200 lines that matches the reference within a dB.

## Files

- `x5_pipeline.py`. The pipeline.
- `PIPELINE.md`. Architecture: the twelve stages, the MEI model, the blending math, the known limitations.
- `x5_pipeline.md`. Longer notes: container format, IMU axis calibration, rolling shutter, optical flow experiments.
- `old/`. First implementation, plus `FINDINGS.md` with the reverse-engineering notes the rewrite is built on. See `old/README.md`.

## Architecture

Everything fuses into one backward remap per output pixel (following the pattern from the Insta360 SDK and Qualcomm's stabilization patent):

1. Parse the `.insv` into two H.265 streams and an IMU track.
2. Parse the `.pb` sidecar for MEI calibration (xi = 2.0, 13 distortion coefficients per lens, per-lens extrinsics).
3. Derive per-frame stabilization from IMU gravity.
4. Derive per-scanline rolling-shutter rotations, 32 SLERP keyframes across a 21 ms readout.
5. For each output pixel: ray, stabilize, transform into the lens frame, MEI-project, distort, sample.
6. Blend on longitude preference times coverage depth. No hardcoded feather width.
7. Symmetric per-channel gain across the seam.
8. Optional DIS optical flow for close-range parallax. Optional bilateral denoise.

Full treatment in `PIPELINE.md`.

## Install

Python 3.12+, with `ffmpeg` and `ffprobe` on `PATH`.

```bash
uv sync
# or
pip install -e .
```

## Usage

```bash
# single frame, full resolution, with denoising
uv run python x5_pipeline.py input.insv -o output.jpg -w 7680 --denoise

# full video
uv run python x5_pipeline.py input.insv -o output.mp4 -w 3840 --video

# stabilization off (required on un-calibrated hardware, see below)
uv run python x5_pipeline.py input.insv --no-stab -o output.jpg

# PSNR against a Studio-rendered reference
uv run python x5_pipeline.py input.insv --gt studio_render.mp4 -o output.jpg
```

The `.insv` needs to sit inside the camera's default layout:

```
DCIM/Camera01/VID_xxx_00_001.insv
MISC/Camera01/VID_xxx_00_001.insv.pb
```

## Limitations

IMU calibration is camera-specific. The `IMU_TO_CAM` rotation in `x5_pipeline.py` was solved via Wahba's method against ground-truth gravity on one X5 unit. Unit-to-unit PCB mounting variation will degrade stabilization on other cameras. Pass `--no-stab`, or re-solve against a Studio render from your own hardware.

Close-object parallax. Around 18 px of ghosting at the stitch line for objects under 3 m, a function of the 30 mm inter-lens baseline. DIS flow helps but does not match Insta360's learned `ai_stitch_model_v2.ins` on repetitive patterns like fence mesh or foliage.

Per-frame ffmpeg decode. Each frame spawns its own ffmpeg process, about 2 s of overhead. Piped batch decoding is the obvious next step for video throughput.

## License

MIT. See `LICENSE`.
