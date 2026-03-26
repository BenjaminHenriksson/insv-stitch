# old/

First implementation, kept for reference. Not wired into the current pipeline.

- `stitch.py`. First-pass stitcher. Separate undistort, remap, DP seam-find, and multiband blend passes (double resampling). Superseded by the single-remap design in `../x5_pipeline.py`.
- `compare.py`. Standalone ground-truth PSNR harness. Folded into the main pipeline.
- `FINDINGS.md`. Reverse-engineering notes on the `.insv` container, the `.pb` sidecar, the base64 calibration string, the MEI parameters, and the IMU axis convention. The rewrite is built on these.
- `pyproject.toml`. Older dep set, before `telemetry-parser` and `opencv-contrib-python`.
