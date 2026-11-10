# Validation performed for this branch

Environment: Linux, Python 3.12, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu,
timm 1.0.15. This differs from the user's Windows CUDA runtime.

- Unit suite: 18 tests discovered, 17 passed, 1 CUDA-only test skipped.
- Existing `smoke.py`: passed real CPU training steps, calibration, features, checker
  fitting, held-out evaluation and single-image prediction on generated images.
- Extended CLI run: passed R18 grading, robust training, donor-conditioned sampling,
  common corruption resolution, prepared-image caching, gradient accumulation,
  natural+synthetic fitting and patient-bootstrap reporting using synthetic fixtures.
- Representation integration: passed frozen grader embedding export, training-only
  reference construction, P6S fitting and held-out evaluation using generated images.
- CPU benchmark command: passed with R18, random tensors and FP32.
- Compile check and `git diff --check`: passed.

The donor-profile smoke fixture uses generated images and an explicitly permissive
threshold solely to exercise command wiring. It is not evidence of natural degradation
recognition or a recommended research threshold.

Not run: CUDA numerical/memory/speed tests, Windows worker tests, real EyePACS/EyeQ audit,
real-data pilot, pretrained I-JEPA/DINOv2 checkpoint parity, lesion-level validation,
architecture/resolution comparisons or confirmatory ablations. No clinical metrics or
speedup claims are inferred from smoke tests. No pretrained weights were downloaded.
