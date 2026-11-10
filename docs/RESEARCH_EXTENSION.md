# GPU optimisation and degradation-aware reliability experiments

## Scope and evidence

This branch extends the audited baseline without replacing its architecture defaults,
logistic checkers, calibration, sequential model loading, or existing corruption suite.
CUDA AMP already existed. It is now explicit and recorded; this is not a new AMP claim.
No real retinal results, GPU speedups or lesion-preservation claims are supplied.

Implemented: explicit precision controls, profiling, prepared-image cache, lightweight
architecture options, read-only dataset audit, donor-conditioned sampling, frozen
representation extraction, representation-distance checker ablation, natural/synthetic
report separation and patient bootstrap grading intervals.

Not established: your dataset integrity, real natural degradation labels, pretrained
I-JEPA/DINOv2 checkpoint parity, CUDA memory on RTX A400, clinical suitability of
retake rules, or improved performance. No huge checkpoint is downloaded automatically.
There is no end-to-end differentiable generator or semantic consistency training loss.

## Windows setup and pilot

Run from the repository after activating `.venv`. Preserve your matched CUDA torch and
torchvision installation. No new required dependency was added to requirements.txt.

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python smoke.py --out runs/smoke-new
```

First inventory the ZIP without extracting it (ZIP only; nested ZIP/TAR files are not
expanded). The output contains exactly matched members for planning selective extraction.
Do not use `inventory_only` as a completed content audit.

```powershell
python -m dr.audit --labels data/EyeQ/data/Label_EyeQ_train.csv data/EyeQ/data/Label_EyeQ_test.csv --archive data/diabetic-retinopathy-detection.zip --out runs/archive-audit.json
python -m dr.audit --labels data/EyeQ/data/Label_EyeQ_train.csv data/EyeQ/data/Label_EyeQ_test.csv --images data/EyePACS --out runs/dataset-audit.json
python -m dr prepare-eyeq --labels data/EyeQ/data/Label_EyeQ_train.csv data/EyeQ/data/Label_EyeQ_test.csv --images data/EyePACS --out data/eyeq.csv
```

Review missing/ambiguous names, duplicate decoded pixels, label encoding and unreadable
files before preparing the manifest. The audit does not detect near duplicates or prove
release provenance. Independently reconcile EyePACS grades where available. The adapter
still deliberately combines official EyeQ label files into new patient-grouped splits;
this is not the official EyeQ benchmark split. No training automatically follows an audit.

Starting configuration (not a guarantee of fitting in 4 GB):

```powershell
python run_pipeline.py --manifest data/eyeq.csv --out runs/r50-pilot-320 --arch r50 --size 320 --batch 2 --accumulate 4 --workers 2 --device cuda --precision fp16 --seeds 11 --epochs 5 --test-mode natural --cache-dir data/prepared-cache
```

Keep validation/calibration FP32. `--precision auto` preserves CUDA FP16 training and
CPU FP32. Inference defaults to FP32; opt in with `--inference-precision fp16` in the
runner or `--precision fp16` in `features`/`predict`. Feature metadata records non-FP32
inference; checkers reject mismatched inference protocols. BF16 explicitly checks device
support. GradScaler is used for FP16 only.

If memory fails, create a NEW run with batch 1/accumulate 8; if needed use size 256.
Do not call these equivalent to a physical batch of 8: BatchNorm still sees physical
microbatches. Accumulation retains the legacy mean-of-microbatch-loss objective, including
its class-weight/presence denominators. The final partial accumulation window is handled.

`*.pt.json` records settings, software, parameter counts, elapsed time, epoch history,
checkpoint size and peak allocated/reserved CUDA bytes. CPU memory fields are null.
Feature sampling sidecars record runtime and generation settings. Timing includes the
training loop, validation and calibration, but not initial model construction or saving.
Workers remain nonpersistent so epoch-dependent randomness is not accidentally frozen.

```powershell
python -m dr.benchmark --device cuda --arch r50 --size 320 --batch 2 --accumulate 4 --precisions fp32 fp16 --out runs/benchmark-r50-320.json
```

The benchmark uses random tensors/weights, warms up AdamW, synchronizes CUDA, and reports
training throughput plus amortized inference time per image. It excludes I/O and is not
clinical evidence or single-request latency. Its inference memory includes resident
training optimizer state; use feature-extraction profiles for deployment-stage memory.
Report cold preprocessing and warm-cache end-to-end times separately. Out-of-memory does
not silently alter the configuration.

## Architectures and resolution

`r50` remains the default. `r18`, `mobilenetv3` (Large), and `efficientnetb0` are explicit
experimental grader options. Quality remains R50 and damage remains R18. They share
existing heads, manifests, normalisation, optimization and checkpoint interfaces.
Use the same manifest, seeds, training budgets and evaluation subsets. Equal epochs
across robust and nonrobust training are not equal compute: robust duplicates Good sources.

For corruption-based resolution comparisons, explicitly use `--corruption-size 384`
for ALL arms. Corruptions are applied at that common size and then resized to 256/320/384.
Uncorrupted originals retain ordinary direct preprocessing. Without this option, legacy
pixel-based corruption strength is preserved; it varies relative to retinal size across
resolutions. Validate small-lesion sensitivity using lesion annotations/regions, not
aggregate AUROC alone. Retain source DR labels separately from degraded-image gradability.

## Degradation-conditioned generation

The estimator has eight multi-label presence outputs and eight severities, trained on
synthetic operations. EyeQ quality is not degradation-family ground truth.

A donor profile is an explicit JSON list of naturally degraded TRAINING patients. Manual
labels use `natural_manual`; model estimates use `natural_estimated`. They are never
silently pooled into ground truth. Example schema (identifiers must exist in your manifest):

```json
{"version":1,"donors":[{"patient":"123","image":"123_left.jpeg","split":"train","provenance":"natural_manual","spec":[{"operator":"defocus","level":2},{"operator":"dark","level":3}]}]}
```

Levels are the simulator's discrete 1–5 settings, not a validated clinical scale. Specify
one to three distinct families. Profile sampling preserves donor combinations; exclusion
removes complete incompatible donor specifications rather than silently dropping one defect.

For a first weakly supervised profile, use a baseline damage checkpoint from the same
manifest. It evaluates only original Usable/Reject training images:

```powershell
python -m dr.conditioning --manifest data/eyeq.csv --checkpoint runs/r50-pilot-320/damage.pt --device cuda --batch 2 --exposure-operator dark --noise-operator noise --out runs/natural-estimates.json
```

The two operator options are REQUIRED because the existing estimator cannot distinguish
bright vs dark or noise vs JPEG. These choices are assumptions, not inferred parameters.
The default presence threshold is 0.5; at most three strongest families are retained.
Continuous severity is rounded to discrete simulator levels. Inspect estimated profiles
against independently annotated natural images before interpreting them scientifically.
Motion angle and spot location still come from the legacy seeded sampler.

```powershell
python run_pipeline.py --manifest data/eyeq.csv --out runs/conditioned-320 --arch r50 --size 320 --batch 2 --accumulate 4 --workers 2 --device cuda --seeds 11 --robust --condition-profile runs/natural-estimates.json --corruption-size 384 --fit-mode balanced-real --test-mode natural --cache-dir data/prepared-cache
```

This applies conditioning to damage training, robust grader training and fit/policy
synthetic views. Test sampling stays independent. To isolate effects, use individual CLI
stages and change only one of these stages at a time. Never estimate a donor profile from
test patients. Stage-level `--condition-profile` also supports manual profiles.

For real annotation: label each family independently with absent/present/uncertain,
record severity and affected retinal regions, keep quality and lesion visibility separate,
use two masked readers with adjudication and agreement reporting, and retain missing labels
rather than treating them as negatives. Direct partially labelled natural-estimator training
is not implemented in this branch; do not invent exact degradation targets from EyeQ.

## Frozen representation experiments

Start with the already-trained grader. This adds no backbone download:

```powershell
python -m dr.representations --manifest data/eyeq.csv --checkpoint runs/r50-pilot-320/r50-11.pt --encoder grader --size 320 --split test --mode balanced-real --device cuda --batch 1 --out runs/embeddings/test.npz
```

The NPZ, row CSV and JSON form a fingerprinted offline cache. `.pairs.csv` contains cosine
and L2 distances only for same-source clean/synthetic pairs. Natural poor images never
receive an invented clean reference. Pooling can hide local lesions; scores are exploratory.
Join paired rows with grading features by image/variant and assess grade consistency,
referable probability change, error, calibration and independently annotated lesion visibility.

Optional local checkpoints:

| Encoder | Expected checkpoint | Input | License / limitation |
|---|---|---|---|
| `ijepa` | Official I-JEPA ViT-H/14 checkpoint with `target_encoder` | 224 | CC BY-NC 4.0; large; no automatic download |
| `dinov2` | Official DINOv2 ViT-S/14 non-register backbone state dict | 224 | Apache-2.0; ~21M parameters |
| `grader` | This project's grading checkpoint | Checkpoint size | Existing backbone baseline |

I-JEPA's 32-block, width-1280, no-CLS encoder is instantiated with timm and loaded strictly;
mean pooling follows final token normalization. DINOv2 uses timm's dynamic position
resampling from its native 518 grid. Exact official-checkpoint output parity remains to be
validated; unit tests only exercise a small structural analogue. Run frozen, one model at
a time; batch 1 and FP16 may reduce memory. Neither this nor 224-pixel input proves that
small DR lesions are represented. Frozen models still occupy parameter/activation memory.
Official full checkpoint files can include training state and exceed inference weight size;
check remote file size before downloading. No claim of 4 GB feasibility is made.

Sources: https://github.com/facebookresearch/ijepa (including LICENSE and model code),
https://github.com/facebookresearch/dinov2 .

For a deployable *single-image* exploratory baseline, extract train, fit, policy and test
embeddings with the same encoder/preprocessing, using modes natural, balanced, balanced,
natural respectively to match the ordinary runner. Then augment each feature table:

```powershell
python -m dr.semantic_features --features runs/r50-pilot-320/fit.csv --embeddings runs/embeddings/fit.npz --reference runs/embeddings/train.npz --out runs/semantic/fit.csv
```

Repeat for policy/test. The reference uses Good ORIGINAL training images only. It adds
cosine distance to the normalized training centroid and deviation of embedding norm.
It blocks reference/evaluation patient overlap and mismatched protocols. These features
measure training-distribution distance, not verified lesion damage. This first version
supports partitions of the same development manifest; external-cohort joining is rejected.
Sampling seed, mode and reference corruption size must match the base feature file.
Conditioned/excluded view joins are rejected until their exact specifications can be replayed.
Fit/evaluate using augmented files through existing commands. `P6S` is an extra logistic
checker over U+D+Q plus these two features; P6 and deployed routing remain unchanged.
Augmented checkers are offline experiments; standard `predict` intentionally rejects their
metadata rather than silently omitting representation computation.

Do not optimize a consistency loss yet. Current Pillow/NumPy/SciPy corruptions are not
end-to-end differentiable. A future loss needs an actual trainable parameterization and
an independently justified lesion-visibility gate; severe degradation must be allowed to
increase uncertainty rather than be forced to preserve the original prediction.

## Controlled ablations

| Arm | Change from its matched comparator |
|---|---|
| E0 | Audited original commit, archived outputs |
| E1 | Systems changes only, same model/data/precision; optional inference AMP separately |
| E2 | One lightweight architecture at a time |
| E3 | Random robust augmentation vs clean-only grading |
| E4 | Conditioned vs random generation with matched views/updates |
| E5 | Frozen representation evaluation; no training feedback |
| E6 | Conditioned generation plus representation evaluation/optional P6S |
| E7 | Natural-quality reliability, calibration and deferral comparisons |

First use seed 11 for exploratory feasibility, then freeze choices and use multiple seeds
for confirmatory experiments. Keep the same patient manifest. Donor fitting belongs only
to training patients; selection/calibration/fit/policy have their original distinct roles.
Use independently selected natural test cohorts and separately report synthetic conditions.
Do not pool 51 synthetic views of Good images into a claimed natural population estimate.

Evaluation now adds natural/synthetic groups, exact score-tie endpoint risk–coverage CSVs,
and patient-bootstrap grading confidence intervals (`--bootstrap`, default still 0).
AUGRC intervals remain paired policy comparisons. Record finite bootstrap replicate counts
when rare classes make AUROC undefined. Compare P6S/P6 with a prespecified paired analysis
before making an incremental-benefit claim; this branch does not claim significance.
Keep rank coverage distinct from deployment fixed-threshold coverage. Retake remains a
heuristic without prospective retake outcomes or clinical safety guarantees.

## Changes, trade-offs and tests

| Files | Before → change / benefit | Downside / tests |
|---|---|---|
| training/runtime/CLI | Implicit CUDA AMP → explicit controls, logs, timings, safe outputs | CPU rejects CUDA-only precision; accumulation tail tests; CUDA tests still pending |
| images/data | Repeated prepare → optional lossless PNG cache | Disk use; source path/size/mtime/version key, not content hash; pixel equivalence tests |
| features | FP32-only → opt-in AMP, nonblocking input copies | Numerical changes require calibration/routing agreement; defaults unchanged |
| networks/CLI/runner | R50/ViT/RETFound grading → optional R18/MobileNet/EfficientNet | Accuracy trade-off; head shape and R18 restore tests |
| audit | Basename matching → inventory and content audit report | Near duplicates/provenance remain manual; duplicate/missing fixtures |
| conditioning/data | Random-only → explicit training-donor sampling | Synthetic-estimator bias; leakage/exclusion tests |
| representations | No cache → frozen embedding export and paired distances | Large checkpoints/clinical validity unverified; pooling/pair tests |
| semantic_features/policy | U/D/Q → optional P6S research comparison | Centroid shift is not semantic truth; reference/patient safeguards |
| evaluation | Pooled reports → explicit natural/synthetic reports, curves and wider CIs | Bootstrap runtime; smoke integration |

Existing outputs are protected by default. Individual writers accept `--overwrite` where
applicable; the runner still requires an empty output directory. Persistent workers,
gradient checkpointing, GANs, diffusion, and automatic resolution/batch fallback were not
added. They need evidence of benefit or would change the experimental question.
