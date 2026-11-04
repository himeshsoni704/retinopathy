# Retinal grading and deferral

A local research implementation of the proposed pipeline in *Quality, Uncertainty or Task-Specific Reliability? Comparing Deferral Signals for Diabetic Retinopathy Grading Under Image Degradation*. It trains a grader, an image-quality assessor, a degradation estimator and a small model that predicts when that specific grader will be wrong.

The inputs are colour fundus photographs. A prediction returns the five disease probabilities, a separate probability of grade-based referable disease, the estimated error risk, and an accept/review/retake decision. The retake branch is exploratory. The proposal contains no experimental results, and this package includes source code only. Supply data and train the task-specific weights before using the prediction command.

## Install on the RTX workstation

Use Python 3.11 or 3.12. Run commands from the directory containing this README. Create and activate a virtual environment.

```bash
python -m venv .venv
```

Linux activation is `source .venv/bin/activate`. PowerShell activation is `.\.venv\Scripts\Activate.ps1`. Install a matched CUDA build of **torch and torchvision** using the PyTorch selector for the workstation's OS, GPU and driver. Then install the remaining dependencies.

```bash
python -m pip install -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

[PyTorch local installation](https://pytorch.org/get-started/locally/)

The runner accepts `--device cuda` to require an NVIDIA GPU. Its defaults are 384 px images, a physical batch of 8, and two gradient accumulation steps. If GPU memory is tight, try `--batch 4 --accumulate 4`. The RTX performs the neural network computation. The CPU prepares images and fits the logistic regressions. Ensemble members are loaded sequentially, so their model weights do not all need to fit on the GPU at once.

## Check the installation without any downloads

After installing the libraries, this smoke test generates artificial pictures, trains randomly initialised ResNet models for one epoch, calibrates the grader, fits the checkers, evaluates policies and predicts one image. It uses CPU and a fresh output directory. Its metrics have no medical interpretation.

```bash
python -m unittest discover -s tests -v
python smoke.py --out runs/smoke
```

## Prepare EyeQ

Obtain the original EyePACS photographs and both EyeQ label files. The EyeQ repository provides the labels. The original images are distributed separately. The adapter searches the supplied image tree recursively for the exact basenames in the EyeQ CSVs. Use a directory containing one copy of each image.

```bash
python -m dr prepare-eyeq --images data/EyePACS --labels data/EyeQ/data/Label_EyeQ_train.csv data/EyeQ/data/Label_EyeQ_test.csv --out data/eyeq.csv
```

[EyeQ labels and instructions](https://github.com/HzFu/EyeQ) · [EyePACS image source](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)

The adapter combines the rows and assigns **new** patient partitions for this experiment. It uses 20 stratified patient-group folds, allocated 12/1/1/2/2/2 to train, select, calibration, failure-fitting, policy-selection and test. Both eyes stay together. Exact image counts vary. Identifiers come from the EyePACS left/right filenames. The adapter checks label ranges, IDs, missing image paths and duplicate basenames. Audit duplicate image content, unreadable files and overlap across your actual releases before final research runs. Store a copy of the finished manifest with your experiment.

## Train and evaluate

A first real-data run can use one seed and original test photographs. This trains a complete pipeline while reducing the computation. With one seed, ensemble disagreement is zero apart from floating-point roundoff.

```bash
python run_pipeline.py --manifest data/eyeq.csv --out runs/pilot --arch r50 --seeds 11 --epochs 5 --device cuda --test-mode natural
```

For the full ResNet-50 ensemble, use the default five seeds, 50-epoch maximum and all 50 synthetic test variants per Good source. The optional bootstrap count below requests 2,000 paired patient resamples. This takes substantially more time than the pilot.

```bash
python run_pipeline.py --manifest data/eyeq.csv --out runs/r50 --arch r50 --device cuda --bootstrap 2000
```

Run the other architectures separately, each in a fresh directory. The transformer position embeddings are interpolated for the chosen image size.

```bash
python run_pipeline.py --manifest data/eyeq.csv --out runs/vit --arch vit --device cuda
python run_pipeline.py --manifest data/eyeq.csv --out runs/retfound --arch retfound --retinal-weights weights/RETFound_cfp_weights.pth --device cuda
```

ResNet-50 and ViT-B/16 use Torchvision ImageNet weights for initialisation. The retinal option expects the original **RETFound-MAE colour-fundus** ViT-L/16 PyTorch state dictionary, supplied locally. The example filename is the user's local filename. Only its prediction heads train. The loader checks required encoder tensors and interpolates the positional embeddings. Record the checkpoint version and any EyePACS pretraining overlap. The official repository also distributes other model families with different architectures.

[Torchvision model catalogue](https://docs.pytorch.org/vision/stable/models.html) · [RETFound official weights and code](https://github.com/rmaphoh/RETFound)

## What the runner executes

| Stage | Training inputs | Output |
|---|---|---|
| Quality | Training photographs with EyeQ Good, Usable or Reject labels | A ResNet-50 giving three probabilities |
| Damage | EyeQ-Good training images and on-demand labelled corruptions, including mixtures | A ResNet-18 with eight presence and eight severity outputs |
| Grading | EyeQ-Good training images and DR grades | One two-head grader per seed, selected by validation quadratic weighted kappa |
| Calibration | The calibration partition only | A fitted temperature per seed and head, stored in the grader checkpoint |
| Feature extraction | Frozen checkpoints on fit, policy and test partitions | Per-image CSVs with predictions, four uncertainty scores, eight degradation features and three quality features |
| Checker fitting | Error labels on fit patients | Separate exact-grade and binary-error models, with patient-grouped regularisation selection where feasible |
| Policy selection | Policy patients only | Frozen thresholds targeting 80% and 90% coverage, plus exploratory routing thresholds |
| Evaluation | Held-out test patients | Risk, coverage, grading metrics, subgroup deferral and per-image decisions |

Each image contributes grade probabilities from the original view. The TTA feature uses the original, horizontal flip, vertical flip and 180-degree rotation. Member probabilities are calibrated before averaging. The four uncertainty features are one minus maximum probability, entropy, ensemble mutual information and TTA entropy. For each defect family, the degradation feature is estimated presence probability multiplied by estimated severity.

The checker target is 1 for an error and 0 for a correct prediction. A second target compares the binary head's decision at 0.5 with the reference grade being at least 2. Logistic regressions learn the two tasks separately. Retrain the checkers whenever the grader ensemble or auxiliary models change. Checkpoint digests are recorded with features and enforced when scoring.

## Predict one image

After the one-seed pilot, use its checkpoints together. For the full run, list all five grader files after `--graders`.

```bash
python -m dr predict --image data/example.jpeg --graders runs/pilot/r50-11.pt --quality runs/pilot/quality.pt --damage runs/pilot/damage.pt --checker runs/pilot/checker.joblib --coverage 0.8 --device cuda --workers 0
```

The JSON contains the provisional grade and probabilities, defect scores, quality scores, both error estimates, route, and a reported grade. The reported grade is null when the grade checker defers. A grade accepted as reliable may itself indicate referable disease. Trusting a grading prediction and predicting referable disease are separate decisions.

## Inspect the results

| File under the run directory | Contents |
|---|---|
| `fit.csv`, `policy.csv`, `test.csv` | Image/patient IDs, variant, reference labels, features and predictions |
| `*.meta.json`, `*.sampling.json` | Model fingerprints, architecture, manifests, seed and view sampling settings |
| `checker.joblib` | Fitted checkers, scalers and policy thresholds |
| `reports/policies.csv` | AUGRC, AUROC-F, AUPRC-F, full-coverage error, accepted errors, missed referrals, score calibration |
| `reports/grading.csv` | Five-grade and binary metrics per group |
| `reports/strata.csv`, `reports/overlap.csv` | Deferral by grade and quality, plus Jaccard overlap |
| `reports/decisions.csv`, `reports/retakes.csv` | Individual decisions and observed referable share in exploratory retake groups |
| `reports/bootstrap.csv` | Optional paired intervals for P6 minus P2/P3/P5 AUGRC on evaluable images |

In `policies.csv`, `rank` compares approximately the same number of accepted cases within each group. `fixed` applies the threshold chosen on the policy partition and reports actual coverage. Ties at fixed thresholds are accepted inclusively. Rank comparisons break ties by a seeded hash. AUGRC uses trapezoidal integration and averages over tied scores. Per-family and per-variant reports keep operator differences visible. The pooled `all` rows describe the supplied sample, including its mix of variants. For a family-balanced summary, average the family results with one vote per family.

P0 is seeded random ranking, P1 uses confidence or entropy, P2 uses TTA entropy, P3 uses predicted image quality, P4 averages standardised uncertainty, P5 learns from uncertainty and quality, P6 adds estimated degradation, and P7 uses reference errors as an oracle. Extra U and UD fits isolate uncertainty alone and uncertainty plus degradation.

## Evaluate an external dataset

Supply one CSV per cohort with the same six required columns. Paths can be absolute or relative to the CSV. `quality` uses 0=Good, 1=Usable, 2=Reject and -1=unknown. `grade` uses 0–4 or -1 for an unavailable reference. For a DDR ungradable case, use grade -1. Use distinct real patient IDs when available. If patient identity is unavailable, supply unique image-level IDs and identify any bootstrap interval as image-level.

```csv
image,patient,path,grade,quality,split
photo001,patient001,photos/photo001.jpg,2,1,external
photo002,patient002,photos/photo002.jpg,-1,-1,external
```

Using the pilot checkpoints as an example:

```bash
python -m dr features --manifest data/external.csv --split external --mode natural --graders runs/pilot/r50-11.pt --damage runs/pilot/damage.pt --quality runs/pilot/quality.pt --device cuda --out runs/pilot/deepdrid.csv
python -m dr evaluate --features runs/pilot/deepdrid.csv --checker runs/pilot/checker.joblib --out runs/pilot/reports-deepdrid
```

External cohort labels must be mapped to the manifest by the researcher. DeepDRiD and DDR support real-image tests. APTOS, IDRiD and Messidor-2 can supply additional external grading tests. Document the third-party reference grades chosen for Messidor-2. FQS supplies quality categories and opinion scores only. Type-level synthetic-to-real rank correlations require independently matched real degradation annotations.

[DeepDRiD](https://github.com/deepdrdoc/DeepDRiD) · [DDR](https://github.com/nkicsl/DDR-dataset) · [APTOS](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) · [IDRiD](https://idrid.grand-challenge.org/) · [Messidor-2](https://www.adcis.net/en/third-party/messidor2/) · [FQS](https://figshare.com/articles/dataset/FIQS_Dataset_Fundus_Image_Quality_Scores_/28129847)

## Research controls and implementation choices

Use `python -m dr --help` and the per-command help for stage-level runs. The runner prints every command it executes and leaves completed checkpoints on disk. After an interruption, invoke the remaining commands individually.

- `features --mode balanced` produces one clean and one seeded corruption per Good image, drawing families and severity levels uniformly in expectation. `balanced-real` additionally includes naturally Usable/Reject originals. `suite` produces all 50 single-operator variants per Good image. `mixture` produces one combination of two or three distinct families per Good image. Transformations and parameters are in `dr/images.py`. The pixel intensity mask is an implementation choice.
- For a leave-one-family-out experiment, pass the same `--exclude FAMILY` during **both fit and policy feature extraction** with `--mode balanced`, refit the checkers, then evaluate on the full suite. For the stricter experiment also exclude that family when training the degradation estimator. `train --robust` includes synthetic training views for the grading ablation. `fit --mlp` adds a two-hidden-layer classifier.
- The exploratory retake rule chooses the strongest estimated family and requires presence probability of at least 0.5. For acquisition defects, it selects the largest binary disease-probability cutoff on deferred policy patients satisfying a referable share cap of 5% with at least 20 labelled images. If none qualifies, all deferred cases go to review. The 5%, 20-image and presence cutoffs are implementation choices, not validated clinical targets or paper findings.
- An official study run needs audited datasets and real labels. This package implements the core pipeline and main comparisons. The optional embedding ablation, manually aligned synthetic-to-real rank analysis, clinician outcomes, profiling on the user's GPU, and the final research tables require further experiment work. Only load checker artifacts you produced or trust, since joblib uses pickle.

## Verification performed

The public commands passed an end-to-end CPU smoke run on generated images using Python 3.12, PyTorch 2.5.1 and Torchvision 0.20.1. The run exercised real ResNet-50/18 training from random initialisation, temperature calibration, feature extraction, fitting both checkers, paired evaluation and single-image routing. The unit tests cover all 50 corruption settings, patient isolation, error denominators, unknown grades and AUGRC ties. Both transformer backbones also completed a 384 px CPU forward pass with random weights. The official RETFound checkpoint, real retinal cohorts, five-seed training and the target RTX workstation were not available for verification.
