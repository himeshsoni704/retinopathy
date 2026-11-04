"""Manifest preparation and deterministic, on-demand variants."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from .images import FAMILIES, FAMILY_OF, degrade, prepare, sample_spec, seed_for

SPLITS = ("train", "select", "cal", "fit", "policy", "test")
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]


def read_manifest(path):
    path = Path(path).resolve()
    frame = pd.read_csv(path, dtype={"patient": str, "image": str})
    required = {"image","patient","path","grade","quality","split"}
    if required - set(frame):
        raise ValueError(f"Missing columns: {required-set(frame)}")
    if frame[list(required)].isna().any().any() or frame.image.duplicated().any():
        raise ValueError("Missing required values or duplicate image IDs (use -1 for unknown labels)")
    if frame.groupby("patient").split.nunique().max() > 1:
        raise ValueError("Patient crosses partitions")
    frame["path"] = [str((path.parent / p).resolve()) for p in frame.path]
    if frame.path.duplicated().any() or not set(frame.split) <= set(SPLITS)|{"external"}:
        raise ValueError("Duplicate image paths or unknown split names")
    if not set(frame.grade) <= set(range(-1,5)) or not set(frame.quality) <= set(range(-1,3)):
        raise ValueError("Use grade 0..4, quality 0..2, or -1 for unknown")
    return frame


def make_eyeq(csvs, root, output, seed):
    labels = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    root = Path(root)
    wanted = set(labels.image)
    paths = {}
    for path in root.rglob("*"):
        if path.name in wanted and path.is_file():
            if path.name in paths:
                raise ValueError(f"Multiple files named {path.name}")
            paths[path.name] = str(path.resolve())
    missing = wanted - paths.keys()
    if missing:
        raise ValueError(f"{len(missing)} EyeQ images missing, e.g. {sorted(missing)[:3]}")
    frame = labels.rename(columns={"DR_grade":"grade"})[["image","grade","quality"]].copy()
    frame["patient"] = frame.image.str.extract(r"^(.+)_(?:left|right)\.[^.]+$", flags=re.I)[0]
    if frame.patient.isna().any() or frame.image.duplicated().any():
        raise ValueError("Duplicate IDs or unrecognised EyePACS left/right filenames")
    if not set(frame.grade) <= set(range(5)) or not set(frame.quality) <= {0,1,2}:
        raise ValueError("Unexpected EyeQ grade or quality encoding")
    frame["path"] = frame.image.map(paths)
    frame["split"] = ""
    strata = frame.grade.astype(str)+"_"+frame.quality.astype(str)
    folds = StratifiedGroupKFold(20, shuffle=True, random_state=seed)
    allocation = ["train"]*12+["select","cal"]+["fit"]*2+["policy"]*2+["test"]*2
    for name, (_, indices) in zip(allocation, folds.split(frame, strata, frame.patient), strict=True):
        frame.loc[indices,"split"] = name
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.groupby("split")[["image","patient"]].nunique())


class Photos(Dataset):
    def __init__(self, frame, size, seed, mode="natural", augment=False, exclude=None):
        # natural: original rows. balanced: one clean and one synthetic copy per Good source.
        # suite: original rows plus all 50 variants of Good sources. mixture: one mixed copy per Good source.
        if mode in ("balanced","damage","mixture"):
            frame = frame[frame.quality == 0]
        self.frame = frame.reset_index(drop=True)
        self.size, self.seed, self.mode = size, seed, mode
        self.augment, self.exclude, self.epoch = augment, exclude, 0
        self.items = []
        for idx, quality in enumerate(self.frame.quality):
            if mode != "mixture":
                self.items.append((idx,"clean",0))
            if mode in ("balanced","mixture") or (mode == "balanced-real" and quality == 0):
                variant = "balanced" if mode == "balanced-real" else mode
                self.items.append((idx, variant, 0))
            if mode == "suite" and quality == 0:
                self.items.extend((idx,op,level) for op in FAMILY_OF for level in range(1,6))
        if not self.items:
            raise ValueError(f"No eligible images for {mode}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row_id, op, level = self.items[index]
        row = self.frame.iloc[row_id]
        rng = np.random.default_rng(seed_for(self.seed, self.epoch, row.image, op, level))
        image = prepare(row.path, self.size)
        if self.mode == "damage":
            roll = rng.random()
            spec = [] if roll < .2 else sample_spec(rng, roll > .65, self.exclude)
        elif op in ("balanced","mixture"):
            spec = sample_spec(rng, op == "mixture", self.exclude)
        elif op in FAMILY_OF:
            spec = [(op,level)]
        else:
            spec = []
        present = torch.zeros(8)
        severity = torch.zeros(8)
        for operator, strength in spec:
            image = degrade(image, operator, strength, rng)
            k = FAMILIES.index(FAMILY_OF[operator])
            present[k], severity[k] = 1., strength/5
        if self.augment:
            if rng.random() < .5:
                image = TF.hflip(image)
            image = TF.rotate(image, int(rng.choice([0,90,180,270])))
        return {"x": TF.normalize(TF.to_tensor(image), MEAN, STD),
                "grade": int(row.grade), "quality": int(row.quality),
                "image": str(row.image), "patient": str(row.patient),
                "family": "+".join(FAMILY_OF[o] for o,s in spec) or "original",
                "variant": "+".join(f"{o}-{s}" for o,s in spec) or "original",
                "split": row.split, "present": present, "severity": severity}
