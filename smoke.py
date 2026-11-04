"""Exercise public commands using generated pictures and a real CPU training step."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,help="Empty folder for disposable smoke artifacts")
    args = p.parse_args()
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()):
        p.error("Use a fresh folder")
    images = root/"images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(7)
    yy,xx = np.mgrid[:48,:48]
    mask = (yy-24)**2+(xx-24)**2 < 22**2
    labels = []
    for patient in range(160):
        for eye in ("left","right"):
            grade = patient%5
            quality = 0 if patient%4 < 2 else patient%4-1
            arr = rng.integers(30,160,size=(48,48,3),dtype=np.uint8)
            arr[~mask] = 0
            name = f"{patient}_{eye}.jpeg"
            Image.fromarray(arr).save(images/name)
            labels.append((name,grade,quality))
    csv = root/"labels.csv"
    pd.DataFrame(labels,columns=["image","DR_grade","quality"]).to_csv(csv,index=False)
    env = {**os.environ,"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"}
    def run(args, capture=False):
        return subprocess.run([sys.executable,*map(str,args)],check=True,env=env,
                              capture_output=capture,text=True)
    manifest = root/"manifest.csv"
    run(["-m","dr","prepare-eyeq","--labels",csv,"--images",images,"--out",manifest])
    splits = pd.read_csv(manifest)
    assert splits.groupby("patient").split.nunique().max()==1
    run(["run_pipeline.py","--manifest",manifest,"--out",root/"run","--scratch","--size",32,
         "--seeds",11,"--epochs",1,"--batch",16,"--accumulate",1,"--workers",0,
         "--device","cpu","--test-mode","natural","--bootstrap",4])
    folder = root/"run"
    outcome = run(["-m","dr","predict","--image",images/"0_left.jpeg","--graders",folder/"r50-11.pt",
                   "--quality",folder/"quality.pt","--damage",folder/"damage.pt",
                   "--checker",folder/"checker.joblib","--coverage",.8,"--size",32,"--workers",0,
                   "--device","cpu"],capture=True)
    result = json.loads(outcome.stdout)
    assert result["route"] in {"accept","review","retake"}
    assert abs(sum(result["grade_probabilities"])-1)<1e-5
    assert 0 <= result["estimated_grade_error"] <= 1
    assert (result["reported_grade"] is not None)==(result["route"]=="accept")
    report = pd.read_csv(folder/"reports"/"policies.csv")
    rows = report[(report.group=="all") & (report.policy=="P6") & (report["mode"]=="rank")]
    assert len(rows)==4 and rows.augrc.notna().all()
    print("PASS: patient split, training, calibration, features, fit, evaluation and single-image prediction")


if __name__ == "__main__":
    main()
