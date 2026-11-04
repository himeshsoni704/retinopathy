import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import Photos, make_eyeq
from .evaluation import evaluate
from .features import D, PROBS,Q,U, collect, extract
from .images import FAMILIES
from .policy import fit, route
from .training import train


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return n


def shared(p):
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--size",type=positive,default=384)
    p.add_argument("--device",default="auto",help="auto, cpu, cuda or cuda:0")
    p.add_argument("--batch",type=positive,default=8)
    p.add_argument("--workers",type=int,default=4)


def sources(p):
    shared(p)
    p.add_argument("--graders",nargs="+",required=True)
    p.add_argument("--quality",required=True)
    p.add_argument("--damage",required=True)


def predict(args):
    image = Path(args.image).resolve()
    row = pd.DataFrame([dict(path=str(image),image=image.name,patient="inference",grade=-1,quality=-1,split="external")])
    frame,meta = collect(Photos(row,args.size,args.seed),args)
    bundle = joblib.load(args.checker)
    if bundle["meta"] != meta:
        raise ValueError("Checkpoint identity differs from checker training")
    r,accepted,routing = route(frame,bundle["targets"]["grade"],args.coverage)
    referral_risk = bundle["targets"]["referral"]["models"]["P6"].predict_proba(frame[U+D+Q])[0,1]
    print(json.dumps(dict(image=image.name,provisional_grade=int(frame.prediction.iloc[0]),
        grade_probabilities=frame[PROBS].iloc[0].tolist(),referable_probability=float(frame.ref_probability.iloc[0]),
        uncertainty=frame[U].iloc[0].to_dict(),degradation=frame[D].iloc[0].to_dict(),
        quality=frame[Q].iloc[0].to_dict(),estimated_grade_error=float(r[0]),
        estimated_binary_error=float(referral_risk),route=routing[0],
        reported_grade=int(frame.prediction.iloc[0]) if accepted[0] else None),indent=2))


def main():
    parser = argparse.ArgumentParser(prog="python -m dr")
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("prepare-eyeq")
    p.add_argument("--labels",nargs="+",required=True)
    p.add_argument("--images",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=42)
    p = sub.add_parser("train")
    shared(p)
    p.add_argument("--manifest",required=True)
    p.add_argument("--task",choices=["grade","quality","damage"],required=True)
    p.add_argument("--arch",choices=["r50","vit","retfound"],default="r50")
    p.add_argument("--retinal-weights")
    p.add_argument("--epochs",type=positive,default=50)
    p.add_argument("--accumulate",type=positive,default=1)
    p.add_argument("--robust",action="store_true",help="Train the grader on additional synthetic views")
    p.add_argument("--exclude",choices=FAMILIES)
    p.add_argument("--scratch",action="store_true",help="No pretrained downloads; for smoke testing")
    p.add_argument("--out",required=True)
    p = sub.add_parser("features")
    sources(p)
    p.add_argument("--manifest",required=True)
    p.add_argument("--split",choices=["fit","policy","test","external"],required=True)
    p.add_argument("--mode",choices=["natural","balanced","balanced-real","suite","mixture"])
    p.add_argument("--exclude",choices=FAMILIES)
    p.add_argument("--out",required=True)
    p = sub.add_parser("fit")
    p.add_argument("--fit",required=True)
    p.add_argument("--policy",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--mlp",action="store_true")
    p.add_argument("--min-retakes",type=positive,default=20)
    p.add_argument("--max-referable-share",type=float,default=.05)
    p = sub.add_parser("evaluate")
    p.add_argument("--features",required=True)
    p.add_argument("--checker",required=True)
    p.add_argument("--bootstrap",type=int,default=0,help="Paired patient resamples; 2000 for study")
    p.add_argument("--out",required=True)
    p = sub.add_parser("predict")
    sources(p)
    p.add_argument("--image",required=True)
    p.add_argument("--checker",required=True)
    p.add_argument("--coverage",type=float,choices=[.8,.9],default=.8)
    args = parser.parse_args()
    if args.command == "prepare-eyeq":
        make_eyeq(args.labels,args.images,args.out,args.seed)
    else:
        {"train":train,"features":extract,"fit":fit,"evaluate":evaluate,"predict":predict}[args.command](args)


if __name__ == "__main__":
    main()
