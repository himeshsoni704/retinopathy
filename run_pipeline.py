"""Run one complete architecture experiment on one GPU, keeping each stage on disk."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",required=True,help="New experiment folder")
    p.add_argument("--arch",choices=["r50","r18","mobilenetv3","efficientnetb0","vit","retfound"],default="r50")
    p.add_argument("--retinal-weights")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,22,33,44,55])
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--accumulate",type=int,default=2)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--size",type=int,default=384)
    p.add_argument("--device",default="auto")
    p.add_argument("--precision",choices=["auto","fp32","fp16","bf16"],default="auto")
    p.add_argument("--inference-precision",choices=["auto","fp32","fp16","bf16"],default="auto")
    p.add_argument("--cache-dir")
    p.add_argument("--robust",action="store_true")
    p.add_argument("--exclude")
    p.add_argument("--condition-profile")
    p.add_argument("--corruption-size",type=int)
    p.add_argument("--fit-mode",choices=["balanced","balanced-real"],default="balanced")
    p.add_argument("--bootstrap",type=int,default=0)
    p.add_argument("--test-mode",choices=["suite","natural","mixture"],default="suite")
    p.add_argument("--scratch",action="store_true",help="Random initial weights for smoke test")
    args = p.parse_args()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        p.error("Use an empty experiment folder; individual CLI stages can be rerun separately.")
    if args.arch=="retfound" and not args.scratch and not args.retinal_weights:
        p.error("Pass --retinal-weights for RETFound-MAE")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("Each grading seed must be distinct")
    out.mkdir(parents=True,exist_ok=True)
    def run(*items):
        cmd = [sys.executable,"-m","dr",*map(str,items)]
        print(subprocess.list2cmdline(cmd),flush=True)
        subprocess.run(cmd,check=True)
    runtime = ["--batch",args.batch,"--workers",args.workers,"--size",args.size,"--device",args.device]
    if args.cache_dir:
        runtime += ["--cache-dir",args.cache_dir]
    train = ["train", "--precision",args.precision,"--manifest",args.manifest,*runtime,"--accumulate",args.accumulate,"--epochs",args.epochs]
    if args.corruption_size:
        train += ["--corruption-size",args.corruption_size]
    if args.exclude:
        train += ["--exclude",args.exclude]
    if args.scratch:
        train.append("--scratch")
    for task in ("quality","damage"):
        run(*train,*(["--condition-profile",args.condition_profile] if task=="damage" and args.condition_profile else []),"--task",task,"--out",out/f"{task}.pt")
    graders = []
    for seed in args.seeds:
        path = out/f"{args.arch}-{seed}.pt"
        extra = ["--retinal-weights",args.retinal_weights] if args.arch=="retfound" and args.retinal_weights else []
        if args.robust:
            extra += ["--robust"]
            if args.condition_profile:
                extra += ["--condition-profile",args.condition_profile]
        run(*train,"--task","grade","--arch",args.arch,"--seed",seed,"--out",path,*extra)
        graders.append(path)
    for split in ("fit","policy","test"):
        mode = args.test_mode if split=="test" else args.fit_mode
        extra = ["--precision",args.inference_precision]
        if args.corruption_size:
            extra += ["--corruption-size",args.corruption_size]
        if split != "test":
            if args.exclude:
                extra += ["--exclude",args.exclude]
            if args.condition_profile:
                extra += ["--condition-profile",args.condition_profile]
        run("features",*extra,"--split",split,"--mode",mode,"--manifest",args.manifest,*runtime,
            "--graders",*graders,"--quality",out/"quality.pt","--damage",out/"damage.pt",
            "--out",out/f"{split}.csv")
    run("fit","--fit",out/"fit.csv","--policy",out/"policy.csv","--out",out/"checker.joblib")
    run("evaluate","--features",out/"test.csv","--checker",out/"checker.joblib",
        "--out",out/"reports","--bootstrap",args.bootstrap)


if __name__=="__main__":
    main()

