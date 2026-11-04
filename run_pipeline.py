"""Run one complete architecture experiment on one GPU, keeping each stage on disk."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",required=True,help="New experiment folder")
    p.add_argument("--arch",choices=["r50","vit","retfound"],default="r50")
    p.add_argument("--retinal-weights")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,22,33,44,55])
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--accumulate",type=int,default=2)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--size",type=int,default=384)
    p.add_argument("--device",default="auto")
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
    train = ["train","--manifest",args.manifest,*runtime,"--accumulate",args.accumulate,"--epochs",args.epochs]
    if args.scratch:
        train.append("--scratch")
    for task in ("quality","damage"):
        run(*train,"--task",task,"--out",out/f"{task}.pt")
    graders = []
    for seed in args.seeds:
        path = out/f"{args.arch}-{seed}.pt"
        extra = ["--retinal-weights",args.retinal_weights] if args.arch=="retfound" and args.retinal_weights else []
        run(*train,"--task","grade","--arch",args.arch,"--seed",seed,"--out",path,*extra)
        graders.append(path)
    for split in ("fit","policy","test"):
        mode = args.test_mode if split=="test" else "balanced"
        run("features","--split",split,"--mode",mode,"--manifest",args.manifest,*runtime,
            "--graders",*graders,"--quality",out/"quality.pt","--damage",out/"damage.pt",
            "--out",out/f"{split}.csv")
    run("fit","--fit",out/"fit.csv","--policy",out/"policy.csv","--out",out/"checker.joblib")
    run("evaluate","--features",out/"test.csv","--checker",out/"checker.joblib",
        "--out",out/"reports","--bootstrap",args.bootstrap)


if __name__=="__main__":
    main()
