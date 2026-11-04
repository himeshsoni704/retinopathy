"""Extract all signals with one network resident on the accelerator at a time."""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import Photos, read_manifest
from .images import FAMILIES
from .networks import restore
from .training import device_for, digest, loader

U = ["unc_max","unc_entropy","unc_mi","unc_tta"]
D = ["d_"+f for f in FAMILIES]
Q = ["q_good","q_usable","q_reject"]
PROBS = [f"p{k}" for k in range(5)]


def entropy(p):
    return -(p*np.log(np.maximum(p,1e-12))).sum(axis=-1)


def collect(data, args):
    device = device_for(args.device)
    batches = loader(data,args)
    checkpoint_paths = [*args.graders, args.damage, args.quality]
    if len(set(map(lambda p:Path(p).resolve(),checkpoint_paths))) != len(checkpoint_paths):
        raise ValueError("Duplicate checkpoint paths")
    meta = {"graders":sorted(digest(p) for p in args.graders),"damage":digest(args.damage),
            "quality":digest(args.quality),"size":args.size}
    n = len(data)
    mean = np.zeros((n,5))
    view_mean = np.zeros((n,5))
    member_entropy = np.zeros(n)
    refer = np.zeros(n)
    q = np.empty((n,3))
    d = np.empty((n,8))
    kinds = np.empty((n,8))
    rows, seeds, arch, run_id = [], set(), None, None
    for position,path in enumerate(checkpoint_paths):
        net, ck = restore(path,device)
        expected = "grade" if position < len(args.graders) else ("quality" if position==len(checkpoint_paths)-1 else "damage")
        if ck["task"] != expected or ck["size"] != args.size:
            raise ValueError(f"Wrong checkpoint task or size: {path}")
        if run_id is not None and ck["run_id"] != run_id:
            raise ValueError("Models were trained on different manifests")
        run_id = ck["run_id"]
        if expected == "grade":
            if ck["seed"] in seeds or (arch and ck["arch"] != arch):
                raise ValueError("Use distinct seeds from a single architecture")
            seeds.add(ck["seed"])
            arch = ck["arch"]
        start = 0
        with torch.inference_mode():
            for batch in batches:
                x = batch["x"].to(device)
                sl = slice(start,start+len(x))
                if position == 0:
                    rows.extend(zip(batch["image"],batch["patient"],batch["grade"].tolist(),
                                    batch["quality"].tolist(),batch["split"],batch["family"],batch["variant"]))
                a,b = net(x)
                if expected == "grade":
                    tg,tr = ck["temperatures"]
                    p = (a.float()/tg).softmax(-1).cpu().numpy()
                    mean[sl] += p
                    member_entropy[sl] += entropy(p)
                    refer[sl] += (b.float()/tr).sigmoid().flatten().cpu().numpy()
                    view_mean[sl] += p
                    for dims in ([-1],[-2],[-2,-1]):
                        logits,_ = net(x.flip(dims))
                        view_mean[sl] += (logits.float()/tg).softmax(-1).cpu().numpy()
                elif expected == "quality":
                    q[sl] = a.softmax(-1).cpu().numpy()
                else:
                    kinds[sl] = a.sigmoid().cpu().numpy()
                    d[sl] = kinds[sl]*b.sigmoid().cpu().numpy()
                start = sl.stop
        del net
        gc.collect()
    m = len(seeds)
    mean /= m
    table = pd.DataFrame(rows,columns=["image","patient","grade","quality","split","family","variant"])
    table[PROBS] = mean
    table["prediction"] = mean.argmax(axis=1)
    table["ref_probability"] = refer/m
    table[U] = np.column_stack([1-mean.max(axis=1),entropy(mean),
                               np.maximum(0,entropy(mean)-member_entropy/m),entropy(view_mean/(4*m))])
    table[D], table[Q] = d,q
    table[["present_"+f for f in FAMILIES]] = kinds
    meta.update(arch=arch,training_manifest=run_id)
    return table,meta


def extract(args):
    frame = read_manifest(args.manifest)
    selected = frame[frame.split == args.split]
    mode = args.mode or ("balanced" if args.split in ("fit","policy") else "natural")
    data = Photos(selected,args.size,args.seed,mode,exclude=args.exclude)
    table,meta = collect(data,args)
    if args.split != "external" and meta["training_manifest"] != digest(args.manifest):
        raise ValueError("Development features require the original training manifest")
    path = Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(path,index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(meta,indent=2))
    path.with_suffix(".sampling.json").write_text(json.dumps(
        dict(manifest=digest(args.manifest), split=args.split, mode=mode, seed=args.seed,
             excluded_family=args.exclude, patients=int(table.patient.nunique())), indent=2))
    print(f"{len(table)} rows saved to {path}")
