import hashlib
import math
import random
import time
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import cohen_kappa_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import Photos, read_manifest
from .networks import build
from .runtime import precision, autocast, environment, Profile, protect_all


def device_for(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def loader(data, args, shuffle=False):
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    return DataLoader(data, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=device_for(args.device).type == "cuda")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def temperature(logits, targets, binary=False):
    def nll(log_t):
        scaled = logits / math.exp(log_t)
        if binary:
            return F.binary_cross_entropy_with_logits(scaled.flatten(), targets.float()).item()
        return F.cross_entropy(scaled, targets).item()
    res = minimize_scalar(nll, bounds=(-4,4), method="bounded")
    return math.exp(res.x)


def data_for(frame, split, args):
    subset = frame[frame.split == split]
    mode = "natural"
    if args.task == "grade":
        subset = subset[(subset.quality == 0) & (subset.grade >= 0)]
        if args.robust and split == "train":
            mode = "balanced"
    elif args.task == "damage":
        mode = "damage"
    else:
        subset = subset[subset.quality >= 0]
    return Photos(subset, args.size, args.seed, mode, augment=(split=="train"), exclude=args.exclude, cache_dir=getattr(args,"cache_dir",None),
                  condition_profile=getattr(args,"condition_profile",None) if split=="train" else None,
                  corruption_size=getattr(args,"corruption_size",None),
                  donor_patients=frame.loc[frame.split=="train","patient"])


def loss_for(task, output, batch, device, class_weights):
    first, second = output
    if task == "quality":
        return F.cross_entropy(first, batch["quality"].to(device, non_blocking=device.type=="cuda"))
    if task == "grade":
        y = batch["grade"].to(device, non_blocking=device.type=="cuda")
        return (F.cross_entropy(first, y, weight=class_weights)
                +F.binary_cross_entropy_with_logits(second.flatten(), (y>=2).float()))
    kinds, strength = batch["present"].to(device, non_blocking=device.type=="cuda"), batch["severity"].to(device, non_blocking=device.type=="cuda")
    severity_loss = F.smooth_l1_loss(second.sigmoid(), strength, reduction="none")
    return F.binary_cross_entropy_with_logits(first,kinds)+(severity_loss*kinds).sum()/kinds.sum().clamp_min(1)


@torch.no_grad()
def validation(net, batches, args, device, class_weights):
    net.eval()
    total = count = 0
    predictions, labels = [], []
    for batch in batches:
        out = net(batch["x"].to(device, non_blocking=device.type=="cuda"))
        total += loss_for(args.task, out, batch, device, class_weights).item()*len(batch["x"])
        count += len(batch["x"])
        if args.task == "grade":
            predictions.extend(out[0].argmax(1).cpu().tolist())
            labels.extend(batch["grade"].tolist())
    if predictions and len(set(labels+predictions)) > 1:
        return cohen_kappa_score(labels,predictions,labels=list(range(5)),weights="quadratic")
    return -total/count


def accumulation_window(step, total, accumulate):
    """Number of microbatches in the current optimizer window."""
    if accumulate < 1 or not 0 <= step < total:
        raise ValueError("Invalid accumulation window")
    return min(accumulate,total-(step//accumulate)*accumulate)


def train(args):
    output=Path(args.out)
    protect_all([output,output.with_suffix(output.suffix+".json")],getattr(args,"overwrite",False))
    if getattr(args,"condition_profile",None) and not (args.task=="damage" or (args.task=="grade" and args.robust)):
        raise ValueError("Conditioning requires damage training or robust grade training")
    if args.task != "grade":
        args.arch = {"quality":"r50","damage":"r18"}[args.task]
    if args.size < 32 or args.size % 16:
        raise ValueError("Use image size >=32 divisible by 16")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = read_manifest(args.manifest)
    train_data = data_for(frame,"train",args)
    val_data = data_for(frame,"select",args)
    device = device_for(args.device)
    net = build(args.task,args.arch,args.size,not args.scratch,args.retinal_weights).to(device)
    params = [{"params": [p for p in net.encoder.parameters() if p.requires_grad], "lr":1e-4},
              {"params": [p for name,p in net.named_parameters() if not name.startswith("encoder.")], "lr":1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    def schedule(e):
        if e < 5:
            return (e+1)/5
        return .5*(1+math.cos(math.pi*(e-5)/max(1,args.epochs-5)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,schedule)
    mode = precision(device, getattr(args,"precision","auto"))
    scaler = torch.amp.GradScaler("cuda", enabled=mode=="fp16")
    profile = Profile(device)
    history = []
    counts = np.bincount(train_data.frame.grade.clip(lower=0).astype(int), minlength=5)
    class_weights = torch.tensor(1/np.maximum(counts,1), dtype=torch.float32,device=device)
    batches = loader(train_data,args,True)
    val_batches = loader(val_data,args)
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    best, stale = -float("inf"), 0
    best_state = None
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_data.epoch = epoch
        net.train()
        opt.zero_grad(set_to_none=True)
        for step,batch in enumerate(batches):
            # Divide final, incomplete accumulation window by its actual length.
            window = accumulation_window(step,len(batches),args.accumulate)
            with autocast(device, mode):
                loss = loss_for(args.task, net(batch["x"].to(device, non_blocking=device.type=="cuda")), batch, device, class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss/window).backward()
            if (step+1)%args.accumulate == 0 or step+1 == len(batches):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        scheduler.step()
        score = validation(net,val_batches,args,device,class_weights)
        history.append(dict(epoch=epoch+1, validation=float(score), seconds=time.perf_counter()-epoch_start,
                            training_samples=len(train_data), microbatches=len(batches)))
        print(f"epoch={epoch+1} validation={score:.5f}",flush=True)
        if score > best:
            best, stale = score, 0
            best_state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= 8:
                break
    if best_state is None:
        raise ValueError("Training did not produce a finite validation score")
    net.load_state_dict(best_state)
    temps = [1.,1.]
    if args.task == "grade":
        logits, binary, labels = [], [], []
        net.eval()
        with torch.no_grad():
            for batch in loader(data_for(frame,"cal",args),args):
                a,b = net(batch["x"].to(device, non_blocking=device.type=="cuda"))
                logits.append(a.cpu())
                binary.append(b.cpu())
                labels.append(batch["grade"])
        y = torch.cat(labels)
        temps = [temperature(torch.cat(logits),y), temperature(torch.cat(binary),(y>=2),True)]
    checkpoint = dict(state=best_state, task=args.task, arch=args.arch, size=args.size,
                      seed=args.seed, temperatures=temps, runtime=environment(device),
                      profile=profile.finish(), precision=mode, history=history,
                      parameter_count=sum(p.numel() for p in net.parameters()),
                      trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
                      condition_profile_sha256=digest(args.condition_profile) if getattr(args,"condition_profile",None) else None,
                      accumulation_reduction="mean of microbatch losses; actual tail window", selection_score=float(best),
                      run_id=digest(args.manifest), params=vars(args).copy())
    tmp = out.with_suffix(out.suffix+".tmp")
    torch.save(checkpoint,tmp)
    tmp.replace(out)
    summary = {k:v for k,v in checkpoint.items() if k != "state"}
    summary["checkpoint_bytes"] = out.stat().st_size
    out.with_suffix(out.suffix+".json").write_text(json.dumps(summary,indent=2))

