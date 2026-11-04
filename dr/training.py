import hashlib
import math
import random
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import cohen_kappa_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import Photos, read_manifest
from .networks import build


def device_for(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def loader(data, args, shuffle=False):
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
    return Photos(subset, args.size, args.seed, mode, augment=(split=="train"), exclude=args.exclude)


def loss_for(task, output, batch, device, class_weights):
    first, second = output
    if task == "quality":
        return F.cross_entropy(first, batch["quality"].to(device))
    if task == "grade":
        y = batch["grade"].to(device)
        return (F.cross_entropy(first, y, weight=class_weights)
                +F.binary_cross_entropy_with_logits(second.flatten(), (y>=2).float()))
    kinds, strength = batch["present"].to(device), batch["severity"].to(device)
    severity_loss = F.smooth_l1_loss(second.sigmoid(), strength, reduction="none")
    return F.binary_cross_entropy_with_logits(first,kinds)+(severity_loss*kinds).sum()/kinds.sum().clamp_min(1)


@torch.no_grad()
def validation(net, batches, args, device, class_weights):
    net.eval()
    total = count = 0
    predictions, labels = [], []
    for batch in batches:
        out = net(batch["x"].to(device))
        total += loss_for(args.task, out, batch, device, class_weights).item()*len(batch["x"])
        count += len(batch["x"])
        if args.task == "grade":
            predictions.extend(out[0].argmax(1).cpu().tolist())
            labels.extend(batch["grade"].tolist())
    if predictions and len(set(labels+predictions)) > 1:
        return cohen_kappa_score(labels,predictions,labels=list(range(5)),weights="quadratic")
    return -total/count


def train(args):
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
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    counts = np.bincount(train_data.frame.grade.clip(lower=0).astype(int), minlength=5)
    class_weights = torch.tensor(1/np.maximum(counts,1), dtype=torch.float32,device=device)
    batches = loader(train_data,args,True)
    val_batches = loader(val_data,args)
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    best, stale = -float("inf"), 0
    best_state = None
    for epoch in range(args.epochs):
        train_data.epoch = epoch
        net.train()
        opt.zero_grad(set_to_none=True)
        for step,batch in enumerate(batches):
            # Divide final, incomplete accumulation window by its actual length.
            window = min(args.accumulate,len(batches)-(step//args.accumulate)*args.accumulate)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_for(args.task, net(batch["x"].to(device)), batch, device, class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss/window).backward()
            if (step+1)%args.accumulate == 0 or step+1 == len(batches):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        scheduler.step()
        score = validation(net,val_batches,args,device,class_weights)
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
                a,b = net(batch["x"].to(device))
                logits.append(a.cpu())
                binary.append(b.cpu())
                labels.append(batch["grade"])
        y = torch.cat(labels)
        temps = [temperature(torch.cat(logits),y), temperature(torch.cat(binary),(y>=2),True)]
    checkpoint = dict(state=best_state, task=args.task, arch=args.arch, size=args.size,
                      seed=args.seed, temperatures=temps, selection_score=float(best),
                      run_id=digest(args.manifest), params=vars(args).copy())
    tmp = out.with_suffix(out.suffix+".tmp")
    torch.save(checkpoint,tmp)
    tmp.replace(out)
