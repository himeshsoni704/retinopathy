"""Opt-in compute benchmark. Synthetic tensors measure systems behavior, not DR accuracy."""
import argparse
import gc
import json
from pathlib import Path
import torch
from .networks import build
from .training import device_for
from .runtime import precision, autocast, environment, Profile, protect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arch',choices=['r50','r18','mobilenetv3','efficientnetb0'],default='r50')
    p.add_argument('--size',type=int,default=320); p.add_argument('--batch',type=int,default=2)
    p.add_argument('--accumulate',type=int,default=4); p.add_argument('--steps',type=int,default=20)
    p.add_argument('--warmup',type=int,default=3); p.add_argument('--seed',type=int,default=11)
    p.add_argument('--device',default='cuda'); p.add_argument('--precisions',nargs='+',default=['fp32','fp16'],choices=['fp32','fp16','bf16'])
    p.add_argument('--out',required=True); p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(); protect(a.out,a.overwrite)
    if min(a.batch,a.accumulate,a.steps,a.warmup)<1 or a.size<32: p.error('Invalid benchmark dimensions')
    device=device_for(a.device); results=[]
    for requested in a.precisions:
        mode=precision(device,requested)
        torch.manual_seed(a.seed)
        net=build('grade',a.arch,a.size).to(device).train()
        optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4)
        scaler=torch.amp.GradScaler('cuda',enabled=mode=='fp16')
        x=torch.randn(a.batch,3,a.size,a.size,device=device)
        y=torch.arange(a.batch,device=device)%5
        def update():
            optimizer.zero_grad(set_to_none=True)
            for _ in range(a.accumulate):
                with autocast(device,mode):
                    grade,refer=net(x)
                    loss=torch.nn.functional.cross_entropy(grade,y)+torch.nn.functional.binary_cross_entropy_with_logits(refer.flatten(),(y>=2).float())
                scaler.scale(loss/a.accumulate).backward()
            scaler.step(optimizer); scaler.update()
        try:
            for _ in range(a.warmup): update()
            profile=Profile(device)
            for _ in range(a.steps): update()
            record=profile.finish()
            record.update(precision=mode,status='ok',training_images_per_second=a.steps*a.batch*a.accumulate/record['seconds'],
                          parameters=sum(p.numel() for p in net.parameters()))
            optimizer.zero_grad(set_to_none=True)
            net.eval()
            with torch.inference_mode(),autocast(device,mode):
                for _ in range(a.warmup): net(x)
                timer=Profile(device)
                for _ in range(a.steps): net(x)
                record['inference']=timer.finish()
                record['inference_ms_per_image']=1000*record['inference']['seconds']/(a.steps*a.batch)
            results.append(record)
        except torch.OutOfMemoryError:
            results.append(dict(precision=mode,status='out_of_memory'))
        finally:
            del net,optimizer,scaler,x,y
            gc.collect()
            if device.type=='cuda': torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(config=vars(a),runtime=environment(device),results=results,
        excludes='Image loading, preprocessing, validation, calibration and checkpoint I/O; random weights/tensors'),indent=2))

if __name__=='__main__': main()
