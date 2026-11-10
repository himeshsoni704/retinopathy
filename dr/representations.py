"""Offline frozen representations; never downloads weights.

I-JEPA requires a local official ViT-H/14 224 checkpoint (CC BY-NC 4.0).
DINOv2 requires a local official ViT-S/14 non-register backbone (Apache-2.0).
Outputs are research measurements, not lesion-preservation guarantees.
"""
import argparse
import json
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import Photos, read_manifest
from .networks import restore
from .training import device_for, digest, loader
from .runtime import protect_all, precision, autocast, Profile, environment


def ijepa_encoder(size=224, width=1280, depth=32, heads=16, patch=14):
    from timm.models.vision_transformer import VisionTransformer
    return VisionTransformer(img_size=size,patch_size=patch,embed_dim=width,depth=depth,
        num_heads=heads,mlp_ratio=4,qkv_bias=True,class_token=False,global_pool='avg',
        fc_norm=False,num_classes=0,norm_layer=partial(nn.LayerNorm,eps=1e-6))


def load_encoder(kind,checkpoint,size,device):
    if kind=='grader':
        net,meta=restore(checkpoint,device)
        if meta['task']!='grade' or meta['size']!=size:
            raise ValueError('Grader task/input size mismatch')
        encoder=net.encoder
    else:
        if size!=224:
            raise ValueError('Use 224 for these pretrained representation protocols')
        raw=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if kind=='ijepa':
            if 'target_encoder' not in raw:
                raise ValueError('Expected official I-JEPA target_encoder checkpoint')
            state={k.removeprefix('module.'):v for k,v in raw['target_encoder'].items()}
            encoder=ijepa_encoder()
        else:
            import timm
            encoder=timm.create_model('vit_small_patch14_dinov2',pretrained=False,img_size=518,dynamic_img_size=True,num_classes=0)
            state=raw
            # Official DINOv2 exposes a training-only mask token; timm's inference model does not.
            state={k:v for k,v in state.items() if k!='mask_token'}
        encoder.load_state_dict(state,strict=True)
    return encoder.eval().requires_grad_(False).to(device)


def paired_distances(rows, embeddings):
    if len(rows)!=len(embeddings): raise ValueError('Embedding/row count mismatch')
    clean={str(r.image):i for i,r in rows.iterrows() if r.variant=='original' and r.quality==0}
    result=[]
    for i,r in rows.iterrows():
        if r.variant=='original' or str(r.image) not in clean: continue
        a,b=embeddings[clean[str(r.image)]],embeddings[i]
        denom=np.linalg.norm(a)*np.linalg.norm(b)
        result.append(dict(image=r.image,patient=r.patient,variant=r.variant,split=r.split,
            cosine_distance=float(1-np.dot(a,b)/denom) if denom>0 else None,
            l2_distance=float(np.linalg.norm(a-b))))
    return pd.DataFrame(result,columns=['image','patient','variant','split','cosine_distance','l2_distance'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--encoder',choices=['grader','ijepa','dinov2'],default='grader')
    p.add_argument('--split',choices=['train','select','cal','fit','policy','test','external'],required=True)
    p.add_argument('--mode',choices=['natural','balanced','balanced-real','suite'],default='balanced-real')
    p.add_argument('--size',type=int,default=224); p.add_argument('--batch',type=int,default=1)
    p.add_argument('--workers',type=int,default=0); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu'); p.add_argument('--precision',choices=['auto','fp32','fp16','bf16'],default='auto')
    p.add_argument('--cache-dir'); p.add_argument('--corruption-size',type=int)
    p.add_argument('--out',required=True); p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    output=Path(a.out)
    protect_all([output,*[output.with_suffix(s) for s in (".rows.csv",".pairs.csv",".json")]],a.overwrite)
    if a.batch<1: p.error('batch must be positive')
    frame=read_manifest(a.manifest)
    data=Photos(frame[frame.split==a.split],a.size,a.seed,a.mode,cache_dir=a.cache_dir,corruption_size=a.corruption_size)
    device=device_for(a.device); mode=precision(device,a.precision,training=False)
    profile=Profile(device)
    encoder=load_encoder(a.encoder,a.checkpoint,a.size,device)
    vectors=[]; rows=[]
    with torch.inference_mode(),autocast(device,mode):
        for batch in loader(data,a):
            z=encoder(batch['x'].to(device,non_blocking=device.type=='cuda'))
            if z.ndim!=2: raise ValueError('Encoder must return pooled N x D embeddings')
            vectors.append(z.float().cpu().numpy())
            rows.extend(zip(batch['image'],batch['patient'],batch['variant'],batch['split'],batch['quality'].tolist()))
    vectors=np.concatenate(vectors)
    rows=pd.DataFrame(rows,columns=['image','patient','variant','split','quality'])
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    # File handle avoids numpy silently appending an extension and defeating overwrite protection.
    with out.open('wb') as stream: np.savez_compressed(stream,embeddings=vectors)
    rows.to_csv(out.with_suffix('.rows.csv'),index=False)
    paired_distances(rows,vectors).to_csv(out.with_suffix('.pairs.csv'),index=False)
    meta=dict(args=vars(a),checkpoint_sha256=digest(a.checkpoint),manifest_sha256=digest(a.manifest),
              precision=mode,profile=profile.finish(),runtime=environment(device),
              embedding_sha256=digest(out),rows_sha256=digest(out.with_suffix('.rows.csv')),
              warning='Paired distances require a clean reference; not deployable on unpaired natural images')
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__': main()
