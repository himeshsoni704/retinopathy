"""Replay explicit corruption specifications sampled from training-only donors.

Natural expert labels and synthetic-estimator predictions remain distinct provenance.
This is a parameterized simulator, not a differentiable or clinically validated generator.
"""
import json
from pathlib import Path
from .images import FAMILY_OF


def load_profile(path, allowed_patients):
    profile = json.loads(Path(path).read_text())
    if profile.get('version') != 1 or not profile.get('donors'):
        raise ValueError('Expected version 1 and nonempty donors')
    for donor in profile['donors']:
        if str(donor.get('patient')) not in allowed_patients or donor.get('split') != 'train':
            raise ValueError('Conditioning donors must belong to training patients')
        if donor.get('provenance') not in ('natural_manual','natural_estimated'):
            raise ValueError('Explicit natural annotation/estimate provenance required')
        spec = donor.get('spec', [])
        if not 1 <= len(spec) <= 3:
            raise ValueError('Each donor needs one to three operations')
        families = []
        for item in spec:
            op, level = item.get('operator'), item.get('level')
            if op not in FAMILY_OF or type(level) is not int or level not in range(1,6):
                raise ValueError('Invalid operator or discrete severity')
            families.append(FAMILY_OF[op])
        if len(set(families)) != len(families):
            raise ValueError('Duplicate families in donor specification')
    return profile


def sample_condition(profile, rng, exclude=None):
    specs = [[(i['operator'],i['level']) for i in d['spec']] for d in profile['donors']]
    specs = [s for s in specs if all(FAMILY_OF[o] != exclude for o,_ in s)]
    if not specs:
        raise ValueError('No eligible donors after family exclusion')
    return specs[int(rng.integers(len(specs)))]


def main():
    """Estimate a proxy donor profile from natural poor-quality training photographs."""
    import argparse
    import numpy as np
    import torch
    from .data import Photos, read_manifest
    from .images import FAMILIES, OPERATORS
    from .networks import restore
    from .training import device_for, digest, loader
    from .runtime import protect
    p=argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('--manifest',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True); p.add_argument('--overwrite',action='store_true')
    p.add_argument('--device',default='cpu'); p.add_argument('--batch',type=int,default=1)
    p.add_argument('--workers',type=int,default=0); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cache-dir'); p.add_argument('--presence-threshold',type=float,default=.5)
    p.add_argument('--exposure-operator',choices=['dark','bright'],required=True)
    p.add_argument('--noise-operator',choices=['noise','jpeg'],required=True)
    a=p.parse_args(); protect(a.out,a.overwrite)
    if not 0<=a.presence_threshold<=1 or a.batch<1: p.error('Invalid threshold or batch')
    frame=read_manifest(a.manifest); device=device_for(a.device)
    net,meta=restore(a.checkpoint,device)
    if meta['task']!='damage' or meta['run_id']!=digest(a.manifest):
        raise ValueError('Expected a damage checkpoint trained with this manifest')
    donors=frame[(frame.split=='train') & frame.quality.isin([1,2])]
    data=Photos(donors,meta['size'],a.seed,cache_dir=a.cache_dir)
    result=[]
    with torch.inference_mode():
        for batch in loader(data,a):
            presence,severity=net(batch['x'].to(device))
            presence=presence.sigmoid().cpu().numpy(); severity=severity.sigmoid().cpu().numpy()
            for j,image in enumerate(batch['image']):
                selected=np.flatnonzero(presence[j]>=a.presence_threshold)
                selected=sorted(selected,key=lambda k:float(presence[j,k]*severity[j,k]),reverse=True)[:3]
                spec=[]
                for k in selected:
                    family=FAMILIES[k]
                    op=a.exposure_operator if family=='exposure' else a.noise_operator if family=='noise_compression' else OPERATORS[family][0]
                    spec.append(dict(operator=op,level=int(np.clip(np.rint(5*severity[j,k]),1,5)),
                                     estimated_presence=float(presence[j,k]),estimated_severity=float(severity[j,k])))
                if spec:
                    result.append(dict(image=image,patient=batch['patient'][j],split='train',
                                       provenance='natural_estimated',spec=spec))
    if not result: raise ValueError('No donors pass presence threshold; inspect estimator rather than silently relaxing it')
    profile=dict(version=1,donors=result,manifest_sha256=digest(a.manifest),checkpoint_sha256=digest(a.checkpoint),
                 options=vars(a),warning='Synthetic-trained estimates are proxies, not natural ground truth. Ambiguous operators are explicit user choices.')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(profile,indent=2))

if __name__=='__main__': main()
