"""Add single-image representation-distance baselines to feature tables.

A reference centroid is fitted only on Good ORIGINAL training images. Unlike paired
clean/corrupt distance, this feature is available for unpaired natural test images.
It measures distance from a training distribution, not clinical information loss.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .training import digest
from .policy import read_table
from .runtime import protect_all

S=['sem_centroid_distance','sem_norm_deviation']


def read_embeddings(path):
    path=Path(path)
    meta=json.loads(path.with_suffix('.json').read_text())
    if meta['embedding_sha256']!=digest(path) or meta['rows_sha256']!=digest(path.with_suffix('.rows.csv')):
        raise ValueError('Embedding cache contents differ from metadata')
    with np.load(path,allow_pickle=False) as arrays:
        z=arrays['embeddings']
    rows=pd.read_csv(path.with_suffix('.rows.csv'),dtype={'image':str,'patient':str})
    if len(rows)!=len(z) or rows.duplicated(['image','variant']).any() or not np.isfinite(z).all():
        raise ValueError('Invalid embedding cache')
    return rows,z,meta


def reference_scores(reference, current):
    norms=np.linalg.norm(reference,axis=1)
    normalized=reference/np.maximum(norms[:,None],1e-12)
    centroid=normalized.mean(axis=0)
    length=np.linalg.norm(centroid)
    if length<1e-12: raise ValueError('Degenerate reference centroid')
    current_norm=np.linalg.norm(current,axis=1)
    distance=1-(current/np.maximum(current_norm[:,None],1e-12))@(centroid/length)
    deviation=np.abs(current_norm-norms.mean())/max(float(norms.std()),1e-8)
    return np.column_stack([distance,deviation])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True); p.add_argument('--embeddings',required=True)
    p.add_argument('--reference',required=True,help='Training split embeddings, including Good originals')
    p.add_argument('--out',required=True); p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    output=Path(a.out)
    protect_all([output,output.with_suffix(".meta.json")],a.overwrite)
    table,meta=read_table(a.features)
    rows,z,em=read_embeddings(a.embeddings); ref,rz,rm=read_embeddings(a.reference)
    sampling=json.loads(Path(a.features).with_suffix('.sampling.json').read_text())
    for key in ('seed','mode','corruption_size'):
        if sampling.get(key)!=em['args'].get(key):
            raise ValueError(f'Feature/embedding view generation mismatch: {key}')
    if sampling.get('condition_profile_sha256') or sampling.get('excluded_family'):
        raise ValueError('Conditioned/excluded embedding joins need an explicit matched-view protocol; not supported yet')
    if set(ref.split)!={'train'}: raise ValueError('Reference must contain only training patients')
    if set(ref.patient)&set(table.patient): raise ValueError('Reference/evaluation patient overlap')
    for key in ('checkpoint_sha256','manifest_sha256','precision'):
        if em[key]!=rm[key]: raise ValueError(f'Representation protocol mismatch: {key}')
    for key in ('encoder','size','corruption_size'):
        if em['args'].get(key)!=rm['args'].get(key): raise ValueError(f'Representation protocol mismatch: {key}')
    # External cohorts may use a different manifest, but require a dedicated protocol in a future extension.
    if em['manifest_sha256']!=meta['training_manifest']:
        raise ValueError('This command currently supports development-manifest partitions only')
    selection=(ref.quality==0)&(ref.variant=='original')
    if not selection.any(): raise ValueError('No clean training reference images')
    values=reference_scores(rz[selection],z)
    for i,name in enumerate(S): rows[name]=values[:,i]
    joined=table.merge(rows[['image','variant','patient',*S]],on=['image','variant','patient'],how='left',validate='one_to_one',sort=False)
    if joined[S].isna().any().any(): raise ValueError('Features lack matching representation rows')
    meta['representation']=dict(reference_sha256=digest(a.reference),checkpoint_sha256=em['checkpoint_sha256'],
        encoder=em['args']['encoder'],size=em['args']['size'],precision=em['precision'],method='clean-train-centroid-v1')
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    joined.to_csv(out,index=False); out.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__': main()
