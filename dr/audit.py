"""Read-only EyeQ/EyePACS audit, including ZIP inventory without extraction.

python -m dr.audit --labels train.csv test.csv --images data/EyePACS --out audit.json
Use --archive competition.zip instead to inspect archive member matching first.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
import pandas as pd
from PIL import Image
from .runtime import protect


def audit(labels, images=None, archive=None):
    errors, warnings, frames = [], [], []
    for csv in labels:
        t = pd.read_csv(csv)
        missing = {'image','DR_grade','quality'}-set(t)
        if missing:
            errors.append(f'{csv}: missing columns {sorted(missing)}')
            continue
        t['label_source'] = str(csv)
        frames.append(t)
    if not frames:
        return dict(status='failed',errors=errors,warnings=warnings)
    table = pd.concat(frames,ignore_index=True)
    if table[['image','DR_grade','quality']].isna().any().any():
        errors.append('Missing labels or image identifiers')
    if table.image.duplicated().any():
        errors.append('Duplicate image IDs across label files')
    if not set(table.DR_grade) <= set(range(5)) or not set(table.quality) <= {0,1,2}:
        errors.append('Invalid grade/quality encoding')
    table['patient'] = table.image.astype(str).str.extract(r'^(.+)_(?:left|right)\.[^.]+$',flags=re.I)[0]
    if table.patient.isna().any():
        errors.append('Unrecognized patient naming convention')
    source_overlap = int((table.groupby('patient').label_source.nunique()>1).sum())
    if source_overlap:
        warnings.append(f'{source_overlap} patients occur in multiple official label files; keep eyes grouped')
    wanted = set(table.image.astype(str))
    paths = {}
    if archive:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if not info.is_dir() and Path(info.filename).name in wanted:
                    paths.setdefault(Path(info.filename).name,[]).append(info.filename)
        warnings.append('Archive inventory only: image content/readability not verified; nested archives are not expanded')
    else:
        for p in Path(images).rglob('*'):
            if p.is_file() and p.name in wanted:
                paths.setdefault(p.name,[]).append(str(p.resolve()))
    missing = sorted(wanted-paths.keys())
    duplicates = {k:v for k,v in paths.items() if len(v)>1}
    if missing: errors.append(f'{len(missing)} missing images')
    if duplicates: errors.append(f'{len(duplicates)} ambiguous basenames')
    hashes, unreadable = {}, []
    if not archive:
        for name,matches in paths.items():
            if len(matches)!=1: continue
            try:
                with Image.open(matches[0]) as im:
                    im.load()
                    rgb = im.convert('RGB')
                    h = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                    hashes.setdefault(h,[]).append(name)
            except (OSError,ValueError) as exc:
                unreadable.append(dict(image=name,error=str(exc)))
    content_duplicates = [v for v in hashes.values() if len(v)>1]
    if content_duplicates: errors.append('Duplicate decoded image content found')
    if unreadable: errors.append('Unreadable images found')
    return dict(status='failed' if errors else ('inventory_only' if archive else 'passed'),
                errors=errors,warnings=warnings,rows=len(table),patients=int(table.patient.nunique()),
                missing=missing,duplicate_basenames=duplicates,duplicate_content=content_duplicates,
                unreadable=unreadable,matched_members=paths,
                label_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in labels},
                limitations=['Exact decoded duplicates only; near duplicates and release provenance require review',
                             'Does not independently reconcile EyePACS competition grades'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',nargs='+',required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--images'); source.add_argument('--archive')
    p.add_argument('--out',required=True); p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(); protect(a.out,a.overwrite)
    report=audit(a.labels,a.images,a.archive)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,indent=2))
    print(report['status'])
    if report['status']=='failed': raise SystemExit(1)

if __name__=='__main__': main()
