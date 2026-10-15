"""Decode authorized official PetFinder images; record reproducible fingerprints.

Never downloads or authenticates. All original images are retained unchanged.
The output manifest is reusable; no claim that a pHash proves animal identity.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
from audit_oxford_images import perceptual_hash


def inspect_image(args):
    root, rel = args
    path = Path(root)/rel
    raw = path.read_bytes()
    stem, number = path.stem.rsplit('-', 1)
    row = {'image_key': 'petfinder:'+rel, 'source_file': rel,
           'source_partition': rel.split('_images/')[0], 'PetID': stem,
           'image_number': int(number), 'bytes': len(raw),
           'file_sha256': hashlib.sha256(raw).hexdigest()}
    try:
        with Image.open(io.BytesIO(raw)) as original:
            im = ImageOps.exif_transpose(original).convert('RGB'); im.load()
            row.update(decode_ok=True, width=im.width, height=im.height,
                       original_mode=original.mode,
                       pixel_sha256=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest(),
                       phash=f'{perceptual_hash(im):016x}')
    except Exception as exc:
        row.update(decode_ok=False, error=type(exc).__name__)
    return row


def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw',required=True)
    p.add_argument('--out',required=True); p.add_argument('--workers',type=int,default=8)
    a=p.parse_args(); root=Path(a.raw).resolve(); out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    files=sorted(str(f.relative_to(root)) for part in ['train_images','test_images']
                 for f in (root/part).glob('*.jpg'))
    if not files: raise ValueError('No real image files')
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(inspect_image, [(str(root),f) for f in files],chunksize=64):
            rows.append(row)
            if len(rows)%5000==0: print('Decoded',len(rows),'of',len(files),flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(out/'image_manifest.csv',index=False)
    audit={'image_count':len(frame),'partition_counts':frame.source_partition.value_counts().to_dict(),
           'decode_failures':int((~frame.decode_ok).sum()),
           'rule':'SHA256 of raw file, SHA256 of EXIF-oriented RGB size+pixels, 64-bit DCT pHash',
           'image_transform':'No changes to original files in this audit'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
