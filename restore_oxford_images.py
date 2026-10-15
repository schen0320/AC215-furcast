"""Restore the original byte-for-byte Oxford archive from the two supplied ZIP parts.

python restore_oxford_images.py --parts /path/to/downloads --out raw/oxford_images.tar.gz
"""
import argparse,hashlib,shutil,zipfile
from pathlib import Path

EXPECTED='67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d'

def main():
    p=argparse.ArgumentParser();p.add_argument('--parts',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise SystemExit('Output already exists; choose another path.')
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_name(out.name+'.partial')
    if tmp.exists():raise SystemExit('Partial file already exists; inspect it before retrying.')
    h=hashlib.sha256()
    try:
        with tmp.open('wb') as f:
            for n in (1,2):
                with zipfile.ZipFile(Path(a.parts)/f'Oxford_Pet_Images_Part_{n}.zip') as z:
                    with z.open(f'oxford_images.tar.gz.part{n}') as src:
                        while chunk:=src.read(1024*1024):f.write(chunk);h.update(chunk)
        if h.hexdigest()!=EXPECTED:raise ValueError('Checksum mismatch; keep partial output for inspection.')
        tmp.rename(out);print('Restored verified archive:',out)
    except Exception:
        raise

if __name__=='__main__':main()
