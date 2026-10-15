"""Reassemble authorized original archive parts, verifying bytes before publish."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--parts-dir',required=True)
    p.add_argument('--manifest',default=str(Path(__file__).resolve().parent/'evidence/petfinder_original_parts.json'))
    p.add_argument('--out',required=True);a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text());out=Path(a.out)
    if out.exists():raise ValueError('Output exists; choose a new path')
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_name(out.name+'.assembling')
    if temp.exists():raise ValueError('Temporary output exists')
    digest=hashlib.sha256();total=0
    try:
        with temp.open('xb') as dst:
            for part in manifest['parts']:
                source=Path(a.parts_dir)/part['filename'];h=hashlib.sha256();count=0
                with source.open('rb') as src:
                    for block in iter(lambda:src.read(8*1024*1024),b''):
                        h.update(block);digest.update(block);dst.write(block);count+=len(block)
                if count!=part['bytes'] or h.hexdigest()!=part['sha256']:raise ValueError('Part integrity failure: '+source.name)
                total+=count
        if total!=manifest['archive_bytes'] or digest.hexdigest()!=manifest['archive_sha256']:
            raise ValueError('Archive integrity failure')
        temp.rename(out);print(str(out.resolve()))
    except Exception:
        temp.unlink(missing_ok=True);raise


if __name__=='__main__':main()
