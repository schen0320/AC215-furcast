"""Fetch the official Stanford Dogs files into a new snapshot directory."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

BASE = 'http://vision.stanford.edu/aditya86/ImageNetDogs/'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=False)
    records = []
    manifest = {'complete': False, 'files': records,
                'license_status': 'Official page/README request citation; no explicit image redistribution or commercial license found. ImageNet image rights remain relevant.',
                'transport': 'Official download links use HTTP; local hashes identify bytes but are not publisher-signed integrity proofs.'}
    try:
        for name in ['lists.tar', 'annotation.tar', 'images.tar']:
            target = root / name
            partial = root / (name + '.partial')
            digest = hashlib.sha256()
            total = 0
            with urlopen(BASE + name, timeout=55) as r, partial.open('wb') as f:
                expected = r.headers.get('Content-Length')
                while chunk := r.read(1024 * 1024):
                    f.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                    if total % (64 * 1024 * 1024) == 0:
                        print(name, total // (1024 * 1024), 'MiB', flush=True)
                if expected is not None and total != int(expected):
                    raise ValueError('Content-Length mismatch for ' + name)
            partial.replace(target)
            records.append({'name': name, 'url': BASE + name, 'bytes': total,
                            'sha256': digest.hexdigest(),
                            'retrieved_at': dt.datetime.now(dt.timezone.utc).isoformat()})
            print(name, total, 'bytes downloaded', flush=True)
        manifest['complete'] = True
    finally:
        (root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    main()
