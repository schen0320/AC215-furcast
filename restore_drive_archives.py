"""Restore the exact four image ZIPs from the authorized Drive handoff parts.

Download this script and the relevant .partNN files to the same folder.
Run: python restore_drive_archives.py --archive PetFinder_Default_Photos_224.zip
Omit --archive to restore all four archives. This script uses only the standard library.
"""
import argparse
import hashlib
from pathlib import Path

ARCHIVES = [{'name': 'Oxford_Pet_Images_Part_1.zip',
  'bytes': 419431026,
  'sha256': 'a4e4dcb6863c47cc970e8578aefc22340ab9bae0d5053936f61e0a1e05709c37',
  'parts': [{'name': 'Oxford_Pet_Images_Part_1.zip.part01',
             'bytes': 94371840,
             'sha256': 'c44d958e3dacb93aafee13f861186308ee9f71fe2a2a8229c5ac4bab552e44ff'},
            {'name': 'Oxford_Pet_Images_Part_1.zip.part02',
             'bytes': 94371840,
             'sha256': '0a90029b2bf7baa7a55109c6c2f27228e739fcb6c354576c39954f92250ebc6d'},
            {'name': 'Oxford_Pet_Images_Part_1.zip.part03',
             'bytes': 94371840,
             'sha256': '5751dd6fe7ba12354c5acc2d36f92f3ea3e09e93f3f25f92267c9aa3965022bb'},
            {'name': 'Oxford_Pet_Images_Part_1.zip.part04',
             'bytes': 94371840,
             'sha256': '735d054bb171f93d7d6e14b40ddb6cb9ba83a9b1249bae69424a946523db865f'},
            {'name': 'Oxford_Pet_Images_Part_1.zip.part05',
             'bytes': 41943666,
             'sha256': 'cd0fab85dd712ea82fa0e67b62535678326ae77c5d3aa5d14b62905d2850e1be'}]},
 {'name': 'Oxford_Pet_Images_Part_2.zip',
  'bytes': 372489197,
  'sha256': '3aab81f14cd09359f4f8314d9855d98e7d5ea4e3b22630cd64647cc44dbf7fca',
  'parts': [{'name': 'Oxford_Pet_Images_Part_2.zip.part01',
             'bytes': 94371840,
             'sha256': 'b89b7e95b71e3f5b3e82706980676418fba83d959490d8cfe43188ef7c84af82'},
            {'name': 'Oxford_Pet_Images_Part_2.zip.part02',
             'bytes': 94371840,
             'sha256': '58b40c163f67cfa34cfc3c759e95bf1190019a18e542c01c1f60994447a0488f'},
            {'name': 'Oxford_Pet_Images_Part_2.zip.part03',
             'bytes': 94371840,
             'sha256': '0c8fd8859a078488cf5f449cbee2ee9f3067f800165840255bb6aa980e022e06'},
            {'name': 'Oxford_Pet_Images_Part_2.zip.part04',
             'bytes': 89373677,
             'sha256': '021b40558dc7846fc420f0b3e12af4711d620648d19ae3f5d740cb23cece94ad'}]},
 {'name': 'PetFinder_Default_Photos_224.zip',
  'bytes': 153239169,
  'sha256': '75c68bc5f51361b828763fc4912b9e8d758e57a3f2982b68f0f5b87cfa0f3e3b',
  'parts': [{'name': 'PetFinder_Default_Photos_224.zip.part01',
             'bytes': 94371840,
             'sha256': '30ea1ab49024107354f2630b18df0a1bfb98e29218c7b2759e52031f273b3fc4'},
            {'name': 'PetFinder_Default_Photos_224.zip.part02',
             'bytes': 58867329,
             'sha256': 'd26f0f43c3805108bfa0bb1d665f66bf646585b895635051f317ed43d82a4d93'}]},
 {'name': 'Stanford_Dogs_224_Preprocessed.zip',
  'bytes': 326735460,
  'sha256': 'a75f9bc5bf1338484cc382144cb760ae555cd04f082e2d5a7879ef56034358bf',
  'parts': [{'name': 'Stanford_Dogs_224_Preprocessed.zip.part01',
             'bytes': 94371840,
             'sha256': 'f3b8925ce793ca89ea319eaebeaf16ea0d4dd48f4152038c5dd4afcf1875eb64'},
            {'name': 'Stanford_Dogs_224_Preprocessed.zip.part02',
             'bytes': 94371840,
             'sha256': '46437540de84ffad3604f07a17816cc011c0a7509ba375244ec037099bf27280'},
            {'name': 'Stanford_Dogs_224_Preprocessed.zip.part03',
             'bytes': 94371840,
             'sha256': '526995cbc5a1849ca5bae4d9799e4799f3ea93ee31c5bd23c5e2bbeda9efc894'},
            {'name': 'Stanford_Dogs_224_Preprocessed.zip.part04',
             'bytes': 43619940,
             'sha256': '819049e14945a3c445811fbff99f09422e203b3a2b38876988b75523f3ad0c1f'}]}]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restore(record, parts_dir, out_dir):
    missing = [p['name'] for p in record['parts'] if not (parts_dir / p['name']).is_file()]
    if missing:
        raise FileNotFoundError('Download the missing parts first: ' + ', '.join(missing))
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / record['name']
    if output.exists():
        if output.stat().st_size == record['bytes'] and sha256(output) == record['sha256']:
            print('Already verified:', output)
            return output
        raise FileExistsError('Existing output does not match; choose a new output directory: ' + str(output))
    temporary = output.with_name(output.name + '.assembling')
    if temporary.exists():
        raise FileExistsError('Existing partial output; inspect it before retrying: ' + str(temporary))
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open('xb') as destination:
            for part in record['parts']:
                part_hash = hashlib.sha256()
                count = 0
                with (parts_dir / part['name']).open('rb') as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        part_hash.update(block)
                        digest.update(block)
                        destination.write(block)
                        count += len(block)
                if count != part['bytes'] or part_hash.hexdigest() != part['sha256']:
                    raise ValueError('Part checksum/size mismatch: ' + part['name'])
                total += count
        if total != record['bytes'] or digest.hexdigest() != record['sha256']:
            raise ValueError('Restored archive checksum/size mismatch: ' + record['name'])
        temporary.rename(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print('Restored and SHA-256 verified:', output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', type=Path, default=Path('.'))
    parser.add_argument('--out', type=Path, default=Path('restored_images'))
    parser.add_argument('--archive', choices=[r['name'] for r in ARCHIVES],
                        help='Restore one named archive; default restores all four')
    args = parser.parse_args()
    selected = [r for r in ARCHIVES if not args.archive or r['name'] == args.archive]
    for record in selected:
        restore(record, args.parts_dir, args.out)


if __name__ == '__main__':
    main()
