"""Offline, fail-closed acquisition boundary for the frozen Pawpularity release.

No credentials, network requests, cloud creation or automatic DVC push. Hashes
bind local bytes to a reviewed release; they do not grant data-use permission.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import stat
import tempfile
from zipfile import ZipFile

import pandas as pd
from furcast_data import PAW
from prepare_pawpularity import sha, validate_labels

CONTRACT = Path('config/pawpularity-source.json')


def expect(condition, message):
    if not condition:
        raise ValueError(message)


def check_hash(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}. Obtain the authorized release; do not use fixtures.')
    actual = sha(path)
    expect(actual == expected, f'SHA-256 mismatch: {path.name}. Stop and review the data version.')
    return actual


def checked_members(z):
    infos = z.infolist()
    names = [i.filename for i in infos]
    expect(len(names) == len(set(names)), 'Duplicate ZIP member names')
    for i in infos:
        p = PurePosixPath(i.filename)
        expect(not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename
               and ':' not in i.filename and str(p) == i.filename.rstrip('/'), 'Unsafe ZIP path')
        mode = i.external_attr >> 16
        expect(not stat.S_ISLNK(mode), 'ZIP symlinks are forbidden')
        expect(not i.flag_bits & 1, 'Encrypted ZIP member')
    expect(sum(i.file_size for i in infos) <= 3_000_000_000, 'Unexpected expanded archive size')
    return infos


def check_raw(raw, contract):
    raw = Path(raw)
    check_hash(raw/'train.csv', contract['train_csv_sha256'])
    train = pd.read_csv(raw/'train.csv', dtype={'Id': str})
    validate_labels(train)
    expect(len(train) == contract['train_rows'], 'Unexpected training row count')
    test = pd.read_csv(raw/'test.csv', dtype={'Id': str})
    expect(set(test.columns) == {'Id', *PAW}, 'Unexpected example-test schema')
    expect(len(test) == contract['example_test_rows'], 'Unexpected example-test row count')
    expect(test.Id.is_unique and test.Id.str.fullmatch('[0-9a-f]{32}').all(), 'Invalid example IDs')
    expect(test[PAW].isin([0, 1]).all().all(), 'Invalid example attributes')
    expect(not set(test.Id) & set(train.Id), 'Training/example IDs overlap')
    for part, frame in [('train', train), ('test', test)]:
        actual = {p.name for p in (raw/part).iterdir()}
        expect(actual == set(frame.Id + '.jpg'), f'{part} image/label mismatch')
    sample = pd.read_csv(raw/'sample_submission.csv', dtype={'Id': str})
    expect(set(sample.columns) == {'Id', 'Pawpularity'} and sample.Id.tolist() == test.Id.tolist(),
           'Invalid submission example')
    return {'status': 'passed_input_contract', 'train_rows': len(train),
            'example_test_rows': len(test), 'train_csv_sha256': sha(raw/'train.csv')}


def check_reference(reference, contract):
    for name, digest in contract['reference_sha256'].items():
        path = Path(reference)/name
        check_hash(path, digest)
        frame = pd.read_csv(path, dtype={'key': str, 'phash': str, 'pixel_sha256': str})
        if 'image_key' in frame:
            frame = frame.rename(columns={'image_key': 'key'})
        if 'decode_ok' in frame:
            frame = frame.loc[frame.decode_ok.eq(True)]
        expect({'key', 'phash', 'pixel_sha256'} <= set(frame), 'Missing fingerprint fields')
        expect(not frame.empty and frame.key.is_unique and frame.key.notna().all(), 'Invalid fingerprint keys')
        expect(frame.phash.str.fullmatch('[0-9a-f]{16}').all(), 'Invalid pHash')
        expect(frame.pixel_sha256.str.fullmatch('[0-9a-f]{64}').all(), 'Invalid pixel hash')


def extract(archive, out, contract):
    archive, out = Path(archive), Path(out)
    expect(not out.exists(), 'Choose a new raw output directory')
    check_hash(archive, contract['archive_sha256'])
    expect(archive.stat().st_size == contract['archive_bytes'], 'Archive size mismatch')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix='.pawpularity-') as tmp:
        root = Path(tmp)/'raw'
        with ZipFile(archive) as z:
            infos = checked_members(z)
            expect(z.testzip() is None, 'ZIP CRC failure')
            for i in infos:
                if i.is_dir():
                    continue
                p = PurePosixPath(i.filename)
                expect(i.filename in {'train.csv', 'test.csv', 'sample_submission.csv'}
                       or (len(p.parts) == 2 and p.parts[0] in {'train', 'test'}
                           and p.suffix == '.jpg'), 'Unexpected archive member')
            z.extractall(root)
        report = check_raw(root, contract)
        report['archive_sha256'] = sha(archive)
        (root/'input_validation.json').write_text(json.dumps(report, indent=2)+'\n')
        root.rename(out)
    return report


def import_delivery(package, out, contract):
    package, out = Path(package), Path(out)
    expect(not out.exists(), 'Choose a new delivery output directory')
    check_hash(package, contract['processed_package_sha256'])
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix='.delivery-') as tmp:
        with ZipFile(package) as z:
            checked_members(z)
            expect(z.testzip() is None, 'Delivery ZIP CRC failure')
            z.extractall(tmp)
        root = Path(tmp)/'FurCast_Pawpularity_Processed_v1'
        expect(root.is_dir(), 'Unexpected delivery layout')
        for line in (root/'SHA256SUMS.txt').read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            name = name.strip().lstrip('*')
            target = root/name
            expect(target.resolve().is_relative_to(root.resolve()), 'Unsafe checksum path')
            check_hash(target, digest)
        check_reference(root/'reference', contract)
        root.rename(out)
    return {'status': 'verified_delivery_import', 'package_sha256': sha(package)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', default=str(CONTRACT))
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('extract'); a.add_argument('--archive', required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('check'); a.add_argument('--raw', required=True); a.add_argument('--reference', required=True)
    a = sub.add_parser('import-delivery'); a.add_argument('--package', required=True); a.add_argument('--out', default='artifacts/delivery')
    a = p.parse_args()
    if not __debug__:
        p.error('Do not use python -O; inherited validation requires assertions')
    contract = json.loads(Path(a.contract).read_text())
    if a.command == 'extract':
        report = extract(a.archive, a.out, contract)
    elif a.command == 'check':
        report = check_raw(a.raw, contract); check_reference(a.reference, contract)
    else:
        report = import_delivery(a.package, a.out, contract)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
