"""Validate delivered Pawpularity manifests and every prepared photo.

This checks saved outputs independently of the image-export loop. It evaluates
only a constant training-mean reference on validation; it never scores test.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from furcast_data import PAW, verify_splits


def validate(out, photos, size=300, report_path=None):
    if not __debug__:
        raise RuntimeError('Validation requires assertions; do not use python -O')
    out, photos = Path(out), Path(photos)
    read = lambda name: pd.read_csv(out / name, dtype={'Id': str}, keep_default_na=False)
    labels = read('labels_and_splits.csv')
    images = read('image_splits.csv')
    inputs = read('image_inputs.csv')
    prepared = read('prepared_image_manifest.csv')
    verify_splits(images, 'key', 'duplicate_component')
    active = labels.loc[labels.split.isin(['train', 'validation', 'test'])]
    assert active.Id.is_unique and inputs.Id.is_unique and prepared.Id.is_unique
    assert set(active.Id) == set(inputs.Id) == set(prepared.Id)
    assert not (set(inputs.columns) & (set(PAW) | {'Pawpularity'}))
    assert active.exclusion_reason.eq('').all()
    assert images.loc[images.source_partition.eq('test'), 'split'].eq('excluded').all()
    joined = active.merge(inputs, on='Id', validate='one_to_one', suffixes=('_label', '_input'))
    assert joined.split_label.eq(joined.split_input).all()
    assert joined.duplicate_component_label.eq(joined.duplicate_component_input).all()
    edges = read('duplicate_candidates.csv')
    split_lookup = images.set_index('key').split.to_dict()
    cross_pairs = 0
    for a, b in edges[['id_a', 'id_b']].itertuples(index=False, name=None):
        sa, sb = split_lookup.get(a), split_lookup.get(b)
        if sa is not None and sb is not None:
            assert sa == 'excluded' or sb == 'excluded' or sa == sb
        else:
            cross_pairs += 1
            assert (sa if sa is not None else sb) == 'excluded'
    for split in ['train', 'validation', 'test']:
        table = read(f'labels_{split}.csv').sort_values('Id').reset_index(drop=True)
        expected = active.loc[active.split.eq(split), ['Id', 'Pawpularity']].sort_values('Id').reset_index(drop=True)
        pd.testing.assert_frame_equal(table, expected)
    manifest = prepared.merge(inputs[['Id', 'prepared_member']], on='Id', validate='one_to_one', suffixes=('', '_input'))
    assert manifest.prepared_member.eq(manifest.prepared_member_input).all()
    with ZipFile(photos) as z:
        assert len(z.namelist()) == len(set(z.namelist())) == len(manifest)
        assert set(z.namelist()) == set(manifest.prepared_member)
        for row in manifest.itertuples():
            data = z.read(row.prepared_member)  # checks CRC
            assert hashlib.sha256(data).hexdigest() == row.sha256
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                assert im.mode == 'RGB' and im.size == (size, size)
    audit = json.loads((out/'audit.json').read_text())
    with photos.open('rb') as f:
        zip_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    assert zip_sha == audit['prepared_zip_sha256']
    mean = float(active.loc[active.split.eq('train'), 'Pawpularity'].mean())
    target = active.loc[active.split.eq('validation'), 'Pawpularity'].to_numpy()
    result = {
        'status': 'passed', 'validated_prepared_images': len(manifest),
        'prepared_size': [size, size], 'prepared_zip_sha256': zip_sha,
        'split_counts': active.split.value_counts().to_dict(),
        'cross_source_pairs_quarantined': cross_pairs,
        'detected_duplicate_components_crossing_splits': 0,
        'train_mean_reference': {
            'prediction': mean,
            'validation_rmse': float(np.sqrt(np.mean((target-mean)**2))),
            'validation_mae': float(np.mean(np.abs(target-mean))),
            'test_evaluated': False,
            'note': 'Constant statistical reference, not a trained image model.'
        },
        'limits': 'Checks detected duplicate links only; does not establish complete animal identity or temporal independence.'
    }
    report_path = Path(report_path) if report_path else out/'output_validation.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--photos', required=True)
    parser.add_argument('--size', type=int, default=300)
    parser.add_argument('--report', help='Separate report path, avoids mutating DVC dependencies')
    args = parser.parse_args()
    print(json.dumps(validate(args.out, args.photos, args.size, args.report), indent=2))
