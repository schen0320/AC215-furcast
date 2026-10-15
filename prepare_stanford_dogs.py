"""Audit actual Stanford bytes, protect held-out images, export padded 224px RGB.

Near matches are quarantined conservatively, not asserted to be the same animal.
The Oxford hashes are from the previously decoded original Oxford image archive.
Breed class IDs remain dataset-specific; no unreviewed taxonomy merge is performed.
"""
import argparse
import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from audit_oxford_images import perceptual_hash
from furcast_data import connected_groups, write_json


def load_lists(path):
    frames = {}
    with tarfile.open(path) as t:
        for name in ['file', 'train', 'test']:
            m = loadmat(io.BytesIO(t.extractfile(name + '_list.mat').read()))
            frames[name] = pd.DataFrame({
                'image_id': [str(x[0][0]) for x in m['file_list']],
                'class_id': m['labels'].ravel().astype(int)})
    full, train, test = (frames[x] for x in ['file', 'train', 'test'])
    assert not full.image_id.duplicated().any()
    assert set(train.image_id).isdisjoint(test.image_id)
    assert set(full.image_id) == set(train.image_id) | set(test.image_id)
    labels = full.set_index('image_id').class_id
    for subset in [train, test]:
        assert np.array_equal(labels.loc[subset.image_id], subset.class_id)
    tr, va = train_test_split(train.image_id, test_size=.2,
                             stratify=train.class_id, random_state=215)
    full['split'] = np.where(full.image_id.isin(test.image_id), 'test',
                            np.where(full.image_id.isin(va), 'validation', 'train'))
    full['breed_name'] = full.image_id.str.split('/').str[0].str.split('-', n=1).str[1]
    full['species'] = 'Dog'
    assert full.groupby('class_id').breed_name.nunique().eq(1).all()
    return full


def find_edges(frame):
    """Exact uint64 Hamming search in bounded batches (including cross-dataset)."""
    ids = frame.key.to_numpy()
    values = np.array([int(x, 16) for x in frame.phash], dtype=np.uint64)
    byte_bits = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    result = {}
    for start in range(0, len(frame), 64):
        xor = values[start:start + 64, None] ^ values[None, :]
        distance = byte_bits[xor.view(np.uint8).reshape(len(xor), len(frame), 8)].sum(axis=2)
        aa, bb = np.where(distance <= 6)
        for a, b in zip(aa, bb):
            a = start + int(a); b = int(b)
            if a > b:
                result[(a, b)] = int(distance[a - start, b])
    for _, g in frame.reset_index(drop=True).groupby('pixel_sha256'):
        idx = g.index.tolist()
        for a in idx:
            for b in idx:
                if a > b:
                    result[(a, b)] = (int(values[a]) ^ int(values[b])).bit_count()
    pix = frame.pixel_sha256.to_numpy()
    return pd.DataFrame([
        {'id_a': ids[a], 'id_b': ids[b], 'phash_distance': d,
         'kind': 'exact_pixels' if pix[a] == pix[b] else 'perceptual_candidate',
         'cross_dataset': ids[a].split(':', 1)[0] != ids[b].split(':', 1)[0]}
        for (a, b), d in sorted(result.items())],
        columns=['id_a', 'id_b', 'phash_distance', 'kind', 'cross_dataset'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--oxford-manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--export-zip', required=True)
    a = p.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    table = load_lists(raw / 'lists.tar')
    bbox = {}
    with tarfile.open(raw / 'annotation.tar') as t:
        for m in t:
            if not m.isfile() or '/.' in m.name or m.name.startswith('._'):
                continue
            rel = m.name.split('/', 1)[-1] + '.jpg'
            try:
                doc = ET.fromstring(t.extractfile(m).read())
                width = int(doc.findtext('size/width')); height = int(doc.findtext('size/height'))
                boxes = doc.findall('object/bndbox')
                valid = bool(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = [int(box.findtext(k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
                    valid &= 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                bbox[rel] = {'bbox_count': len(boxes), 'bbox_valid': bool(valid)}
            except (ET.ParseError, TypeError, ValueError):
                bbox[rel] = {'bbox_count': 0, 'bbox_valid': False}
    rows = []
    with tarfile.open(raw / 'images.tar') as t:
        for m in t:
            if not m.isfile() or not m.name.lower().endswith(('.jpg', '.jpeg', '.png')) or '/.' in m.name:
                continue
            data = t.extractfile(m).read()
            rel = m.name.split('/', 1)[-1]
            row = {'image_id': rel, 'archive_member': m.name, 'file_sha256': hashlib.sha256(data).hexdigest(),
                   'bytes': len(data), **bbox.get(rel, {'bbox_count': 0, 'bbox_valid': False})}
            try:
                with Image.open(io.BytesIO(data)) as orig:
                    im = ImageOps.exif_transpose(orig).convert('RGB'); im.load()
                    row.update(decode_ok=True, width=im.width, height=im.height,
                               pixel_sha256=hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest(),
                               phash=f'{perceptual_hash(im):016x}')
            except Exception as e:
                row.update(decode_ok=False, error=type(e).__name__)
            rows.append(row)
            if len(rows) % 2000 == 0: print('Decoded', len(rows), flush=True)
    images = pd.DataFrame(rows)
    assert not images.image_id.duplicated().any()
    table = table.merge(images, on='image_id', how='left', validate='one_to_one')
    table['key'] = 'stanford:' + table.image_id
    table['clean_split'] = table.split
    table['exclusion_reason'] = ''
    table.loc[~table.decode_ok.fillna(False).astype(bool), ['clean_split', 'exclusion_reason']] = ['excluded', 'missing_or_undecodable']
    for _, g in table.dropna(subset=['pixel_sha256']).groupby('pixel_sha256'):
        if g.class_id.nunique() > 1:
            table.loc[g.index, ['clean_split', 'exclusion_reason']] = ['excluded', 'exact_pixel_label_conflict']
    ox = pd.read_csv(a.oxford_manifest, dtype={'phash': str})
    ox['key'] = 'oxford:' + ox.image_id
    fields = ['key', 'phash', 'pixel_sha256', 'clean_split']
    both = pd.concat([ox[fields], table[fields]], ignore_index=True)
    both = both.dropna(subset=['phash', 'pixel_sha256']).reset_index(drop=True)
    print('Checking duplicate candidates across', len(both), 'images', flush=True)
    edges = find_edges(both)
    edges.to_csv(out / 'duplicate_candidates.csv', index=False)
    both['duplicate_component'] = connected_groups(both, 'key', edges=edges)
    both['previous_split'] = both.clean_split
    both['exclusion_reason'] = ''
    priority = {'train': 1, 'validation': 2, 'test': 3, 'excluded': 0}
    for _, g in both.groupby('duplicate_component'):
        highest = g.clean_split.map(priority).max()
        drop = g.index[g.clean_split.map(priority).between(1, highest - 1)]
        both.loc[drop, ['clean_split', 'exclusion_reason']] = ['excluded', 'cross_split_duplicate_candidate']
    kept = both[both.clean_split.ne('excluded')]
    assert kept.groupby('duplicate_component').clean_split.nunique().le(1).all()
    both.to_csv(out / 'joint_duplicate_split_guard.csv', index=False)
    decisions = both.set_index('key')
    ready = table.key.isin(decisions.index)
    for col in ['clean_split', 'exclusion_reason', 'duplicate_component']:
        if col == 'exclusion_reason':
            update = table.loc[ready, 'key'].map(decisions[col])
            table.loc[update.index[update.ne('')], col] = update.loc[update.ne('')]
        else:
            table.loc[ready, col] = table.loc[ready, 'key'].map(decisions[col])
    table['model_class_index'] = table.class_id - 1
    table['prepared_member'] = table.apply(lambda r: f'{r.clean_split}/{r.class_id:03d}_{r.breed_name}/{Path(r.image_id).name}' if r.clean_split != 'excluded' else '', axis=1)
    table.to_csv(out / 'image_splits.csv', index=False)
    counts = table.groupby(['class_id', 'breed_name', 'clean_split']).size().unstack(fill_value=0).reset_index()
    counts.to_csv(out / 'breed_counts.csv', index=False)
    keep = table[table.clean_split.ne('excluded')].set_index('archive_member')
    zip_path = Path(a.export_zip); zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists(): raise ValueError('Refusing to overwrite image export')
    with ZipFile(zip_path, 'w', compression=ZIP_STORED) as z, tarfile.open(raw / 'images.tar') as t:
        n = 0
        for m in t:
            if m.name not in keep.index: continue
            with Image.open(io.BytesIO(t.extractfile(m).read())) as orig:
                im = ImageOps.exif_transpose(orig).convert('RGB')
                im.thumbnail((224, 224), Image.Resampling.LANCZOS)
                canvas = Image.new('RGB', (224, 224), (128, 128, 128))
                canvas.paste(im, ((224-im.width)//2, (224-im.height)//2))
                buf = io.BytesIO(); canvas.save(buf, format='JPEG', quality=90)
                z.writestr(keep.at[m.name, 'prepared_member'], buf.getvalue()); n += 1
        z.write(out / 'image_splits.csv', 'image_splits.csv')
        z.write(out / 'breed_counts.csv', 'breed_counts.csv')
        z.writestr('README.txt', 'Stanford Dogs research preprocessing: RGB, EXIF orientation, aspect-ratio-preserving fit within 224 x 224, gray padding, JPEG quality 90. No random augmentation. Excluded candidates are not exported. Labels are dataset-specific. Keep all crops/augmentations in their assigned split. Use joint_duplicate_split_guard.csv from the starter if combining with Oxford. Not a shelter breed benchmark or an adoption model. Original source and rights: http://vision.stanford.edu/aditya86/ImageNetDogs/main.html. Cite Khosla et al. 2011 and Deng et al. 2009. No explicit redistribution/commercial license confirmed.\n')
    report = {'listed_images': len(table), 'archive_images': len(images), 'classes': int(table.class_id.nunique()),
              'decode_or_missing_failures': int((~table.decode_ok.fillna(False).astype(bool)).sum()),
              'invalid_or_missing_bbox_rows': int((~table.bbox_valid.fillna(False).astype(bool)).sum()),
              'original_split_counts': table.split.value_counts().to_dict(),
              'clean_split_counts': table.clean_split.value_counts().to_dict(),
              'clean_split_class_counts': table[table.clean_split.ne('excluded')].groupby('clean_split').class_id.nunique().to_dict(),
              'all_joint_candidate_pairs': len(edges), 'cross_dataset_candidate_pairs': int(edges.cross_dataset.sum()),
              'exact_pixel_pairs': int(edges.kind.eq('exact_pixels').sum()),
              'exported_images': n, 'export_zip_bytes': zip_path.stat().st_size,
              'raw_download_manifest': json.loads((raw/'download_manifest.json').read_text()),
              'scope': 'Independent Stanford breed benchmark. No adoption labels; no cat breeds or shelter-domain coverage guarantee. Oxford/Stanford label mapping needs review before joint model training.',
              'duplicate_review': 'pHash <=6 candidate components isolated by test > validation > train priority; human review pending. Does not prove same-animal independence.',
              'bbox_policy': 'Audited only. Full images used; invalid boxes do not silently crop data.',
              'pixel_preprocessing': 'RGB EXIF transpose, fit inside 224x224 without upscaling small images, gray padding, JPEG quality 90; fixed train/validation/test transforms; augmentation left to model owner.'}
    write_json(out / 'audit.json', report)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
