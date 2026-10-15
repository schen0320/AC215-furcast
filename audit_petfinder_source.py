"""Audit authorized real PetFinder CSV/lookups before claiming breed coverage.

No download and no invented records. Candidate breed counts are listing counts,
not proof of correct visual breed labels. Existing furcast_data.py owns splits.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from furcast_data import PF_REQUIRED, profile, require

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--breed-labels', required=True)
    p.add_argument('--images', help='Authorized extracted train_images directory')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    df = pd.read_csv(a.csv, dtype={'PetID': str, 'RescuerID': str})
    lookup = pd.read_csv(a.breed_labels)
    require(df, PF_REQUIRED); require(lookup, ['BreedID', 'Type', 'BreedName'])
    if df.PetID.isna().any() or df.PetID.duplicated().any(): raise ValueError('Invalid PetID')
    if lookup.BreedID.duplicated().any(): raise ValueError('Ambiguous breed lookup')
    for c in ['Type', 'Quantity', 'Gender', 'Breed1', 'Breed2', 'AdoptionSpeed']:
        df[c] = pd.to_numeric(df[c], errors='raise')
    if not df.AdoptionSpeed.isin(range(5)).all(): raise ValueError('Unexpected target')
    allowed = set(lookup.BreedID) | {0}
    for c in ['Breed1','Breed2']:
        if not set(df[c]).issubset(allowed): raise ValueError(f'Unknown {c}; wrong lookup version')
        kinds = df[c].map(lookup.set_index('BreedID').Type)
        if (df[c].ne(0) & kinds.ne(df.Type)).any(): raise ValueError(f'Species mismatch in {c}')
    df['breed_name'] = df.Breed1.map(lookup.set_index('BreedID').BreedName)
    df['single_breed_candidate'] = (df.Quantity.eq(1) & df.Gender.isin([1,2]) & df.Breed1.ne(0) & df.Breed2.eq(0))
    # A primary category named Mixed Breed / Domestic Short Hair etc. is retained
    # verbatim for human taxonomy review, not silently treated as a pure breed.
    if a.images:
        paths = sorted(Path(a.images).rglob('*.jpg'))
        photos = pd.DataFrame([{'PetID': q.stem.rsplit('-',1)[0], 'image_path': str(q),
                               'default_photo': q.stem.endswith('-1')} for q in paths],
                             columns=['PetID','image_path','default_photo'])
        photos.to_csv(out/'photo_inventory.csv', index=False)
        df['actual_photo_count'] = df.PetID.map(photos.groupby('PetID').size()).fillna(0).astype(int)
        df['default_photo_present'] = df.PetID.isin(photos.loc[photos.default_photo, 'PetID'])
        df['single_breed_candidate'] &= df.default_photo_present
    else:
        df['actual_photo_count'] = pd.NA; df['default_photo_present'] = pd.NA
    summary = df.groupby(['Type','Breed1','breed_name'], dropna=False).agg(
        total_listings=('PetID','size'), candidate_single_breed_listings=('single_breed_candidate','sum'))
    summary.reset_index().to_csv(out/'breed_coverage.csv', index=False)
    df[['PetID','Breed1','Breed2','breed_name','Quantity','Gender','single_breed_candidate',
        'actual_photo_count','default_photo_present']].to_csv(out/'listing_review.csv', index=False)
    report = {'data_profile': profile(df), 'lookup_rows': len(lookup),
              'class_counts': df.AdoptionSpeed.value_counts().sort_index().to_dict(),
              'single_breed_candidates': int(df.single_breed_candidate.sum()),
              'images_supplied': bool(a.images),
              'status': 'coverage audit only; requires visual label review, image decoding/dedup, rescuer-group split and listing-time provenance review',
              'target_definition': {'0':'same day','1':'1-7 days','2':'8-30 days','3':'31-90 days','4':'not adopted after 100 days'},
              'official_gap_note': 'Official page explicitly states no pets waited between 90 and 100 days; do not invent an exact LOS.'}
    (out/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

if __name__ == '__main__':
    main()
