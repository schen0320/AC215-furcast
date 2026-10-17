# Local preprocessing runbook

Run commands from the repository root with Python 3.11+ and `requirements.txt` installed. Use new output paths for each run. Examples use POSIX line continuations; on PowerShell put a command on one line or use PowerShell continuation syntax. Never run Python with `-O`: the inherited image/split scripts contain correctness assertions.

## Existing data package

Extract the authorized starter ZIP outside the Git checkout. It provides:

- `raw/{wter-evkm,9t4d-g238,pyqf-r2dc,gsvs-ypi7}.json`: frozen Austin snapshots.
- `raw/oxford_annotations.tar.gz`: official Oxford annotations.
- `results/oxford/image_splits.csv`: frozen Oxford image audit/splits.
- `results/stanford_dogs/joint_duplicate_split_guard.csv`: joint breed-source fingerprints and guard.
- `results/petfinder_images/image_manifest.csv`: PetFinder image fingerprints.
- `results/petfinder_adoption/` and `results/austin_*_arrays/`: aligned sparse matrices/labels and train-fitted transformers.

Only load serialized transformers from your trusted, verified team package. `reports/prior_run/` in Git contains aggregate reports and cannot substitute for row-level manifests.

## Restore source archives

```bash
python restore_petfinder_archive.py --parts-dir /path/to/authorized_parts --out snapshots/petfinder.zip
python -m zipfile -e snapshots/petfinder.zip snapshots/petfinder_original
python restore_oxford_images.py --parts /path/to/authorized_parts --out snapshots/oxford_images.tar.gz
```

PetFinder expects the official extraction layout: `train/train.csv`, `test/test.csv`, `train_images/`, `test_images/`, `breed_labels.csv`, `color_labels.csv`, `state_labels.csv`. Optional metadata/sentiment directories are audited if present. Do not silently substitute another competition or image-expanded dataset.

## Rebuild Austin and Oxford annotation stages

```bash
python run_data_pipeline.py \
  --data-root /path/to/FurCast_Data_Starter/raw \
  --reference-results /path/to/FurCast_Data_Starter/results \
  --out results_rebuild
```

The wrapper uses fixed prior-snapshot cutoffs. To specify different observation/split dates, use the explicit Austin entry point, then export train-fitted arrays:

```bash
python furcast_data.py austin --era current \
  --intakes /path/to/pyqf-r2dc.json --outcomes /path/to/gsvs-ypi7.json \
  --cutoff 2026-10-11T19:00:00-05:00 \
  --validation-start 2026-01-01T00:00:00-06:00 \
  --test-start 2026-07-01T00:00:00-05:00 --out results_current
python export_austin_arrays.py --input results_current --out results_current_arrays
```

The dates above reproduce the prior run; do not present a refreshed snapshot as the same experiment without comparing source hashes and coverage.

## Breed image stages

```bash
python audit_oxford_images.py --images snapshots/oxford_images.tar.gz \
  --annotations /path/to/FurCast_Data_Starter/raw/oxford_annotations.tar.gz \
  --out results_oxford_new
python fetch_stanford_dogs.py --out snapshots/stanford_new
python prepare_stanford_dogs.py --raw snapshots/stanford_new \
  --oxford-manifest results_oxford_new/image_splits.csv \
  --out results_stanford_new --export-zip outputs/Stanford_Dogs_224.zip
```

Create the `outputs/` parent directory before direct image export commands. Stanford requires `images.tar`, `lists.tar`, `annotation.tar` and `download_manifest.json` as produced by its fetcher. Keep source taxonomies separate until reviewed. Use the newly generated joint guard for combined training, not just Oxford's earlier split file.

## PetFinder full pipeline using existing breed fingerprints

```bash
python run_data_pipeline.py --skip-austin \
  --data-root /path/to/empty_or_source_data_directory \
  --reference-results /path/to/FurCast_Data_Starter/results \
  --petfinder-raw snapshots/petfinder_original \
  --out results_petfinder_new
```

This runs the image audit, cross-source candidate search and conservative listing preparation. If rebuilding breed images in the same wrapper invocation, additionally supply `--oxford-images` and `--stanford-raw`; it will use the new Stanford joint manifest. Missing required cross-source fingerprints are an error, never permission to skip leakage checks.

Results are under `results_petfinder_new/petfinder_adoption/`; the photo ZIP is beside that directory. Matrices contain only allowed features. For a baseline:

```python
from pathlib import Path
import pandas as pd
from scipy.sparse import load_npz

root = Path('/path/to/FurCast_Data_Starter/results/petfinder_adoption')
X = load_npz(root / 'X_train.npz')
labels = pd.read_csv(root / 'labels_train.csv')
assert len(labels) == X.shape[0]
assert labels['matrix_row'].tolist() == list(range(X.shape[0]))
y = labels['AdoptionSpeed'].astype(int)
```

Do not append every column from the labels/audit CSVs to X. The ID, split, target, duplicate group and outcome-review fields are not ordinary model features.

## Pawpularity — real preprocessing completed 2026-10-16

After obtaining authorized real data, extract it to a directory containing `train.csv` and `train/{Id}.jpg`. Record archive hash and acquisition time, then:

```bash
python run_data_pipeline.py --skip-austin \
  --data-root /path/to/empty_or_source_data_directory \
  --reference-results /path/to/FurCast_Data_Starter/results \
  --pawpularity-raw /path/to/pawpularity_original \
  --out results_pawpularity_new
```

Alternatively run only Pawpularity with explicit fingerprint inputs:

```bash
python prepare_pawpularity.py --raw /path/to/pawpularity_original \
  --external-manifest /path/to/FurCast_Data_Starter/results/stanford_dogs/joint_duplicate_split_guard.csv \
  --external-manifest /path/to/FurCast_Data_Starter/results/petfinder_images/image_manifest.csv \
  --out results_pawpularity_direct --export-images outputs/Pawpularity_Photos_300.zip
```

Expected outputs include separate image inputs, target labels, auxiliary visual labels, duplicate candidates, review tables, audit and prepared photos. Missing inputs must not be replaced with test fixtures. Official synthetic example test data are excluded from modeling. The direct command was executed on real files on 2026-10-16, using 4 workers and 300-pixel images. It produced 7,811 eligible images; see `reports/pawpularity_2026-10-16/audit.json`. Validate saved outputs with:

```bash
python validate_pawpularity_outputs.py --out results_pawpularity_direct --photos outputs/Pawpularity_Photos_300.zip
```

Use `requirements-pawpularity.lock.txt` for the exact preprocessing package versions of that run.

## Checks and run status

```bash
python -m unittest discover -v
python -m compileall -q .
```

The suite covers episode pairing, future outcomes, transfer/adoption distinctions, unresolved episodes, group isolation, feature exclusions, image duplicate radius search, Pawpularity synthetic export/input checks and wrapper path dependencies. It does not certify model performance or eliminate unknown timestamp/identity risks.

The wrapper writes `run_status.json` after execution begins, including partial/failed stages. Input preflight failures exit before creating an output directory. With no data, it can produce a status report with zero completed stages; this is not a successful dataset build. Never infer acquisition from a zero exit code alone. Keep failed outputs for diagnosis and choose a new directory for retries.

## DVC-managed Pawpularity reproduction

Use [docs/DVC.md](DVC.md) for the pinned environment, verified delivery import, raw-data preflight, DVC pipeline and local remote roundtrip. `dvc repro validate_delivery` revalidates existing real outputs; `dvc repro validate_pawpularity` requires the original ZIP and rebuilds from raw data. These are separate evidence claims.
