# AC215-furcast
Harvard AC215 Group 17 project: pet breed identification, adoption risk prediction, and shelter data preprocessing

## 数据工作区 / Data workstream

Sam's data acquisition, preprocessing and leakage checks, updated with real Pawpularity preprocessing on **2026-10-16**.
This repository contains code, field contracts and aggregate audit reports. The separately delivered data packages contain the actual datasets, frozen splits and prepared images. Cloning this repository does **not** download them.

| Source | Task | Verified prior work | In this repository |
|---|---|---|---|
| Oxford-IIIT Pet | Breed classification | 7,349 annotated image IDs, 37 classes; image audit and split manifests | Code and aggregate audit |
| Stanford Dogs | Dog breed classification | 20,580 original images; 20,415 prepared images, 120 classes | Code and aggregate audit |
| PetFinder Adoption Prediction | AdoptionSpeed ordinal classification | 14,993 labeled profiles; conservative cohort 11,553; 72,776 images audited | Code and aggregate audit |
| Austin Animal Center | Observed all-exit length of stay (LOS) | Separate legacy/current snapshots, episode pairing, temporal/animal splits and sparse features | Code and aggregate audits |
| PetFinder Pawpularity | Photo/profile engagement regression | 9,912 labeled originals checked; 7,811 prepared photos after cross-source quarantine; fixed 4,689/1,562/1,560 split | Code, real-run aggregate audit and model recommendations |

Oxford, Stanford, PetFinder Adoption and Austin entries describe earlier snapshot results. Pawpularity was processed on actual files on October 16; see [the completed run and training plan](docs/PAWPULARITY_TRAINING.md). No predictive model has been trained by this data handoff. Frontend and container integration remain separate team workstreams.

## DVC continuation — 2026-10-17

The real October 16 delivery has been independently revalidated (7,811 photos) and versioned with DVC. Raw-data reconstruction is wired and tested with synthetic inputs; this environment could not retrieve the 1 GB raw ZIP. See [DVC reproduction and exact access findings](docs/DVC.md). No shared or paid cloud remote has been selected.

## Start here

Python 3.11+:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -v
python run_data_pipeline.py --help
```

The tests use synthetic fixtures; they do not download data or establish model accuracy.

- [Pawpularity processing results and model recommendations](docs/PAWPULARITY_TRAINING.md)
- [中文交接与分工](docs/HANDOFF_ZH.md)
- [Data access and dataset limits](docs/DATA_SOURCES.md)
- [Run commands and required local files](docs/RUNBOOK.md)
- [Targets, splitting and leakage policy](docs/TARGETS_AND_SPLITS.md)
- [Machine-readable field mapping](field_mapping.json)
- [Current acquisition status](data_status.json)
- [Prior aggregate audit reports](reports/prior_run/README.md)

## Run with the existing starter package

Extract the authorized `FurCast_Data_Starter_2026-10-12.zip` outside this repository. The filename retains its original date; the previously delivered package was updated on October 15. Supply its `raw/` and `results/` directories explicitly:

```bash
python run_data_pipeline.py --data-root /path/to/FurCast_Data_Starter/raw --reference-results /path/to/FurCast_Data_Starter/results --out results_rebuild
```

This rebuilds available Austin/annotation stages; full image stages require additional inputs described in the runbook. Each output directory must be new. Missing optional inputs are recorded in `run_status.json`; missing dependencies of an explicitly requested stage stop execution. The wrapper's fixed Austin dates reproduce the earlier snapshot experiment; use explicit dates with `furcast_data.py austin` for a newly designed experiment.

## Keep targets separate

**AdoptionSpeed ≠ Austin LOS ≠ Pawpularity.** PetFinder measures time categories from listing, Austin measures time from intake to an observed exit (including non-adoption exits), and Pawpularity measures engagement. Never concatenate these as one supervised target or manufacture exact LOS from class midpoints.

Raw data, individual-level outputs, images, fitted transformers, archives and credentials are excluded from Git. Obtain sources through their official access process or the authorized team handoff. Review each dataset's terms; repository code does not grant rights to its data. No project-wide software license has been selected yet.
