# Data access and actual availability

Status date: 2026-10-15. Counts below refer to the earlier audited snapshots. A fresh download can differ and must be audited again. No dataset bytes are included in Git.

| Dataset | Official source | Data/label role | Access and limitations |
|---|---|---|---|
| Oxford-IIIT Pet | https://www.robots.ox.ac.uk/~vgg/data/pets/ | Images and annotations; 37 breed classes | Earlier review recorded CC BY-SA 4.0 on the page and research-use wording in the archive README. Preserve attribution and review this discrepancy before redistributing images. |
| Stanford Dogs | http://vision.stanford.edu/aditya86/ImageNetDogs/ | Images, bounding boxes and official splits; 120 dog classes | Citation requested; no blanket image redistribution license established in the earlier review. ImageNet/source-image rights remain relevant. |
| PetFinder Adoption Prediction | https://www.kaggle.com/competitions/petfinder-adoption-prediction/data | Listing fields, photos, optional metadata/sentiment JSON; AdoptionSpeed 0–4 | Individual authorized Kaggle access and applicable competition rules: https://www.kaggle.com/competitions/petfinder-adoption-prediction/rules . Do not redistribute raw files publicly. |
| Austin legacy | https://data.austintexas.gov/resource/wter-evkm.json and https://data.austintexas.gov/resource/9t4d-g238.json | Intake and outcome events, paired into episodes | Earlier metadata identified Public Domain; repeated admissions require event pairing. |
| Austin current | https://data.austintexas.gov/resource/pyqf-r2dc.json and https://data.austintexas.gov/resource/gsvs-ypi7.json | New-system intake and outcome events | License metadata was unspecified in the earlier snapshot. Do not assume old/new animal IDs correspond. |
| PetFinder Pawpularity | https://www.kaggle.com/competitions/petfinder-pawpularity-score/data | `Id`, 12 binary visual attributes, `Pawpularity` 1–100 | Separate rules: https://www.kaggle.com/competitions/petfinder-pawpularity-score/rules . Team-supplied real ZIP verified and preprocessed on 2026-10-16; original/derived images remain outside Git. |

The access/rights observations above record the existing source review, not a new legal review or renewed download in this repository setup.

## Existing team files

Request these from Sam through the authorized handoff. Chat-session download links are deliberately not published in the repository.

- `FurCast_Data_Starter_2026-10-12.zip`: v0.4 code plus actual Austin snapshots, Oxford annotations, PetFinder CSVs/lookups, row-level audits, fixed split manifests and sparse arrays. **Contains data; do not upload this ZIP to GitHub.**
- `PetFinder_Default_Photos_224.zip`: 11,318 processed default photos; join through the starter's prepared image manifest.
- `Stanford_Dogs_224_Preprocessed.zip`: 20,415 processed dog images; use the starter's fixed splits.
- `Oxford_Pet_Images_Part_1.zip` and `Oxford_Pet_Images_Part_2.zip`: reconstruct the original Oxford image archive with `restore_oxford_images.py`.
- `PetFinder_Original_Part_01.bin` through `PetFinder_Original_Part_05.bin`: exact pieces of the official archive, **not independent ZIP files**. Restore and verify with `restore_petfinder_archive.py`; archive/part hashes are in `evidence/petfinder_original_parts.json`.
- `Pawpularity_Handoff_2026-10-15.md`: diagnostic and schema handoff only; no real Pawpularity data.

Before receiving competition data, each recipient must satisfy the applicable access and sharing terms. Downloading under one person's account does not automatically grant every collaborator access.

## Acquisition scripts

```bash
python fetch_public.py --out snapshots/austin_new
python fetch_stanford_dogs.py --out snapshots/stanford_new
```

Austin acquisition checks counts and update metadata around the snapshot, records hashes, and rejects changing/truncated retrievals. Stanford acquisition records sizes and hashes, but its official HTTP transport is not a publisher-signed integrity guarantee. Oxford archives can be obtained from the official page; retain a checksum and retrieval timestamp with the originals.

Kaggle files require a separately authenticated local Kaggle CLI/session and acceptance of the relevant rules. Do not put API keys in code, chat, committed `.env` files or command arguments. No credential setup or Kaggle retry is performed by this GitHub initialization.

## Pawpularity acquisition and preprocessing

Earlier Kaggle downloads were blocked by authentication. On 2026-10-16 the team supplied an original ZIP through Drive; its 1,030,728,211 bytes were downloaded, and its SHA-256 was recorded as `6b57735ba5e6041c79b73833fff267db54e425134efbab42728e3025c712153c`.

Local measurements: 9,912 training records, 14 columns, 9,912 training photos, and 8 unlabeled example records/photos. All 9,920 source images decoded. These counts are from actual files, not website metadata. Example test data and sample-submission scores are not evaluation ground truth and are excluded from modeling.

Preprocessing compared 100,705 prior-source fingerprints with all Pawpularity photos. Conservative cross-source quarantine removed 2,101 records from this modeling cohort, leaving 7,811 images with fixed group-disjoint splits. Candidate lists are retained for review; near matches are not confirmed same-animal identities. No rows were deleted from the original ZIP.

The older October 15 handoff remains a historical diagnostic document. Current results and model recommendations: [PAWPULARITY_TRAINING.md](PAWPULARITY_TRAINING.md). Obtaining files does not grant redistribution rights; use the official competition terms and authorized team access.
