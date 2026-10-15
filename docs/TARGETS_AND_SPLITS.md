# Target definitions and leakage boundaries

## Independent targets

| Source | Target | Interpretation |
|---|---|---|
| PetFinder | AdoptionSpeed 0 | Adopted on listing day |
| PetFinder | AdoptionSpeed 1 | Adopted 1–7 days after listing |
| PetFinder | AdoptionSpeed 2 | Adopted 8–30 days after listing |
| PetFinder | AdoptionSpeed 3 | Adopted 31–90 days after listing |
| PetFinder | AdoptionSpeed 4 | Not adopted after 100 days; not an exact duration or permanent outcome |
| Austin | Observed all-exit LOS | Actual interval from matched intake to exit; exit may be adoption, transfer, death or another event |
| Pawpularity | Engagement score | Normalized photo/profile engagement, not objective beauty or causal adoption benefit |
| Oxford / Stanford | Dataset-specific breed class | Different taxonomies and coverage; class numbers must not be concatenated blindly |

PetFinder's optional `slow_listing_risk_label = AdoptionSpeed in {3,4}` is a project experiment, not a shared LOS target. Multi-pet listings describe adoption of all pets; the first baseline restricts to `Quantity=1`. Do not convert class intervals into invented exact days.

Austin's `long_stay_30d` is derived from observed exit duration. The delivered cohort is conditional on confidently paired observed exits. Unmatched records are unresolved, **not confirmed right-censored**. This cohort cannot establish prospective adoption probability for all arrivals. An adoption-specific survival model requires a separately defined risk set, competing exits and verified observation coverage.

## PetFinder inputs and split

The production entry for the delivered cohort is `prepare_petfinder_real.py`. `furcast_data.py petfinder` is a provisional CSV-only adapter and does not reproduce that cohort.

- Features: `Type`, `Breed1`, `Breed2`, `Gender`, `Color1`, `Color2`, `Color3`, `MaturitySize`, `FurLength`, `State`, `Age`. Age is months; categorical codes are not continuous quantities.
- Excluded from X: labels, identifiers, `Name`, `Quantity`, `Description`, `PhotoAmt`, `VideoAmt`, `Vaccinated`, `Dewormed`, `Sterilized`, `Health`, `Fee`. Mutable fields and photo updates lack first-listing provenance. Even retained fields have not been independently verified as original listing snapshots.
- Build transitive components from rescuer ID, image duplicates and exact normalized descriptions of at least 40 characters **before** cohort exclusions. Unlabeled official test profiles can bridge identities but never fit transformers or enter scores.
- Freeze five StratifiedGroupKFold folds at seed 215: fold 0 internal test, fold 1 validation, others train. Propagate splits to every photo, crop, augmentation and embedding.
- No listing dates are available: this is group validation, not temporal validation. Fit imputation/encoding/scaling on train only. Select thresholds on validation; reserve internal test for final assessment.
- Prior cohort: 6,931 / 2,311 / 2,311 records; 290 encoded features. Largest connected group contributes 976 records to test (~42.2%); account for group clustering in uncertainty and do not optimize random seeds for results.

## Austin inputs and split

Use only intake-side attributes: species, intake type/condition, intake sex, breed/color at intake, approximate intake age and intake month. Outcome timestamps, outcome types, days in shelter, outcome-side age/sex and record IDs are not prediction inputs. IDs are for pairing/grouping; timestamps establish labels and observation boundaries.

Legacy records are paired within an animal's successive intake windows. Current records use the intake timestamp with animal ID. Ambiguous timelines, multiple outcomes and impossible orderings are quarantined. Do not perform a Cartesian merge on animal ID.

Separate legacy/current experiments use temporal windows. Exclude animals spanning windows, and require labels to have become available before the next training/evaluation boundary. Transformations fit on train only. Default reproduction boundaries:

| Era | Validation starts | Test starts | Snapshot label cutoff |
|---|---|---|---|
| Legacy | 2023-01-01 00:00 −06:00 | 2024-01-01 00:00 −06:00 | 2025-05-05 00:00 −05:00 |
| Current | 2026-01-01 00:00 −06:00 | 2026-07-01 00:00 −05:00 | 2026-10-11 19:00 −05:00 |

These reproduce the prior experiment. A fresh data snapshot needs a deliberate cutoff design. Date of birth in the current source is used to derive age; later correction history is unknown.

## Images and Pawpularity

SHA-256 checks byte/pixel duplicates; 64-bit pHash Hamming distance ≤6 generates near-duplicate candidates. Components are conservatively isolated; a candidate is not proof of the same animal, and different views/crops can escape detection. Human review remains open.

For combined Oxford/Stanford work use `joint_duplicate_split_guard.csv`; it removes 31 additional Oxford development images in the prior run. Stanford's retained test is 8,550 images, not the full official 8,580. ImageNet pretraining overlap has not been ruled out. Review taxonomy before training one combined breed classifier.

Pawpularity requires existing breed and PetFinder fingerprint manifests. Quarantine entire new Paw components connected to prior-source duplicates, preserving prior frozen holdouts. Split remaining components approximately 60/20/20 with seed 215. Scores do not select the seed. There are no reliable rescuer/time/animal IDs to support stronger independence claims.

Pawpularity's 12 manual visual attributes are auxiliary labels, not automatically available features for a new photo. Any downstream use of learned attribute predictions needs group-isolated out-of-fold predictions for training and train-only fitted models for validation/test. Image inputs and regression labels are exported separately. Real Pawpularity execution is still pending.

## Evaluation planning

No model scores are claimed. Agree on success thresholds before final testing. Suggested reports: breed macro-F1/per-class recall; AdoptionSpeed QWK/macro-F1; separately defined slow-listing PR-AUC and calibration; observed LOS MAE with cohort limitations; Pawpularity RMSE/MAE against a train-mean baseline. Engagement associations alone cannot validate an intervention feedback loop; that requires later prospective measurement.
