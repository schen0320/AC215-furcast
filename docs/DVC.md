# Pawpularity data versions and reproduction

Updated 2026-10-17. This change builds on PR #1 (`27d8e87`), which already
processed real Pawpularity data on October 16. Do not restart from the older
main-branch claim that only synthetic data have been processed.

## What is verified

- The existing processed package was obtained and every checksum in its
  `SHA256SUMS.txt` was checked. The independent validator then decoded and
  hashed all 7,811 prepared photos. Fixed splits remain 4,689 train / 1,562
  validation / 1,560 held-out test. Test performance is not evaluated.
- `artifacts/delivery.dvc` versions that real delivery, and
  `artifacts/reference.dvc` versions its two required prior-source manifests.
- `dvc.lock` contains the **actually executed** `validate_delivery` stage.
  It does not pretend that raw-data reconstruction ran in this environment.
- The real delivery was also pushed to a temporary local remote and restored
  into a fresh empty-cache clone. Independent validation reproduced the same
  report for all 7,811 photos; temporary storage was cleaned up.
- An isolated synthetic integration test ran extraction, preparation,
  validation, no-op and forced reruns, local push, empty-cache clone/pull,
  and independent output validation. Forced runs produced identical locks.
  This proves the DVC wiring, not acquisition or reconstruction of real raw data.
- No cloud remote was created, no paid service selected, and no raw data,
  prepared photos, row-level CSVs, caches or credentials were uploaded to GitHub.

## Environment

Use Python **3.12.14**, matching the prior real run. The complete tested
Linux environment, including DVC 3.66.1 and image-processing packages, is pinned:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dvc.lock.txt
python -m unittest discover -v
```

Run from the repository root. Do not use Python `-O`. All pipeline commands
use `python` from your activated environment. Other Python/platform/codec
versions must be revalidated; package pins alone do not prove identical JPEG
bytes across every platform.

## Restore and verify the existing real release (works without the raw ZIP)

Obtain the authorized `FurCast_Pawpularity_Processed_2026-10-16.zip` from Sam.
SHA-256: `7eff738a725af6e6bf10a45db6ebacf0fbee2e8ebdce90e8893c1f2025bdcdb0`.
Size: 186,240,803 bytes. Do not publicly redistribute the package.

```bash
python pawpularity_inputs.py import-delivery --package /path/to/FurCast_Pawpularity_Processed_2026-10-16.zip
python -c "import shutil; shutil.copytree('artifacts/delivery/reference', 'artifacts/reference')"
dvc commit artifacts/delivery.dvc artifacts/reference.dvc
dvc repro validate_delivery
dvc status artifacts/delivery.dvc artifacts/reference.dvc validate_delivery
```

Use a fresh checkout, or restore existing cached data with `dvc checkout`
instead of overwriting an existing directory. The importer verifies the
frozen package hash, safe ZIP paths, CRCs, checksums and both reference hashes.
`dvc commit` fills your local cache using the existing pointer versions;
review `git diff` afterward and stop if any pointer unexpectedly changes.
The validator writes to `artifacts/validation/delivery.json`, outside its
input directory, so it does not invalidate the delivery dependency.

## Acquire real raw input

Official sources:
- [Competition data](https://www.kaggle.com/competitions/petfinder-pawpularity-score/data)
- [Competition rules](https://www.kaggle.com/competitions/petfinder-pawpularity-score/rules)
- [Official CLI authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication)
- [Official download commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md)

The browser session on October 17 showed Sam Chen signed in and the competition
rules already accepted. No new agreement was accepted. The rules allow
non-commercial academic research/education and restrict redistribution to
eligible participants who have accepted the rules. A Harvard group membership
alone does not establish Kaggle authorization. Each recipient must satisfy the
rules before data sharing. The 8 example test images/rows are random examples,
not the hidden evaluation set.

The current execution environment has no Kaggle CLI or credentials. Install
Kaggle separately (so acquisition tooling does not alter the processing pins),
then use its official OAuth login or a private token file. Never put a token
in Git, chat, a command argument, a report or a DVC remote URL.

```bash
uv tool install kaggle
kaggle auth login
kaggle competitions files petfinder-pawpularity-score
kaggle competitions download petfinder-pawpularity-score -p /private/downloads
```

Alternatively, use Kaggle's Data -> Download all files -> Download as zip,
or the already-authorized team raw archive. Preserve the ZIP for versioning.
The frozen local source hash is in `config/pawpularity-source.json`; it comes
from the October 16 verified archive, **not a Kaggle-published checksum**.
A changed container hash must be investigated rather than bypassed. If Kaggle
repackages the same content, compare full member hashes in a separate review
and deliberately version the updated contract.

Observed October 17 access boundaries:
- Drive metadata allowed access to the existing 1,030,728,211-byte raw ZIP.
  The Drive download API rejected it with HTTP 413 because its maximum is
  268,435,456 bytes. This is a connector size limitation, not a Kaggle auth error.
- Kaggle browser ZIP download and Drive browser ZIP download both returned
  `Download ... failed`. The page did not supply a cause. The session was
  authenticated; no evidence establishes an authentication or bot-detection cause.
- Browser policy blocked inspection of `chrome://downloads/`, so no lower-level
  browser diagnostic or workaround was attempted.
- Therefore the real raw ZIP was **not downloaded into this run's workspace**.
  Real raw extraction/preparation stages remain unexecuted here.

The existing Drive file UI also showed **Anyone with the link** access. It was
not used as a DVC remote, and its sharing settings were not changed. Before
using Drive for team data synchronization, review this access against the
competition's recipient restrictions; do not assume link access grants a license.

## Rebuild from the frozen original data

Place the verified original ZIP at `artifacts/inputs/pawpularity.zip` and restore
the two reference files using the preceding delivery workflow. Then:

```bash
# Fails before extraction if bytes/hash/schema/IDs/files do not match.
# DVC itself invokes the extraction check; this command creates the raw pointer.
dvc add artifacts/inputs/pawpularity.zip
dvc repro validate_pawpularity
dvc status artifacts/inputs/pawpularity.zip.dvc artifacts/reference.dvc extract_pawpularity prepare_pawpularity validate_pawpularity
git diff -- dvc.lock artifacts/inputs/pawpularity.zip.dvc
```

`extract_pawpularity` verifies frozen archive hash/size, paths, CRCs,
train.csv hash, row counts, original schemas and exact image/ID membership.
`prepare_pawpularity` rechecks raw/reference contracts, then reuses the existing
pipeline unchanged: seed 215, pHash distance <=6, transitive cross-source
quarantine, component-isolated 60/20/20 splits, and deterministic 300-pixel
JPEG export. Code, pins, reference files, raw files, target specification and
image/worker settings are explicit DVC dependencies/parameters.
`validate_pawpularity` independently verifies saved photos, labels, split and
duplicate consistency. DVC generates its additional lock entries only on success.

For this frozen release, compare the rebuilt photo ZIP hash to
`5fe966d538969f897c31a6098f552810db6f58434be5fb5869fc1682afdfe923` and all
split/manifest hashes against the October 16 delivery. Do not report full
reproduction solely because the row counts match. Never copy a synthetic
lock into the release. Failed runs must not be marked completed.

## Storage without a paid cloud decision

DVC uses a local `.dvc/cache` immediately; a cloud account is unnecessary.
For an existing private disk or mounted team storage with suitable ACLs:

```bash
dvc remote add --local -d team /absolute/private/furcast-dvc
# Push only the versioned, authorized delivery that actually exists:
dvc push artifacts/delivery.dvc artifacts/reference.dvc validate_delivery
```

On another machine, check out the same Git commit, configure its corresponding
private mounted path with the same command, then:

```bash
dvc pull artifacts/delivery.dvc artifacts/reference.dvc validate_delivery
dvc repro validate_delivery
dvc status artifacts/delivery.dvc artifacts/reference.dvc validate_delivery
```

`--local` writes `.dvc/config.local`, excluded from Git. A local path is NOT a
shared service by itself. The mount must actually exist and recipients must
already have appropriate access. This session tested a temporary local remote;
**no persistent shared team remote is configured**. Existing ZIP delivery plus
hash-checked import remains the no-cloud reproduction route. GCS/S3 or another
paid service requires a separate storage decision; creating a remote does not
change data permissions. No automatic upload is embedded in `dvc repro`.

Once real raw rebuilding succeeds, push its raw pointer plus the three raw
pipeline stage targets only to an approved private remote, then commit the real
`dvc.lock` and `.dvc` pointers in the same reviewed Git change. A Git checkout
identifies code and data hashes; `dvc pull` supplies bytes; `dvc repro` rebuilds.
Review changes to both Git and DVC before declaring a new release.

## Independent integration verification

```bash
python verify_dvc_roundtrip.py
```

This uses temporary synthetic inputs and a temporary local remote, including a
fresh consumer checkout with no cache. It checks deterministic forced rebuilds,
changed-reference rejection and successful restored-output validation, then
cleans up only its own temporary directories. It does not consume Kaggle
credentials, upload real data, or change the real-release lock.
