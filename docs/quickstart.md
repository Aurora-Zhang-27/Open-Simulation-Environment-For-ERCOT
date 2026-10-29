# Run RTCB-Gym on a CPU

The development goal is a reusable storage-strategy tool. No agent training,
GPU or market-data key is required for the installation example below.
The new commands are local development changes until included in a release.

## Install and run the offline example

From the repository root, with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m rtcb_gym.demo --out results/demo
```

On Windows, activate with `.venv\Scripts\activate` instead. Installing Python
dependencies may need network access; running the example needs no API access.
Do not install the optional `data` extra unless you intend to download data.

The example generates 72 synthetic five-minute price rows. The first 12 rows
provide history; three policies each execute the remaining 60 intervals with
a 1 MW / 2 MWh battery, 85% round-trip efficiency, $2/MWh degradation,
initial SoC 50% and terminal minimum SoC 50%.

- `idle`: request zero energy and zero ECRS.
- `lagged_price_rule`: charge below $20/MWh, discharge above $60/MWh, using
  the latest observable energy price. Request ECRS whenever not charging.
  Shared power/inventory limits can clip those requests; the trace retains both.
- `perfect_foresight`: an optimizer with future prices, used to check the
  model's bound. It is not a deployable forecast strategy.

Prices are assumed visible five minutes after interval end. The model reserves
ECRS for one hour, excludes ECRS while charging, assumes full awards and has no
AS deployment or market clearing. These are explicit reduced-model choices.
The cash-flow calculation is a **five-minute LMP/MCPC proxy**, not a validated
ERCOT settlement ledger. See [the model](ecrs_gate.md).

## Inspect the outputs

| File | Purpose |
|---|---|
| `input_prices.csv` | Generated prices and assumed availability times |
| `trace.csv` | Each policy's requests, execution, SoC, clipping and proxy ledger |
| `summary.json` | Per-policy totals and oracle-replay check |
| `manifest.json` | Battery, rule, time window, assumptions, software and code hashes |
| `SHA256SUMS` | Hashes of the four generated files |

The command exits unsuccessfully if oracle replay disagrees with its optimizer
or a policy exceeds the bound. Compare runs using the same inputs and manifest,
with numerical tolerances: solver versions may choose different equally optimal
actions, and file hashes need not match across software versions. Re-running
with the same output directory replaces these generated files.

The existing unit suite and this example run in GitHub's configured Python
3.10/3.12 matrix when the workflow is published. Local test success is separate
from an actual remote CI run.

## Use real data

The offline example does not meet the real-data release criterion. Current
energy-only workflows are documented in [the README](../README.md#get-the-data).
The experimental public-data pilots are described in [the gate documentation](ecrs_gate.md).
They are exploratory proxy results, not official AS settlement validation.

The SCED export implementation from commit `c86ae8d` is included locally with
an offline default and a carry-boundary fix. Code does not contain the cloud's
CSV/report bundle. To review that bundle, first copy the existing export
folder, including `SHA256SUMS`, `REPORT.md`, `report.json`, energy CSVs, ECRS CSV,
raw runs and gap details, to `data/exports/<bundle-name>/`. This does not require
a new market-data download. Keep the files private until redistribution rights
are established.

To regenerate exports from an existing **raw parquet cache**, preserve its
`data/raw/gridstatusio/<dataset>/<key>/<UTC-range>.parquet` paths, then run:

```bash
python scripts/export_5min.py --out data/exports/local_review
```

This is cache-only. A missing cache chunk raises an error before constructing
an API client. It does not consume API quota. The filenames include exact UTC
request bounds; having a different cached date range is not a cache hit.
The existing export CSVs alone are not a replacement for raw parquet caches.

For an intentional fresh download, install `.[data]`, configure your own key
as described in the README, and explicitly add `--allow-download`. Do not
redownload a full year merely to obtain an already-generated cloud bundle.

Imported/exported values retain their statuses. `partial` values average only
the covered part; never treat them as full intervals. `imputed` values are
estimates. `carried` depends on the supplied observed SCED clock. The export
script's SPP comparison includes fully covered imputed/carried rows and compares
only overlapping available intervals. A matching price average does not prove
that multiplying variable five-minute power by those prices reproduces the
actual settlement cash flow.

Before a real-data AS release, complete the bundle audit, formal settlement
mapping, explicit data-availability assumptions and a documented replay/import
entry point. Those remain open in [the blueprint](../BLUEPRINT.md).
