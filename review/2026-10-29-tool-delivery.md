# Tool delivery checkpoint — 2026-10-29

The target is a CPU tool other users can clone, install and run on versioned
real data. Agent training is deferred. Economic superiority is not a release
criterion. See [the blueprint](../BLUEPRINT.md) and [quickstart](../docs/quickstart.md).

Local changes:

- Added `python -m rtcb_gym.demo`, with synthetic prices, two causal rules and
  an explicit perfect-foresight oracle. Outputs requests/execution, SoC,
  clipping, proxy ledger, software/code provenance and file checksums.
- Integrated SCED export code from `c86ae8d3ff6ffbf05cf3a4210128e058be8c8064`
  with local changes. Default cache-only operation fails on a missing chunk
  without constructing an API client. Downloads require `--allow-download`.
- Fixed expiry inside a price segment: retain its valid prefix and report
  partial coverage. Added `covered_seconds`; partial bins remain unsuitable
  as full-interval inputs. Original cloud artifacts have not been regenerated.
- Preserved timestamp microseconds in export CSVs, labelled cache mtimes
  accurately, and made reconstruction/settlement caveats explicit.
- Updated README, workflow, workspace guide and product blueprint. Preserved
  prior documents under `archive/`. No GitHub push or release was performed.

Validation: **187 passed, 213 warnings in 15.01s**, Python 3.13.5, from the
Desktop workspace. The warnings are retained, not counted as failures. A wheel
was built, installed into a separate target directory and the demo was run
outside the source tree. The same demo ran in the Desktop workspace; oracle
objective/replay, terminal state and all output checksums passed. Complete
synthetic export-cache coverage was tested with API-client creation forbidden.
No GridStatus API requests were made. Remote CI and an independent user's
clean clone are still unverified.

This is an installation/reproducibility checkpoint, **not completion of the
real-data AS release**. The demo uses synthetic prices, full ECRS awards,
no deployment, discharge-side reserve and five-minute cash-flow proxies.
Actual cloud CSVs/report/cache are still absent; code in the branch does not
contain those git-ignored files. Formal AS settlement, deployment scope,
real-data import/replay tutorial and custom-policy tutorial remain open.

Source-file hashes and checks: [manifest](evidence/tool_delivery_2026-10-29.json).
