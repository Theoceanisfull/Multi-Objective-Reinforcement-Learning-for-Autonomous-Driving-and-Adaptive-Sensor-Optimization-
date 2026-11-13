# Organization and provenance

`MORL4AVSO` is a separate organized copy of the original project. Original files
outside this folder were left in place. Recorded models were not retrained, historical results were preserved, and no
scientific parameters were changed. Relocation checks include test-only smoke evaluations.

## Relocation map

| Original location | Repository location |
| --- | --- |
| `algorithms/` | `algorithms/` |
| `environments/` source/config/tests | `environments/` |
| `experiments/` notebooks/helpers/tests | `experiments/` |
| `experiments/artifacts/dqn_*` | `runs/comparisons/` |
| Earlier SARSA agent/time-completion artifacts | `runs/training/` |
| Sensor realism/contribution/cut-in study artifacts | `runs/sensor_studies/` |
| `environments/artifacts/` | `runs/environment/` |
| `basic/artifacts/` | `runs/tutorial/highway_q_learning/` |
| `documents/` and root milestone PDF | `docs/` |
| Original root README | `docs/project_background.md` |
| `logs/first_test.png` | `docs/images/first_test.png` |

Run IDs, nested evaluations, and replay folders are preserved. Code/notebook output
paths and documentation links now point to the organized locations. Absolute local
workspace prefixes were removed from copied textual metadata and logs. Notebook
kernel metadata uses portable `python3`; saved charts and numerical results remain.

The virtual environment, Python caches, notebook checkpoints, `.DS_Store`, empty
placeholder folders, and temporary Office lock files were excluded. No license or
remote repository was inferred. Recorded runs are intentionally eligible for Git.

The migration checksums are in [data/file_manifest.csv](../data/file_manifest.csv).
CSV data and binary artifacts retain their source bytes. JSON/notebook/text hashes
may change because of path and presentation metadata updates; original experiment
hashes remain historical provenance. Newly generated documentation, catalogs, and
exported training charts are separate from the copied-file manifest.

## Verification

- All **32 existing tests passed** from the organized project: 18 environment
  tests and 14 experiment tests.
- All 10 notebooks passed format validation; Python sources and non-magic
  notebook code cells compiled.
- Local Markdown links resolved.
- The organized folder was copied to a temporary location. Each saved
  1,000-episode policy loaded its configuration and reproduced the recorded
  clear-weather seed-100000 reward, energy, time, and distance.
- Migrated CSVs, models/arrays, images, GIFs, PDFs, and Word documents matched
  their source checksums byte-for-byte.
- The virtual environment and temporary/cache files are excluded by Git.

Tests reused the existing project Python environment; a fresh install and the
GitHub-hosted workflow have not been executed as part of this local preparation.
