# Data catalog

Data stays beside its matching configuration, model, and results under `runs/`.
This prevents mixing files from different tasks or training settings.

- [catalog.csv](catalog.csv): every recorded run file, with collection, run ID,
  file type, repository-relative path, size, and SHA-256 checksum.
- [Run index](../runs/index.csv): experiment and nested replay/evaluation folders.
- [file_manifest.csv](file_manifest.csv): every file copied from the original
  workspace, its old relative path, new repository path, and source/copy checksums.
  `text_updated=true` identifies path or notebook metadata updates.

The copy manifest covers migrated source files; newly authored repository docs,
indexes, and exported charts are not migration inputs. Historical metadata hashes
remain the recorded experiment's hashes, not a claim that the organized source
was used to produce the original results. No training was repeated during this
organization step.

All migrated CSV data, model/array files, plots, GIFs, and project document binaries
are retained byte-for-byte. Notebook charts and numerical outputs are retained;
local paths and kernel metadata were made portable.

After adding runs, rebuild the searchable catalogs:

```bash
python scripts/catalog_runs.py
```
