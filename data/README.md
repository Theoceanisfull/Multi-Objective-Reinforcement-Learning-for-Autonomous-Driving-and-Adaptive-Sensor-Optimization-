# Data catalog

[Catalog CSV](catalog.csv) lists the files under the recorded run collections, including each file's relative path, size, and SHA-256 digest. [Run index](../runs/index.csv) lists the experiment and replay directories. The source measurements and metadata live beside their experiment's models and configuration under [`runs/`](../runs/).

Regenerate the indexes from the repository root:

```bash
python scripts/catalog_runs.py
```

The catalog excludes `runs/local/`, where new runs and local training state are stored.
