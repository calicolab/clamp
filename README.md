CLuster Any Model Package (clamp)
=========================

## Examples

Run these in a virtual environment, make sure you installed clamp (e.g. with `python -m pip install
-U .`).

Manual pipeline:

```bash
clamp predict "EleutherAI/pythia-14m" data/sentences.txt data/generated/pythia_14m-predictions.tsv
clamp embed "EleutherAI/pythia-14m" data/generated/pythia_14m-predictions.tsv data/generated/pythia_14m_last_layer-embeddings.parquet
clamp cluster --covariance-type diag --progress 10 data/generated/pythia_14m_last_layer-embeddings.parquet data/generated/pythia_14m_last_layer-clusters.tsv
```

**Importan**: the `diagonal` covariance prior for the clustering step is very fast (making this a
good smoke test), but generally terrible at finding good clusters. In other words: run this once to
make sure that everything runs, then use full covariance matrices for any serious work.

Pipeline with parameters grid:

```bash
clamp pipeline data/sentences.txt config/config.yaml data/generated/pipeline/
```