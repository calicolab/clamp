CLuster Any Model Package
=========================

## Examples

Run these in a virtual environment, make sure you installed clamp (e.g. with `python -m pip install
-U -r frozen_requirements.lst`).

Manual pipeline:

```bash
clamp predict "EleutherAI/pythia-14m" data/sentences.txt data/generated/pythia_14m-predictions.tsv
clamp embed "EleutherAI/pythia-14m" data/generated/pythia_14m-predictions.tsv data/generated/pythia_14m_last_layer-embeddings.parquet
clamp cluster --covariance-type diag --progress 10 data/generated/pythia_14m_last_layer-embeddings.parquet data/generated/pythia_14m_last_layer-clusters.tsv
```

Pipeline with parameters grid:

```bash
clamp pipeline data/sentences.txt config/config.yaml data/generated/pipeline/
```
