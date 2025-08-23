---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    custom_cell_magics: kql
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: clamp
    language: python
    name: python3
---

Clustering visualisations
=========================

```python
import cluster_plot
import numpy as np
import pacmap
import plotnine as p9
import polars as pl
import tol_colors
import trimap
import umap

from sklearn import manifold
from matplotlib import pyplot as plt
```

## Load data

```python
embeddings_df = pl.read_parquet(
    "../data/generated/pipeline/embeddings/EleutherAI:pythia-160m-deduped|k=40+layer=7.parquet"
)
embeddings_df
```

```python
clusters_df = pl.read_csv(
    "../data/generated/pipeline/clustering/EleutherAI:pythia-160m-deduped|k=40+layer=7+covariance_type=full|max_iter=400|n_clusters=300|progress=2|weight_concentration_prior=0.1.tsv",
    separator="\t",
)
clusters_df
```

```python
df_all = embeddings_df.join(clusters_df, on=[pl.col("preamble"), pl.col("guess")], validate="1:1")
df = df_all.filter(pl.col("preamble") == "Students were outraged over the administration's new")
df
```

```python
embeddings = df["embedding"].to_numpy()
embeddings
```

```python
clusters = np.unique_inverse(df["cluster_id"].to_numpy()).inverse_indices
n_clusters = clusters.max() + 1
clusters
```

```python
cmap = tol_colors.rainbow_discrete(n_clusters)
cmap
```

```python
labels = df["guess"].to_numpy()
```

## Clustering-agnostic

We vizualize the clusters as colours to get a sense of what happens to them in the projection but the projection methods do not know about them

```python
t_sne = manifold.TSNE(n_components=2, random_state=1871)
X_t_sne = t_sne.fit_transform(embeddings)

x, y = X_t_sne.T
df_tsne = df.with_columns(pl.Series(values=x).alias("x"), pl.Series(values=y).alias("y"))
(
    p9.ggplot(df_tsne, p9.aes(x="x", y="y", color="factor(cluster_id)"))
    # + p9.geom_point()
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors)
)
```

```python
umap_model = umap.UMAP(random_state=1871)
X_umap = umap_model.fit_transform(embeddings)

x, y = X_umap.T
df_tsne = df.with_columns(pl.Series(values=x).alias("x"), pl.Series(values=y).alias("y"))
(
    p9.ggplot(df_tsne, p9.aes(x="x", y="y", color="factor(cluster_id)"))
    # + p9.geom_point()
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors)
)
```

```python
trimap_model = trimap.TRIMAP()
X_trimap = trimap_model.fit_transform(embeddings)

x, y = X_trimap.T
df_tsne = df.with_columns(pl.Series(values=x).alias("x"), pl.Series(values=y).alias("y"))
(
    p9.ggplot(df_tsne, p9.aes(x="x", y="y", color="factor(cluster_id)"))
    # + p9.geom_point()
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors)
)
```

```python
pacmap_model = pacmap.PaCMAP(
    n_components=2, n_neighbors=8, MN_ratio=0.5, FP_ratio=2.0, random_state=1871
)
X_pacmap = pacmap_model.fit_transform(embeddings)

x, y = X_pacmap.T
df_tsne = df.with_columns(pl.Series(values=x).alias("x"), pl.Series(values=y).alias("y"))
(
    p9.ggplot(df_tsne, p9.aes(x="x", y="y", color="factor(cluster_id)"))
    # + p9.geom_point()
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors)
)
```

## Cluster-aware

```python
cluster_plot_model = cluster_plot.ClusterPlot(
    learning_rate=0.5,
    n_iter=32,
    batch_size=1,
    reduce_all_points=True,
    class_to_label={i: f"cluster_{i}" for i in set(clusters)},
    anchors_method="birch",
    show_fig=False,
    save_fig=False,
    save_fig_every=32,
    show_anchors=False,
    show_points=True,
    show_loss_plot=False,
    show_label_level_plots=False,
    k=16,
    random_state=1871,
    magnitude_step=True,
    top_greedy=1,
    alpha=0.1,
    remove_outliers_k=2,
    douglas_peucker_tolerance=0.3,
    smooth_iter=3,
)

X_clusterplot = cluster_plot_model.fit_transform(embeddings, clusters)
cluster_plot_model.cluster_plot()
plt.show()

```
