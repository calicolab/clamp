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
import openTSNE as tsne
import pacmap
import plotnine as p9
import polars as pl
import tol_colors
import trimap
import umap

from clamp.utils import resp_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
df = embeddings_df.join(
    clusters_df, on=[pl.col("preamble"), pl.col("guess")], validate="1:1"
).with_row_index(name="_row_index")
df
```

```python
embeddings = df["embedding"].to_numpy()
embeddings
```

```python
clusters = np.unique_inverse(df["cluster_id"].to_numpy()).inverse_indices
clusters
```

Just a single sentence

```python
df_sent = df.filter(pl.col("preamble") == "Students were outraged over the administration's new")
df_sent
```

```python
sent_row_idx = df_sent["_row_index"].to_numpy()
sent_row_idx
```

```python
n_clusters_sent = np.unique_inverse(df_sent["cluster_id"].to_numpy()).inverse_indices.max() + 1
cmap = tol_colors.rainbow_discrete(n_clusters_sent)
cmap
```

## Clustering-agnostic

We vizualize the clusters as colours to get a sense of what happens to them in the projection but the projection methods do not know about them


### t-SNE

```python
t_sne = tsne.TSNE(
    n_components=2,
    random_state=1871,
    verbose=True,
).fit(embeddings)

df = df.with_columns(
    pl.Series(values=t_sne[:, 0]).alias("x_tsne"), pl.Series(values=t_sne[:, 1]).alias("y_tsne")
)
(
    p9.ggplot(df, p9.aes(x="x_tsne", y="y_tsne", color="cluster_id"))
    + p9.geom_point()
    + p9.theme_bw()
)

```

```python
(
    p9.ggplot(df[sent_row_idx], p9.aes(x="x_tsne", y="y_tsne", color="factor(cluster_id)"))
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors, name="Cluster ID")
)
```

### UMAP

```python
umap_model = umap.UMAP(random_state=1871, verbose=True)
X_umap = umap_model.fit_transform(embeddings)

df = df.with_columns(
    pl.Series(values=X_umap[:, 0]).alias("x_umap"), pl.Series(values=X_umap[:, 1]).alias("y_umap")
)
(
    p9.ggplot(df, p9.aes(x="x_umap", y="y_umap", color="cluster_id"))
    + p9.geom_point()
    + p9.theme_bw()
)
```

```python
(
    p9.ggplot(df[sent_row_idx], p9.aes(x="x_umap", y="y_umap", color="factor(cluster_id)"))
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors, name="Cluster ID")
)
```

### Trimap

```python
trimap_model = trimap.TRIMAP(verbose=True)
X_trimap = trimap_model.fit_transform(embeddings)

df = df.with_columns(
    pl.Series(values=X_trimap[:, 0]).alias("x_trimap"),
    pl.Series(values=X_trimap[:, 1]).alias("y_trimap"),
)
(
    p9.ggplot(df, p9.aes(x="x_trimap", y="y_trimap", color="cluster_id"))
    + p9.geom_point()
    + p9.theme_bw()
)

```

```python
(
    p9.ggplot(df[sent_row_idx], p9.aes(x="x_trimap", y="y_trimap", color="factor(cluster_id)"))
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors, name="Cluster ID")
)

```

### PaCMAP

```python
pacmap_model = pacmap.PaCMAP(
    n_components=2, n_neighbors=8, MN_ratio=0.5, FP_ratio=2.0, random_state=1871, verbose=True
)
X_pacmap = pacmap_model.fit_transform(embeddings)

df = df.with_columns(
    pl.Series(values=X_pacmap[:, 0]).alias("x_pacmap"),
    pl.Series(values=X_pacmap[:, 1]).alias("y_pacmap"),
)
(
    p9.ggplot(df, p9.aes(x="x_pacmap", y="y_pacmap", color="cluster_id"))
    + p9.geom_point()
    + p9.theme_bw()
)

```

```python
(
    p9.ggplot(df[sent_row_idx], p9.aes(x="x_pacmap", y="y_pacmap", color="factor(cluster_id)"))
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors, name="Cluster ID")
)

```

## Cluster-aware


### Supervised UMAP

```python
umap_super_model = umap.UMAP(random_state=1871, verbose=True)
X_umap_super = umap_super_model.fit_transform(embeddings, y=clusters)

df = df.with_columns(
    pl.Series(values=X_umap_super[:, 0]).alias("x_umap_super"),
    pl.Series(values=X_umap_super[:, 1]).alias("y_umap_super"),
)
(
    p9.ggplot(df, p9.aes(x="x_umap_super", y="y_umap_super", color="cluster_id"))
    + p9.geom_point()
    + p9.theme_bw()
)
```

```python
(
    p9.ggplot(df[sent_row_idx], p9.aes(x="x_umap_super", y="y_umap_super", color="factor(cluster_id)"))
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors, name="Cluster ID")
)

```

### LDA

```python
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(embeddings, clusters)

df = df.with_columns(
    pl.Series(values=X_lda[:, 0]).alias("x_lda"), pl.Series(values=X_lda[:, 1]).alias("y_lda")
)
(
    p9.ggplot(df, p9.aes(x="x_lda", y="y_lda", color="cluster_id"))
    + p9.geom_point()
    + p9.theme_bw()
)
```

```python
(
    p9.ggplot(df[sent_row_idx], p9.aes(x="x_lda", y="y_lda", color="factor(cluster_id)"))
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors, name="Cluster ID")
)

```

### Cluster trick

Just add the cluster id or average to the embeddings


#### UMAP


Using cluster ids as an extra coordinate

```python
umap_model = umap.UMAP(random_state=1871, verbose=True)
X_umap_cid = umap_model.fit_transform(
    np.concatenate([embeddings, clusters[:, np.newaxis]], axis=-1)
)

df = df.with_columns(
    pl.Series(values=X_umap_cid[:, 0]).alias("x_umap_cid"),
    pl.Series(values=X_umap_cid[:, 1]).alias("y_umap_cid"),
)
(
    p9.ggplot(df, p9.aes(x="x_umap_cid", y="y_umap_cid", color="cluster_id"))
    + p9.geom_point()
    + p9.theme_bw()
)
```

```python
(
    p9.ggplot(df[sent_row_idx], p9.aes(x="x_umap_cid", y="y_umap_cid", color="factor(cluster_id)"))
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors, name="Cluster ID")
)

```

Using cluster averages: it's just too much for my boy umap

```python
# resp = resp_matrix(clusters).T
# cluster_averages = np.matmul(resp, embeddings) / resp.sum(axis=1, keepdims=True)
# embeddings_with_averages = np.concatenate([embeddings, cluster_averages[clusters, :]], axis=-1)

# umap_model = umap.UMAP(random_state=1871, verbose=True)
# X_umap_cemb = umap_model.fit_transform(
#     np.concatenate([embeddings_with_averages, clusters[:, np.newaxis]], axis=-1)
# )

# df = df.with_columns(
#     pl.Series(values=X_umap_cemb[:, 0]).alias("x_umap_cemb"),
#     pl.Series(values=X_umap_cemb[:, 1]).alias("y_umap_cemb"),
# )
# (
#     p9.ggplot(df, p9.aes(x="x_umap_cemb", y="y_umap_cemb", color="cluster_id"))
#     + p9.geom_point()
#     + p9.theme_bw()
# )

```

### Fit centers + reprojection

As in: we train a projection only on the cluster averages (which should really be the component means once clamps gets us that), then apply that projection on the whole dataset (and use it to visualize a single sentence).

```python
resp = resp_matrix(clusters).T
cluster_averages = np.matmul(resp, embeddings) / resp.sum(axis=1, keepdims=True)
```

#### PCA

```python
pca_model = PCA(n_components=2, random_state=1871)
pca_model.fit(cluster_averages)

X_pca_center = pca_model.transform(embeddings)

df = df.with_columns(
    pl.Series(values=X_pca_center[:, 0]).alias("x_pca_center"),
    pl.Series(values=X_pca_center[:, 1]).alias("y_pca_center"),
)
(
    p9.ggplot(df, p9.aes(x="x_pca_center", y="y_pca_center", color="cluster_id"))
    + p9.geom_point()
    + p9.theme_bw()
)
```

```python
(
    p9.ggplot(df[sent_row_idx], p9.aes(x="x_pca_center", y="x_pca_center", color="factor(cluster_id)"))
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors, name="Cluster ID")
)

```

#### UMAP

```python
umap_model = umap.UMAP(random_state=1871, verbose=True)
umap_model.fit(cluster_averages)

X_umap_center = umap_model.transform(embeddings)

df = df.with_columns(
    pl.Series(values=X_umap_center[:, 0]).alias("x_umap_center"),
    pl.Series(values=X_umap_center[:, 1]).alias("y_umap_center"),
)
(
    p9.ggplot(df, p9.aes(x="x_umap_center", y="y_umap_center", color="cluster_id"))
    + p9.geom_point()
    + p9.theme_bw()
)

```

```python
(
    p9.ggplot(
        df[sent_row_idx], p9.aes(x="x_umap_center", y="y_umap_center", color="factor(cluster_id)")
    )
    + p9.geom_text(p9.aes(label="guess"))
    + p9.theme_bw()
    + p9.scale_color_manual(cmap.colors, name="Cluster ID")
)

```

### ClusterPlot

```python
# cluster_plot_model = cluster_plot.ClusterPlot(
#     learning_rate=0.5,
#     n_iter=32,
#     batch_size=1,
#     reduce_all_points=True,
#     class_to_label={i: f"cluster_{i}" for i in set(clusters)},
#     anchors_method="birch",
#     show_fig=False,
#     save_fig=False,
#     save_fig_every=32,
#     show_anchors=False,
#     show_points=True,
#     show_loss_plot=False,
#     show_label_level_plots=False,
#     k=16,
#     random_state=1871,
#     magnitude_step=True,
#     top_greedy=1,
#     alpha=0.1,
#     remove_outliers_k=2,
#     douglas_peucker_tolerance=0.3,
#     smooth_iter=3,
# )

# X_clusterplot = cluster_plot_model.fit_transform(embeddings, clusters)
# cluster_plot_model.cluster_plot()
# plt.show()

```
