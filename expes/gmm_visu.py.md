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

```python
import numpy as np
import numpy.random
import plotnine as p9
import polars as pl
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from tol_colors import rainbow_discrete
```

```python
# See <https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/> for a nice
# summary of why

def ellipsoid_to_cov(angle, scales):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    antirotation = np.array([[c, s], [-s, c]])
    scaling_sq = np.pow(np.diag(scales), 2)
    return np.linalg.multi_dot([rotation, scaling_sq, antirotation])


ellipsoid_to_cov(-np.pi/4.0, [2.0, 1.0])
```

```python
state = np.random.default_rng(1871)

n_per_component = 1024

components = [
    multivariate_normal(mean=[-2.0, 2.0], cov=ellipsoid_to_cov(-np.pi / 4.0, [1.0, 0.5])),
    multivariate_normal(mean=[1.0, 1.0], cov=[[0.5, 0.3], [0.3, 0.5]]),
    multivariate_normal(mean=[2.0, -2.0], cov=[[0.5, 0.1], [0.1, 0.9]]),
    multivariate_normal(mean=[-2.0, -3.0], cov=[[0.2, 0.01], [0.01, 0.2]]),
]

samples = np.concatenate([d.rvs(size=n_per_component, random_state=state) for d in components])
```

Display with Pyplot

```python
fig, ax = plt.subplots()

ax.scatter(
    samples[:, 0],
    samples[:, 1],
    s=4,
    marker=".",
    c=np.arange(samples.shape[0]) // n_per_component,
    cmap=rainbow_discrete(len(components)),
)

plt.show()
```

Display with plotnine

```python
samples_df = pl.DataFrame({
    "x": samples[:, 0],
    "y": samples[:, 1],
    "component": np.arange(samples.shape[0]) // n_per_component,
})
(
    p9.ggplot(samples_df, p9.aes(x="x", y="y", color="factor(component)"))
    + p9.geom_point(shape=".")
    + p9.theme_bw()
    + p9.scale_color_manual(rainbow_discrete(len(components)).colors)
)

```
