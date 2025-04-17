import pathlib
from typing import cast

import numpy as np
import polars as pl
import click


@click.command(
    help=(
        "Computes a Cluster Agreement Index (CAI) using two clustered dataframes paths"
        "as paremeters. See Theorem 2 in <https://webdocs.cs.ualberta.ca/~zaiane/postscript/aaai17.pdf>"
        "for mathematical formulas and proofs for or CAI computation."
    )
)
@click.argument("clusters1_path", type=click.Path(dir_okay=False))
@click.argument("clusters2_path", type=click.Path(dir_okay=False))
def main(clusters1_path: pathlib.Path, clusters2_path: pathlib.Path):
    clusters_1 = pl.read_csv(clusters1_path, separator="\t").sort(
        by=[pl.col("preamble"), pl.col("guess")]
    )
    clusters_2 = pl.read_csv(clusters2_path, separator="\t").sort(
        by=[pl.col("preamble"), pl.col("guess")]
    )

    matrix_u = get_resp_matrix(clusters_1["guess"].to_numpy())
    matrix_v = get_resp_matrix(clusters_2["guess"].to_numpy())
    contingency_table = np.inner(matrix_u, matrix_v)
    cai = compute_cmi(contingency_table)

    click.echo(str(cai))


def get_resp_matrix(
    clusters: np.ndarray[tuple[int], np.dtype[np.integer]],
) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
    n_clusters = clusters.max()
    return cast(
        np.ndarray[tuple[int, int], np.dtype[np.integer]],
        np.equal(np.arange(n_clusters)[np.newaxis, :], clusters[:, np.newaxis]).astype(np.intp),
    )


def compute_cmi(
    contingency_table: np.ndarray[tuple[int, int], np.dtype[np.integer]],
) -> np.floating:
    # number of points
    n = np.sum(contingency_table)

    # marginal sums
    row_sums = np.sum(contingency_table, axis=1)  # sum of rows - clusters in U
    col_sums = np.sum(contingency_table, axis=0)  # sum of columns - clusters in V

    # compute entropy H(U), H(V), and joint entropy H(U, V)
    entropy_u = -np.sum((row_sums / n) * np.log(row_sums / n + 1e-10))  # +1e-10 to avoid log(0)
    entropy_v = -np.sum((col_sums / n) * np.log(col_sums / n + 1e-10))

    joint_entropy = -np.sum((contingency_table / n) * np.log(contingency_table / n + 1e-10))

    # compute mutual information I(U, V)
    mutual_info = entropy_u + entropy_v - joint_entropy

    # compute Clustering Mutual Information
    clustering_index = mutual_info / ((entropy_u + entropy_v) / 2)

    return clustering_index


if __name__ == "__main__":
    main()
