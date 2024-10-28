import pathlib

import numpy as np
import pandas as pd
import click


@click.command(
    help=(
        "Computes a Cluster Agreement Index (CAI) using two clustered dataframes paths"
        "as paremeters. See Theorem 2 in <https://webdocs.cs.ualberta.ca/~zaiane/postscript/aaai17.pdf>"
        "for mathematical formulas and proofs for or CAI computation."
    )
)
@click.argument("clusters_file1", type=click.Path(readable=True, dir_okay=False))
@click.argument(
    "clusters_file2", type=click.Path(writable=True, dir_okay=False, path_type=pathlib.Path)
)
def main(df1: pathlib.Path | None, df2: pathlib.Path | None):
    click.echo((f"Computing Cluster Agreement Index between Clusters from {df1} and {df2}"))
    cluster_1 = pd.read_csv(df1, sep="\t")
    cluster_2 = pd.read_csv(df2, sep="\t")

    merged_clusters = merge_guesses(cluster_1, cluster_2)
    matrix_u, matrix_v = convert_to_npmatrix(merged_clusters)
    contingency_table = compute_contingency_table(matrix_u, matrix_v)
    cai = compute_cmi(contingency_table)

    click.echo(f"CAI computed: {cai}")

    return cai


def merge_guesses(cluster_1, cluster_2):
    cluster1_df = cluster_1.sort_values("guess")
    cluster2_df = cluster_2.sort_values("guess")

    guess_merged_df = pd.merge(
        cluster1_df[["guess", "cluster_id"]],
        cluster2_df[["guess", "cluster_id"]],
        on="guess",
        suffixes=("_df1", "_df2"),
    )

    click.echo(f"Merged Dataframe\n: {guess_merged_df}")
    return guess_merged_df


def convert_to_npmatrix(merged_df):
    df1_incidence = merged_df.groupby(["guess", "cluster_id_df1"]).size().unstack(fill_value=0)
    df2_incidence = merged_df.groupby(["guess", "cluster_id_df2"]).size().unstack(fill_value=0)

    max_cluster_id_df1 = merged_df["cluster_id_df1"].max()
    max_cluster_id_df2 = merged_df["cluster_id_df2"].max()

    # reindex to have all cluster IDs
    df1_incidence = df1_incidence.reindex(columns=range(max_cluster_id_df1 + 1), fill_value=0)
    df2_incidence = df2_incidence.reindex(columns=range(max_cluster_id_df2 + 1), fill_value=0)

    # convert to numpy arrays
    matrix_u = df1_incidence.to_numpy()
    matrix_v = df2_incidence.to_numpy()

    click.echo(f"Matrix U:\n: {matrix_u}")
    click.echo(f"Matrix V:\n: {matrix_v}")

    return matrix_u, matrix_v


def compute_contingency_table(matrix_u, matrix_v):
    contingency_table = np.dot(matrix_u.T, matrix_v)
    return contingency_table


def compute_cmi(contingency_table):
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
