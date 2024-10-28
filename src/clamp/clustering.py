import pathlib
import pickle
from typing import Literal

import click
import polars as pl
from sklearn.mixture import BayesianGaussianMixture

from clamp.embeds import load_embeddings


@click.command(
    help=(
        "Train a BGMM model using pre-computed embeddings. See"
        " <https://scikit-learn.org/stable/api/sklearn.mixture.html>"
        " for explanations of the parameters."
    )
)
@click.argument("embeddings_file", type=click.Path(readable=True, dir_okay=False))
@click.argument(
    "clusters_file", type=click.Path(writable=True, dir_okay=False, path_type=pathlib.Path)
)
@click.option("--bgmm-file", type=click.Path(writable=True, dir_okay=False, path_type=pathlib.Path))
@click.option(
    "--covariance-type",
    type=click.Choice(["full", "tied", "diag", "spherical"]),
    default="full",
    show_default=True,
)
@click.option(
    "--max-iter", type=click.IntRange(1), default=4000, show_default=True, metavar="INTEGER"
)
@click.option(
    "--n-clusters", type=click.IntRange(1), default=100, show_default=True, metavar="INTEGER"
)
@click.option(
    "--progress",
    help="How many iterations beteween progrss display.",
    type=int,
    default=0,
    show_default=0,
)
@click.option(
    "--tol",
    default=1e-6,
    metavar="FLOAT",
    show_default=True,
    type=click.FloatRange(0, min_open=True),
)
@click.option("--random-state", type=click.INT, default=42, show_default=True)
@click.option(
    "--reg_covar",
    default="1e-6",
    metavar="FLOAT",
    show_default=True,
    type=click.FloatRange(0, min_open=True),
)
@click.option(
    "--verbose/--quiet", default=True, help="Display Log-likelihood at the end.", is_flag=True
)
@click.option(
    "--weight-concentration-prior",
    metavar="FLOAT",
    type=click.FloatRange(0, min_open=True),
)
@click.option(
    "--weight-concentration-prior-type",
    type=click.Choice(["dirichlet_process", "dirichlet_distribution"]),
    default="dirichlet_process",
    show_default=True,
)
def main(
    bgmm_file: pathlib.Path | None,
    clusters_file: pathlib.Path | None,
    covariance_type: Literal["full", "tied", "diag", "spherical"],
    embeddings_file: pathlib.Path,
    max_iter: int,
    n_clusters: int,
    progress: int,
    random_state: int,
    reg_covar: float,
    tol: float,
    verbose: bool,
    weight_concentration_prior: float | None,
    weight_concentration_prior_type: Literal["dirichlet_process", "dirichlet_distribution"],
):
    click.echo((f"Fitting {n_clusters} clusters  with embeddings from {embeddings_file}"))
    embeddings_df = load_embeddings(embeddings_file)
    embeddings = embeddings_df["embedding"].to_numpy()
    bgmm = BayesianGaussianMixture(
        covariance_type=covariance_type,
        max_iter=max_iter,
        n_components=n_clusters,
        random_state=random_state,
        reg_covar=reg_covar,
        tol=tol,
        verbose=2 if progress > 0 else 0,
        verbose_interval=progress,
        weight_concentration_prior=weight_concentration_prior,
        weight_concentration_prior_type=weight_concentration_prior_type,
    )
    cluster_ids = bgmm.fit_predict(embeddings).tolist()
    click.echo("Training done.")
    if verbose > 0:
        click.echo(f"Log-likelihood: {bgmm.score(embeddings)}")
    if bgmm_file is not None:
        with bgmm_file.open("wb") as out_stream:
            pickle.dump(bgmm, out_stream, protocol=pickle.HIGHEST_PROTOCOL)
    embeddings_df.select(pl.all().exclude("embedding")).with_columns(
        pl.Series(values=cluster_ids).alias("cluster_id")
    ).write_csv(clusters_file, separator="\t")


if __name__ == "__main__":
    main()