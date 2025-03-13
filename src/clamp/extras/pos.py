import pathlib
from typing import Union

import click
import numpy as np
import polars as pl

from clamp.embeds import load_embeddings


def add_pos_from_subtlex(
    embeddings_df: pl.DataFrame,
    subtlex_file: Union[str, pathlib.Path],
) -> pl.DataFrame:
    return embeddings_df.join(
        pl.scan_csv(subtlex_file).select("Word", "Dom_PoS_SUBTLEX"),
        how="left",
        left_on="guess",
        right_on="Word",
        validate="m:1",
    )


def center_embeddings(embeddings_df: pl.DataFrame) -> pl.DataFrame:
    embeddings = embeddings_df["embedding"].to_numpy()

    # TODO: there might be a way to get that directly from bgmm? Maybe even the likelihood of each
    # sample belonging to its cluster?
    resp = cluster_ids[np.newaxis, :] == np.arange(cluster_ids.max() + 1)[:, np.newaxis]
    cluster_averages = np.matmul(resp, embeddings) / resp.sum(axis=1, keepdims=True)


@click.command(
    help=(
        "Add POS to an embedding file, optionally substracting to every embedding the average of"
        " the embeddings of responses with the same POS."
    )
)
@click.argument("pos_source", type=str)
@click.argument("embeddings_file", type=click.Path(readable=True, dir_okay=False))
@click.argument(
    "output_file", type=click.Path(writable=True, dir_okay=False, path_type=pathlib.Path)
)
@click.option(
    "--center", help="Susbstract the average embedding of the relevant POS.", is_flag=True
)
def add_pos_to_embeddings(
    center: bool,
    embeddings_file: pathlib.Path,
    pos_source: str,
    output_file: pathlib.Path | None,
):
    embeddings_df = load_embeddings(embeddings_file)
    res_df = add_pos_from_subtlex(embeddings_df, pos_source)

    if center:
        res_df = center_embeddings(res_df)

    res_df = embeddings_df.with_columns(
        pl.Series(values=cluster_ids.tolist()).alias("cluster_id")
    ).write_parquet(output_file)


if __name__ == "__main__":
    main()
