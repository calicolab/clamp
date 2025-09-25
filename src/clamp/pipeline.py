import base64
import hashlib
import pathlib
import json
from typing import Any, NotRequired, TypedDict

import click
from sklearn.model_selection import ParameterGrid
from yaml import safe_load

import clamp.clustering
import clamp.embeds
import clamp.predictions


def aslist(value: Any) -> list:
    if isinstance(value, list):
        return value
    return [value]


class PipelineConfig(TypedDict):
    prediction: NotRequired[dict[str, ParameterGrid]]
    embedding: NotRequired[dict[str, ParameterGrid]]
    clustering: NotRequired[ParameterGrid]


# TODO: do something with pydantic?
def load_pipeline(config_path: str | pathlib.Path):
    with open(config_path, "r") as file:  # noqa: PTH123
        config = safe_load(file)["pipeline"]

    model_lst = aslist(config["model"])

    res = {}

    if "prediction" in config:
        res["prediction"] = {
            model: ParameterGrid({k: aslist(v) for k, v in config["prediction"].items()})
            for model in model_lst
        }

    if "embedding" in config:
        res["embedding"] = {
            model: ParameterGrid({k: aslist(v) for k, v in config["embedding"].items()})
            for model in model_lst
        }

    if "clustering" in config:
        res["clustering"] = ParameterGrid({k: aslist(v) for k, v in config["clustering"].items()})

    return model_lst, res


def str_for_filename(o: Any) -> str:
    return str(o).replace(":", "::").replace("/", ":")


def params_str(params: dict[str, Any]) -> str:
    return "|".join(f"{k}={str_for_filename(v)}" for k, v in params.items())


def file_name_or_meta(name: str, directory: pathlib.Path, max_length: int = 255) -> str:
    """If the name is under the acceptable length, simply return it,
    otherwise return a small identifier and keep track in a metadata file."""
    if len(name) < max_length:
        return name
    metadata_file = directory / "files_metadata.json"
    if metadata_file.exists():
        metadata = json.load(metadata_file.open())
    else:
        metadata = {}
    # This is honestly terrible, but it's fast and short
    name_hash = base64.urlsafe_b64encode(hashlib.sha3_256(name.encode("utf-8")).digest()).decode()
    metadata[name_hash] = name
    json.dump(metadata, metadata_file.open("w"), indent=0)

    return name_hash


@click.command()
@click.argument(
    "sentences_file", type=click.Path(dir_okay=False, readable=True, path_type=pathlib.Path)
)
@click.argument(
    "config_file", type=click.Path(dir_okay=False, readable=True, path_type=pathlib.Path)
)
@click.argument(
    "output_dir", type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path)
)
@click.option("--overwrite", is_flag=True, help="Whether to overwrite existing intermediary files.")
@click.pass_context
def main(
    ctx: click.Context,
    config_file: pathlib.Path,
    output_dir: pathlib.Path,
    overwrite: bool,
    sentences_file: pathlib.Path,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    models, grids = load_pipeline(config_file)

    predictions_output_dir = output_dir / "predictions"
    if "prediction" in grids:
        prediction_files: dict[str, list[pathlib.Path]] = {}
        predictions_output_dir.mkdir(exist_ok=True)
        for model, pred_grid in grids["prediction"].items():
            prediction_files[model] = []
            for predict_config in pred_grid:
                click.echo(f"Predictions for {model} with config {predict_config}")
                config_str = params_str(predict_config)
                pred_stem = file_name_or_meta(
                    f"{str_for_filename(model)}|{config_str}", directory=predictions_output_dir
                )
                prediction_output_file = predictions_output_dir / f"{pred_stem}.tsv"

                prediction_files[model].append(prediction_output_file)
                if prediction_output_file.exists() and not overwrite:
                    continue

                ctx.invoke(
                    clamp.predictions.main,
                    model_name_or_path=model,
                    sentences_file=sentences_file,
                    predictions_file=prediction_output_file,
                    **predict_config,
                )
    else:
        click.echo(
            f"No prediction config: skipping step and use {predictions_output_dir} as inputs source"
        )
        prediction_files = {m: list(predictions_output_dir.glob("*")) for m in models}

    embeddings_output_dir = output_dir / "embeddings"
    if "embedding" in grids:
        embedding_files: list[pathlib.Path] = []
        embeddings_output_dir.mkdir(exist_ok=True)
        for model, embed_grid in grids["embedding"].items():
            for pred_file in prediction_files[model]:
                for embedding_config in embed_grid:
                    click.echo(
                        f"Embeddings for {model} with config {embedding_config} on {pred_file.name}"
                    )
                    config_str = params_str(embedding_config)
                    embed_stem = file_name_or_meta(
                        f"{pred_file.stem}+{config_str}", directory=embeddings_output_dir
                    )
                    # In case we had no prediction step or it was done with a different model
                    # somehow
                    if (m := str_for_filename(model)) not in embed_stem:
                        embed_stem = f"{m}|{embed_stem}"
                    embeddings_output_file = embeddings_output_dir / f"{embed_stem}.parquet"

                    embedding_files.append(embeddings_output_file)
                    if embeddings_output_file.exists() and not overwrite:
                        continue

                    ctx.invoke(
                        clamp.embeds.main,
                        model_name_or_path=model,
                        predictions_file=pred_file,
                        embeddings_file=embeddings_output_file,
                        **embedding_config,
                    )
    else:
        click.echo(
            f"No embedding config: skipping step and use {embeddings_output_dir} as inputs source"
        )
        embedding_files = list(embeddings_output_dir.glob("*"))

    if "clustering" in grids:
        clustering_output_dir = output_dir / "clustering"
        clustering_output_dir.mkdir(exist_ok=True)
        for clustering_config in grids["clustering"]:
            for embed_file in embedding_files:
                click.echo(f"Clustering with config {clustering_config} on {embed_file.name}")
                config_str = params_str(clustering_config)
                cluster_stem = file_name_or_meta(
                    f"{embed_file.stem}+{config_str}", directory=clustering_output_dir
                )
                if clustering_config.get("keep_embeddings", False):
                    clustering_output_file = clustering_output_dir / f"{cluster_stem}.parquet"
                else:
                    clustering_output_file = clustering_output_dir / f"{cluster_stem}.tsv"
                click.echo(f"Saving to {clustering_output_file}")
                ctx.invoke(
                    clamp.clustering.main,
                    embeddings_file=embed_file,
                    clusters_file=clustering_output_file,
                    **clustering_config,
                )
    else:
        click.echo("No clustering config: skipping step.")


if __name__ == "__main__":
    main()
