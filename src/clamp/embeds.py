import pathlib

import click
import polars as pl
import torch
import transformers
from rich.progress import Progress

from clamp.predictions import load_topk_predictions
from clamp.utils import avail_devices, load_device, load_lm_and_tokenizer


@click.command()
@click.argument("model_name_or_path", type=str)
@click.argument(
    "predictions_file", type=click.Path(readable=True, dir_okay=False, path_type=pathlib.Path)
)
@click.argument(
    "embeddings_file", type=click.Path(writable=True, dir_okay=False, path_type=pathlib.Path)
)
@click.option(
    "--device", type=click.Choice(avail_devices()), default=load_device(), show_default=True
)
@click.option("--layer", default=-1, type=int, show_default=True)
@click.option("--model-save-path", type=click.Path(writable=True, file_okay=False))
def main(
    embeddings_file: pathlib.Path,
    device: str,
    layer: int,
    model_name_or_path: str,
    model_save_path: str | None,
    predictions_file: pathlib.Path,
):
    click.echo(f"Loading model and tokenizer from {model_name_or_path}")
    model, tokenizer = load_lm_and_tokenizer(model_name_or_path, device=device)
    model_name = model.config._name_or_path
    click.echo(f"Computing {layer}th layer embeddings of {predictions_file} with {model_name}.")
    topk_df = load_topk_predictions(predictions_file)

    # OK it looks messy but it gets us a progress bar
    with Progress() as progress:
        process_task = progress.add_task(
            description=f"Processing with {model_name}", total=topk_df.height
        )

        # TODO: batch this?
        def process(preamble: str, guess: str) -> torch.Tensor:
            progress.update(process_task, advance=1)  # noqa: B023  # The closure is only ever invoked in the loop so this is OK
            return next_token_embed(
                preamble=preamble,
                guess=guess,
                tokenizer=tokenizer,
                model=model,
                layer=layer,
            )

        # FIXME: to_list+list+conversion to array is horribly inefficient but see
        # <https://github.com/pola-rs/polars/issues/13289> for why. It still saves memory to do it
        # in lazy mode though, which **kinda** makes it worth the trouble instead of building a big
        # ndarray on the side and loading that as a column, which would require fitting all the
        # embeddings in RAM at the same time.
        topk_df.lazy().with_columns(
            pl.struct(["preamble", "guess"])
            .map_elements(
                lambda s: process(preamble=s["preamble"], guess=s["guess"]).tolist(),
                return_dtype=pl.List(pl.Float64),
            )
            .list.to_array(model.config.hidden_size)
            .alias("embedding")
        ).sink_parquet(embeddings_file)

    click.echo(f"Embeddings written to {embeddings_file}")

    if model_save_path is not None:
        click.echo(f"Saving model to {model_save_path}")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)


def load_embeddings(embeddings_file: str | pathlib.Path) -> pl.DataFrame:
    return pl.read_parquet(
        embeddings_file, columns=["preamble", "token_id", "guess", "logit", "embedding"]
    )


@torch.inference_mode()
def next_token_embed(
    preamble: str,
    guess: str,
    tokenizer: transformers.PreTrainedTokenizerFast,
    model: transformers.PreTrainedModel,
    layer: int,
) -> torch.Tensor:
    """Get embeddings for ONE preamble/guess pair"""
    # FIXME: we shouldn't need to tokenize twice I think
    start_ix = len(tokenizer(preamble)["input_ids"])
    preamble_w_guess = preamble + guess
    # tokenize new complete sentence
    new_preamble_tokenized = tokenizer(preamble_w_guess, return_tensors="pt").to(model.device)
    embeds = model(**new_preamble_tokenized, output_hidden_states=True).hidden_states
    last_token_embed = embeds[layer][0, start_ix, :]
    return last_token_embed


if __name__ == "__main__":
    main()
