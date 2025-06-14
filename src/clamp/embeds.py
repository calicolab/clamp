import pathlib
from typing import Literal

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
@click.option(
    "--pooling",
    type=click.Choice(["first", "mean"]),
    default="mean",
    show_default=True,
    help="For multi-token words, wether to use the embedding of the first token or average them.",
)
def main(
    embeddings_file: pathlib.Path,
    device: str,
    layer: int,
    model_name_or_path: str,
    model_save_path: str | None,
    pooling: Literal["first", "mean"],
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
                guess=guess,
                layer=layer,
                model=model,
                mean_pooling=(pooling == "mean"),
                preamble=preamble,
                tokenizer=tokenizer,
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
    return pl.read_parquet(embeddings_file)


# FIXME: batch this
@torch.inference_mode()
def next_token_embed(
    preamble: str,
    guess: str,
    tokenizer: transformers.PreTrainedTokenizerFast,
    model: transformers.PreTrainedModel,
    layer: int,
    mean_pooling: bool = True,
) -> torch.Tensor:
    """Get embeddings for ONE preamble/guess pair"""
    # This doesn't nearly catch all subtle tokenization gotchas but it's something
    guess = guess.strip()
    if len(guess) == 0:
        raise ValueError(f"Empty guess: {guess!r} for preamble {preamble!r}")
    preamble_w_guess = f"{preamble} {guess}"
    # tokenize new complete sentence
    tokenized = tokenizer(preamble_w_guess, add_special_tokens=True, return_tensors="pt").to(
        model.device
    )
    start_idx = tokenized.char_to_token(0, len(preamble_w_guess) - len(guess))
    if start_idx is None:
        raise ValueError(f"Tokenization error for preamble {preamble!r} with guess {guess!r}")
    embeds = model(**tokenized, output_hidden_states=True).hidden_states

    if mean_pooling:
        end_idx = tokenized.char_to_token(0, len(preamble_w_guess) - 1)
        return embeds[layer][0, start_idx : end_idx + 1, :].mean(dim=0)
    else:
        return embeds[layer][0, start_idx, :]


if __name__ == "__main__":
    main()
