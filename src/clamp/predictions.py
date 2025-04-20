import pathlib
from typing import NamedTuple, Sequence, cast

import click
import polars as pl
import torch
import transformers
from rich.progress import track

from clamp.utils import avail_devices, load_device, load_lm_and_tokenizer


# example use
#  python predictions.py gpt2 ../sentences.txt predictions.tsv --k 5
@click.command()
@click.argument("model_name_or_path", type=str)
@click.argument(
    "sentences_file", type=click.Path(readable=True, dir_okay=False, path_type=pathlib.Path)
)
@click.argument(
    "predictions_file", type=click.Path(writable=True, dir_okay=False, path_type=pathlib.Path)
)
@click.option(
    "--device", type=click.Choice(avail_devices()), default=load_device(), show_default=True
)
@click.option("--k", type=int, default=5, show_default=True)
def main(
    device: str,
    k: int,
    model_name_or_path: str,
    predictions_file: pathlib.Path,
    sentences_file: pathlib.Path,
):
    click.echo(f"Loading model and tokenizer from {model_name_or_path}")
    model, tokenizer = load_lm_and_tokenizer(model_name_or_path, device=device)
    model_name = model.config._name_or_path
    with sentences_file.open() as in_stream:
        sentences = [x.rstrip() for x in in_stream]
    click.echo(f"Getting top {k} predictions for {sentences_file} using {model_name!r}.")
    topk = get_topk_completions(sentences, model, tokenizer, k)
    click.echo(f"Saving predictions to {predictions_file}.")
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        data=topk, orient="row", schema=["preamble", "token_id", "guess", "logit"]
    ).write_csv(predictions_file, separator="\t")


@torch.inference_mode()
def nwp_predictions(
    preamble: str,
    model: transformers.GPTNeoXForCausalLM | transformers.GPT2LMHeadModel,
    tokenizer: transformers.GPTNeoXTokenizerFast | transformers.GPT2TokenizerFast,
) -> torch.Tensor:
    target_sent = tokenizer(preamble, return_tensors="pt").to(model.device)
    target_out = model(**target_sent, output_hidden_states=True)
    predictions = target_out.logits[0, -1, :]  # Get the logits for the last token
    return predictions


@torch.inference_mode()
def mlm_predictions(
    preamble: str,
    model: transformers.RobertaForMaskedLM,
    tokenizer: transformers.RobertaTokenizerFast,
) -> torch.Tensor:
    preamble_mask = f"{preamble} <mask>"
    target_sent = tokenizer(preamble_mask, return_tensors="pt").to(model.device)
    target_out = model(**target_sent)
    predictions = target_out.logits[0, -2, :]  # Get the logits for the last token
    return predictions


class TopKReturn(NamedTuple):
    preamble: str
    token_id: int
    guess: str
    logit: float


def get_topk_completions(
    preambles: Sequence[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
    k: int,
) -> list[TopKReturn]:
    if isinstance(model, (transformers.GPTNeoXForCausalLM, transformers.GPT2LMHeadModel)):
        gen_prediction = nwp_predictions
    elif isinstance(model, (transformers.RobertaForMaskedLM, transformers.CamembertForMaskedLM)):
        gen_prediction = mlm_predictions
    else:
        raise ValueError(f"Unsupported model type: {model}")
    # FIXME: isn't there a safer way to do this than relying on decode and the spaces?
    initial_ids = torch.tensor(
        [
            token_id
            for token_id in tokenizer.get_vocab().values()
            if tokenizer.decode(token_id, clean_up_tokenization_spaces=False).startswith(" ")
        ],
        device=model.device,
    )
    data: list[TopKReturn] = []
    for preamble in track(preambles):
        predictions = gen_prediction(preamble, model, tokenizer)  # type: ignore  # we good
        topk = predictions[initial_ids].topk(k=k)
        topk_predictions = initial_ids[topk.indices]
        for token_id, logit in zip(topk_predictions, topk.values, strict=True):
            token_str = tokenizer.decode(token_id).strip()
            data.append(
                TopKReturn(
                    preamble=preamble,
                    token_id=cast(int, token_id.item()),
                    guess=token_str,
                    logit=logit.item(),
                )
            )
    return data


# TODO: Do we need this?
def load_topk_predictions(predictions_file: str | pathlib.Path) -> pl.DataFrame:
    return pl.read_csv(predictions_file, separator="\t")


if __name__ == "__main__":
    main()
