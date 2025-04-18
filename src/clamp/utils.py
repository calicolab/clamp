from typing import Union, cast
import numpy as np
import torch
import transformers


def avail_devices() -> list[str]:
    res = ["cpu"]
    if torch.cuda.is_available():
        res.append("cuda")
    if torch.backends.mps.is_available():
        res.append("mps")
    return res


def load_device() -> str:
    return avail_devices()[-1]


def load_lm_and_tokenizer(
    model_name_or_path: str,
    device: str = "cpu",
) -> Union[
    tuple[transformers.GPTNeoXForCausalLM, transformers.GPTNeoXTokenizerFast],
    tuple[transformers.GPT2LMHeadModel, transformers.GPT2TokenizerFast],
    tuple[transformers.RobertaForMaskedLM, transformers.RobertaTokenizerFast],
]:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device)
    if isinstance(model, transformers.GPTNeoXForCausalLM):
        tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(
            model_name_or_path,
            clean_up_tokenization_spaces=False,
            add_prefix_space=True,
            return_tensors="pt",
        )
    elif isinstance(model, transformers.GPT2LMHeadModel):
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
            model_name_or_path, clean_up_tokenization_spaces=False, return_tensors="pt"
        )
    elif isinstance(model, transformers.RobertaForCausalLM):
        model = transformers.RobertaForMaskedLM.from_pretrained(
            model_name_or_path, device_map=device
        )
        # FIXME: Do we need add_prefix_space here?
        tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
            model_name_or_path, clean_up_tokenization_spaces=False, return_tensors="pt"
        )
    else:
        raise ValueError(f"Model {model_name_or_path} of type {type(model)} is not supported")

    model.eval()  # Put the model in  evaluation mode
    # NOTE: there is no realiable way to freeze the model (inference mode only), we have to use
    # torch.inference_mode everywhere
    # TODO: do we want to torch.compile ? Benchmark if that's faster and doesn't change results ?
    print(f"Using device: {device}")
    return model, tokenizer


def resp_matrix(
    cluster_ids: np.ndarray[tuple[int], np.dtype[np.integer]],
) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
    """Compute the responsibility matrix for an array of cluster assignments.

    ## Arguments

    - `cluster_ids`: an array of positive integers coding that element `i` is in the `cluster_id[i]`
      cluster.

    ## Returns

    - `resp`: a matrix of shape `(n, m)` where `n=cluster_ids.shape[0]` and `m` is the largest
      cluster id such that `resp[i, j]` is `1` if `j == cluster_ids[i]`, and `0` otherwise. Can also
      be seen as stack of one-hot encodings.
    """
    n_clusters = cluster_ids.max() + 1
    return cast(
        np.ndarray[tuple[int, int], np.dtype[np.integer]],
        np.equal(np.arange(n_clusters)[np.newaxis, :], cluster_ids[:, np.newaxis]).astype(np.intp),
    )
