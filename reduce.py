import pathlib

import click


@click.command(help="Reduce a checkpoint file, only keeping the 'state_dict' key for inference.")
@click.argument(
    "input_ckpt", metavar="INPUT_PATH", type=click.Path(
        exists=True, dir_okay=False, readable=True, path_type=pathlib.Path
    ),
)
@click.argument(
    "output_ckpt", metavar="OUTPUT_PATH",
    type=click.Path(
        exists=False, dir_okay=False, writable=True, path_type=pathlib.Path
    ),
)
def reduce(
        input_ckpt: pathlib.Path,
        output_ckpt: pathlib.Path,
):
    import torch
    from inference.api import load_state_dict_for_inference
    state_dict = load_state_dict_for_inference(input_ckpt, ema=True)
    ckpt = {
        "state_dict": state_dict
    }
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_ckpt)


if __name__ == "__main__":
    reduce()
