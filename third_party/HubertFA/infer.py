import pathlib

import click
import torch

from networks.task.forced_alignment_task import LitForcedAlignmentTask
from networks.task.non_lexical_labeler_task import LitNonLexicalLabelerTask
from tools.binarize_util import get_curves
from tools.config_utils import check_configs, load_yaml
from tools.encoder import UnitsEncoder
from tools.infer_base import InferenceBase


class InferenceLit(InferenceBase):
    def __init__(self, nll_path: pathlib.Path, fa_path: pathlib.Path, encoder: pathlib.Path | None):
        super().__init__()
        self.encoder = encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.fa_config = None
        self.hubert_cfg = None

        self.nll_path = nll_path
        self.nll_folder = nll_path.parent
        self.fa_path = fa_path
        self.fa_folder = fa_path.parent

        self.fa_model = None
        self.nll_model = None
        self.unitsEncoder = None

    def load_config(self):
        check_configs(self.fa_folder, suffix='yaml')
        self.vocab = {**load_yaml(self.nll_folder / "vocab.yaml"), **load_yaml(self.fa_folder / "vocab.yaml")}
        self.vocab_folder = self.fa_folder
        self.fa_config = load_yaml(self.fa_folder / "config.yaml")
        self.hubert_cfg = self.fa_config['hubert_config']
        self.mel_cfg = self.fa_config['mel_spec_config']

    def load_model(self):
        self.nll_model = LitNonLexicalLabelerTask.load_from_checkpoint(self.nll_path)
        self.fa_model = LitForcedAlignmentTask.load_from_checkpoint(self.fa_path)
        self.unitsEncoder = UnitsEncoder(hubert_config=self.hubert_cfg, mel_config=self.mel_cfg,
                                         encoder_ckpt=self.encoder, device=self.device)

    def _infer(self, padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx, wav_length, non_lexical_phonemes):
        input_feature = self.unitsEncoder.forward(torch.as_tensor(padded_wav, device=self.device).unsqueeze(0),
                                                  self.mel_cfg["sample_rate"], self.mel_cfg["hop_size"])  # [B, T, C]
        input_feature = input_feature.transpose(1, 2).half().float()
        curves = get_curves(torch.as_tensor(padded_wav, device=self.device), input_feature.shape[-1],
                            self.fa_model.window_size, self.fa_model.hop_size)  # [B, C, T]

        with torch.no_grad():
            (
                ph_frame_logits,  # (B, C, T)
                ph_edge_logits,  # (B, T)
                ctc_logits,  # (B, C, T)
            ) = self.fa_model.forward(input_feature, curves)
            cvnt_logits = self.nll_model.forward(input_feature)  # [B, N, T]

        words, _ = self.fa_decoder.decode(
            ph_frame_logits.float().cpu().numpy(),  # (B, C, T)
            ph_edge_logits.float().cpu().numpy(),
            wav_length, ph_seq, word_seq, ph_idx_to_word_idx
        )

        non_lexical_words = self.nll_decoder.decode(
            cvnt_logits=cvnt_logits.float().cpu().numpy(),
            wav_length=wav_length,
            non_lexical_phonemes=non_lexical_phonemes
        )

        return words, non_lexical_words


@click.command()
@click.option("--nll_path", "-nll", required=True, type=pathlib.Path, help="Path to nll models")
@click.option("--fa_path", "-fa", required=True, type=pathlib.Path, help="Path to fa models")
@click.option("--out_path", "-o", default=None, type=str, help="Path to the output label")
@click.option("--encoder", "-e", default=None, type=str, help="Path to the encoder model")
@click.option("--wav_folder", "-wf", default="segments", type=pathlib.Path, help="Input folder path")
@click.option("--g2p", "-g", default="dictionary", type=str, help="G2P class name")
@click.option("--non_lexical_phonemes", "-np", default="AP", type=str, help="Non speech phonemes, exp. AP,EP")
@click.option("--language", "-l", default="zh", help="Dictionary language")
@click.option("--dictionary", "-d", type=pathlib.Path, help="Custom dictionary path")
@click.option("--pad_times", "-pt", type=int, default=1, help="The number of times to pad blank audio before reasoning")
@click.option("--pad_length", "-pl", type=int, default=5,
              help="The max length of blank audio on the pad before inference")
@click.option("--out_format", "-f", type=click.Choice(['TextGrid', 'HTK']), default='TextGrid',
              help="Format to export predictions to.")
def infer(nll_path: pathlib.Path, fa_path: pathlib.Path, out_path: pathlib.Path | None, encoder: pathlib.Path | None,
          wav_folder: pathlib.Path, g2p: str, non_lexical_phonemes: str, language: str, dictionary: pathlib.Path,
          pad_times: int, pad_length: int, out_format: str):
    assert nll_path.exists() and nll_path.is_file() and nll_path.suffix == '.ckpt', \
        f"Path {nll_path} does not exist or is not a ckpt file."
    assert fa_path.exists() and fa_path.is_file() and fa_path.suffix == '.ckpt', \
        f"Path {fa_path} does not exist or is not a ckpt file."

    inference = InferenceLit(nll_path=nll_path, fa_path=fa_path, encoder=encoder)
    inference.load_config()
    inference.init_decoder()
    inference.load_model()
    inference.get_dataset(wav_folder=wav_folder, language=language, g2p=g2p, dictionary_path=dictionary)
    inference.infer(non_lexical_phonemes=non_lexical_phonemes, pad_times=pad_times, pad_length=pad_length)
    inference.export(output_folder=wav_folder if out_path is None else out_path, output_format=[out_format.lower()] )


if __name__ == '__main__':
    infer()