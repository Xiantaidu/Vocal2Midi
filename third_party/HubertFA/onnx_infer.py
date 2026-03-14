import json
import pathlib

import click
import onnxruntime as ort

from tools.config_utils import check_configs
from tools.infer_base import InferenceBase


class InferenceOnnx(InferenceBase):
    def __init__(self, onnx_path: pathlib.Path):
        super().__init__()
        self.model = None
        self.model_path = onnx_path
        self.model_folder = onnx_path.parent

    def load_config(self):
        check_configs(self.model_folder, suffix='json')
        with open(self.model_folder / 'VERSION', 'r', encoding='utf-8') as f:
            assert int(f.readline().strip()) == 5, f"onnx model version must be 5."
        with open(self.model_folder / 'vocab.json', 'r', encoding='utf-8') as f:
            self.vocab = json.loads(f.read())
        with open(self.model_folder / 'config.json', 'r', encoding='utf-8') as f:
            config = json.loads(f.read())

        self.mel_cfg = config['mel_spec_config']
        self.vocab_folder = self.model_folder

    def load_model(self):
        self.model = self.create_session(self.model_folder / 'model.onnx')

    def _infer(self, padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx, wav_length, non_lexical_phonemes):
        results = self.run_onnx(self.model, {'waveform': [padded_wav]})

        words, _ = self.fa_decoder.decode(
            ph_frame_logits=results['ph_frame_logits'][:, :, padded_frames:],
            ph_edge_logits=results['ph_edge_logits'][:, padded_frames:],
            wav_length=wav_length, ph_seq=ph_seq, word_seq=word_seq, ph_idx_to_word_idx=ph_idx_to_word_idx
        )

        non_lexical_words = self.nll_decoder.decode(cvnt_logits=results['cvnt_logits'][:, :, padded_frames:],
                                                    wav_length=wav_length, non_lexical_phonemes=non_lexical_phonemes)
        return words, non_lexical_words

    @staticmethod
    def run_onnx(session, input_dict):
        output_names = [output.name for output in session.get_outputs()]
        return dict(zip(output_names, session.run(output_names, input_dict)))

    @staticmethod
    def create_session(onnx_path):
        providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(onnx_path), options, providers=providers)


@click.command()
@click.option("--onnx_path", "-m", required=True, type=pathlib.Path, help="Path to Onnx models")
@click.option("--wav_folder", "-wf", default="segments", type=pathlib.Path, help="Input folder path")
@click.option("--out_path", "-o", default=None, type=str, help="path to the output label")
@click.option("--g2p", "-g", default="dictionary", type=str, help="G2P class name")
@click.option("--non_lexical_phonemes", "-np", default="AP", type=str, help="non speech phonemes, exp. AP,EP")
@click.option("--language", "-l", default="zh", help="Dictionary language")
@click.option("--dictionary", "-d", type=pathlib.Path, help="Custom dictionary path")
@click.option("--pad_times", "-pt", type=int, default=1, help="The number of times to pad blank audio before reasoning")
@click.option("--pad_length", "-pl", type=int, default=5,
              help="The max length of blank audio on the pad before inference")
@click.option("--out_format", "-f", type=click.Choice(['TextGrid', 'HTK']), default='TextGrid',
              help="Format to export predictions to.")
def infer(onnx_path: pathlib.Path, wav_folder: pathlib.Path, out_path: pathlib.Path | None, g2p: str,
          non_lexical_phonemes: str, language: str, dictionary: pathlib.Path, pad_times: int, pad_length: int, out_format: str):
    assert onnx_path.exists() and onnx_path.is_file() and onnx_path.suffix == '.onnx', \
        f"Path {onnx_path} does not exist or is not a onnx file."

    inference = InferenceOnnx(onnx_path=onnx_path)
    inference.load_config()
    inference.init_decoder()
    inference.load_model()
    inference.get_dataset(wav_folder=wav_folder, language=language, g2p=g2p, dictionary_path=dictionary)
    inference.infer(non_lexical_phonemes=non_lexical_phonemes, pad_times=pad_times, pad_length=pad_length)
    inference.export(output_folder=wav_folder if out_path is None else out_path, output_format=[out_format.lower()])


if __name__ == '__main__':
    infer()
