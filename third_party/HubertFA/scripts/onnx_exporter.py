import json
import os.path
import pathlib
import shutil

import click
import onnx
import onnxsim
import torch
import torchaudio
from transformers import HubertModel

from networks.task.forced_alignment_task import LitForcedAlignmentTask
from networks.task.non_lexical_labeler_task import LitNonLexicalLabelerTask
from tools.binarize_util import PowerCurveComputer
from tools.config_utils import load_yaml

ONNX_EXPORT_VERSION = 5


def change_input_node_name(model, input_names):
    for i, _input in enumerate(model.graph.input):
        input_name = input_names[i]
        for node in model.graph.node:
            for j, name in enumerate(node.input):
                if name == _input.name:
                    node.input[j] = input_name
        _input.name = input_name


def change_output_node_name(model, output_names):
    for i, output in enumerate(model.graph.output):
        output_name = output_names[i]
        for node in model.graph.node:
            for j, name in enumerate(node.output):
                if name == output.name:
                    node.output[j] = output_name
        output.name = output_name


class Resampler(torch.nn.Module):
    def __init__(self, hubert_config, mel_config, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hop_size = mel_config["hop_size"]
        self.resampler = torchaudio.transforms.Resample(mel_config["sample_rate"], hubert_config["sample_rate"]).to(
            device)

    def forward(self,
                waveform,
                ):
        n_frames = waveform.size(-1) // self.hop_size + 1
        return self.resampler(waveform), n_frames


class UnitsAligner(torch.nn.Module):
    def __init__(self, hubert_config, mel_config, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.ratio = (mel_config["hop_size"] / mel_config["sample_rate"]) / (
                hubert_config["hop_size"] / hubert_config["sample_rate"])

    def forward(self,
                units,  # [B, T, C]
                n_frames):
        # alignment
        index = torch.clamp(torch.round(self.ratio * torch.arange(n_frames).to(self.device)).long(),
                            max=units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
        return units_aligned.transpose(1, 2)  # [B, C, T]


class Encoder(torch.nn.Module):
    def __init__(self, hubert_config, melspec_config, hubert_path=None, device='cpu'):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if hubert_path is None:
            hubert_path = hubert_config["onnx_path"]
        self.resampler = Resampler(hubert_config, melspec_config, device)
        self.curves_computer = PowerCurveComputer(melspec_config["window_size"], melspec_config["hop_size"])
        self.hubert = HubertModel.from_pretrained(hubert_path).to(device).eval()
        self.aligner = UnitsAligner(hubert_config, melspec_config, device)

    def forward(self, waveform):
        waveform_16k, n_frames = self.resampler(waveform)
        curves = self.curves_computer(waveform.squeeze(0), n_frames).unsqueeze(0).unsqueeze(0)
        units = self.hubert(waveform_16k)["last_hidden_state"]
        units_aligned = self.aligner(units, n_frames)
        return units_aligned.half().float(), curves.half().float()


class PredictModel(torch.nn.Module):
    def __init__(self, nll_ckpt_path, fa_ckpt_path):
        super().__init__()
        self.nll = LitNonLexicalLabelerTask.load_from_checkpoint(nll_ckpt_path)
        self.fa = LitForcedAlignmentTask.load_from_checkpoint(fa_ckpt_path)

    def forward(self, x, curves):
        return self.nll(x), self.fa(x, curves)


def export_encoder(encoder_folder, _hubert_config, _mel_config, hubert_path=None, device='cpu'):
    if hubert_path is None:
        hubert_path = _hubert_config["model_path"]
    encoder = Encoder(_hubert_config, _mel_config, hubert_path)
    encoder_onnx_path = str(pathlib.Path(encoder_folder) / "encoder.onnx")

    waveform = torch.randn(1, 22050).to(device)
    torch.onnx.export(
        encoder,
        waveform,
        str(encoder_onnx_path),
        opset_version=17,
        input_names=["waveform"],
        output_names=["units_aligned", "curves"],
        dynamic_axes={
            "waveform": {1: "n_samples"},  # (B, T)
            "units_aligned": {2: "n_samples"},  # (B, C, T)
            "curves": {2: "n_frames"},  # (B, N, T)
        }
    )
    onnx_model, check = onnxsim.simplify(encoder_onnx_path, include_subgraph=True)
    assert check, 'Simplified ONNX model could not be validated'
    onnx.save(onnx_model, encoder_onnx_path)
    print(f'Encoder Model saved to: {encoder_onnx_path}')
    return encoder_onnx_path


def export_predict_model(onnx_folder: pathlib.Path, nll_ckpt_path: pathlib.Path, fa_ckpt_path: pathlib.Path,
                         _hubert_config, _melspec_config, device="cpu"):
    _onnx_path = str(pathlib.Path(onnx_folder) / "predict.onnx")
    model = PredictModel(nll_ckpt_path, fa_ckpt_path).to(device)
    model.eval()

    n_frames = 1000
    input_feature = torch.randn((1, _hubert_config["channel"], n_frames), dtype=torch.float32,
                                device=device)  # (B, C, T)
    curves = torch.randn((1, 1, n_frames), dtype=torch.float32, device=device)  # (B, 1, T)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_feature, curves),
            _onnx_path,
            input_names=['input_feature', 'curves'],
            output_names=['cvnt_logits', 'ph_frame_logits', 'ph_edge_logits', 'ctc_logits'],
            dynamic_axes={
                'input_feature': {2: 'n_samples'},  # (B, C, T)
                'curves': {2: 'n_samples'},  # (B, N, T)
                'cvnt_logits': {2: 'n_samples'},  # (B, N, T)
                'ph_frame_logits': {2: 'n_samples'},  # (B, vocab_size, T)
                'ph_edge_logits': {1: 'n_samples'},  # (B, T)
                'ctc_logits': {2: 'n_samples'},  # (B, vocab_size, T)
            },
            opset_version=17
        )
        onnx_model, check = onnxsim.simplify(_onnx_path, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(onnx_model, _onnx_path)
        print(f'Predict Model saved to: {_onnx_path}')

        export_config = {}
        export_config.update(load_yaml(nll_ckpt_path.parent / 'vocab.yaml'))
        export_config.update(load_yaml(fa_ckpt_path.parent / 'vocab.yaml'))
        with open(onnx_folder / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump(export_config, f, ensure_ascii=False, indent=4)

        with open(onnx_folder / 'config.json', 'w', encoding='utf-8') as f:
            json.dump({'hubert_config': _hubert_config, 'mel_spec_config': _melspec_config},
                      f, ensure_ascii=False, indent=4)

        for txt in fa_ckpt_path.parent.rglob('*.txt'):
            shutil.copy(txt, onnx_folder / txt.name)
    return _onnx_path


def merge_model(merged_path, encoder_path, predict_path):
    encoder = onnx.load(encoder_path)
    encoder = onnx.compose.add_prefix(encoder, prefix="encoder/")

    predict_model = onnx.load(predict_path)
    predict_model = onnx.compose.add_prefix(predict_model, prefix="predict/")

    final_model = onnx.compose.merge_models(
        encoder, predict_model,
        io_map=[
            ("encoder/units_aligned", "predict/input_feature"),
            ("encoder/curves", "predict/curves")
        ]
    )

    change_input_node_name(final_model, ["waveform"])
    change_output_node_name(final_model, [
        'cvnt_logits',
        'ph_frame_logits',
        'ph_edge_logits',
        'ctc_logits'
    ])

    onnx.save(final_model, merged_path)

    onnx_model, check = onnxsim.simplify(merged_path, include_subgraph=True)
    assert check, 'Simplified ONNX model could not be validated'
    onnx.save(onnx_model, merged_path)
    print(f"Merged model saved to: {merged_path}")

    os.remove(encoder_path)
    os.remove(predict_path)


@torch.no_grad()
@click.command(help='')
@click.option('--nll_path', '-nll', required=True, type=str, help='Path to the checkpoint')
@click.option('--fa_path', '-fa', required=True, type=str, help='Path to the checkpoint')
@click.option("--hubert_path", "-h", default=None, type=str, help="path to the encoder model")
@click.option('--out_folder', '-o', required=True, metavar='DIR', help='Path to the onnx')
def export(nll_path, fa_path, hubert_path, out_folder):
    assert nll_path is not None, "Checkpoint directory (nll_path) cannot be None"
    assert fa_path is not None, "Checkpoint directory (fa_path) cannot be None"

    nll_path = pathlib.Path(nll_path)
    fa_path = pathlib.Path(fa_path)
    out_folder = pathlib.Path(out_folder)

    assert nll_path.exists(), f"Checkpoint path does not exist: {nll_path}"
    assert (
                nll_path.parent / "config.yaml").exists(), f"Checkpoint folder {nll_path.parent} does not exist: config.yaml"

    assert fa_path.exists(), f"Checkpoint path does not exist: {fa_path}"
    assert (fa_path.parent / "config.yaml").exists(), f"Checkpoint folder {fa_path.parent}does not exist: config.yaml"

    out_folder.mkdir(parents=True, exist_ok=True)
    onnx_path = str(out_folder / "model.onnx")
    assert not os.path.exists(onnx_path), f"Error: The file '{onnx_path}' already exists."

    nll_config = load_yaml(nll_path.parent / "config.yaml")
    fa_config = load_yaml(fa_path.parent / "config.yaml")

    hubert_config = nll_config['hubert_config']
    mel_spec_config = nll_config['mel_spec_config']
    assert hubert_config == fa_config['hubert_config'], f"nll and fa model must have same hubert config."
    assert mel_spec_config == fa_config['mel_spec_config'], f"nll and fa model must have same mel spec config."

    encoder_path = export_encoder(out_folder, hubert_config, mel_spec_config, hubert_path=hubert_path)
    predict_path = export_predict_model(out_folder, nll_path, fa_path, hubert_config, mel_spec_config)

    merged_path = str(out_folder / "model.onnx")
    merge_model(merged_path, encoder_path, predict_path)

    with open(out_folder / "VERSION", "w") as f:
        f.write(str(ONNX_EXPORT_VERSION))


if __name__ == '__main__':
    export()
