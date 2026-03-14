import pathlib
import shutil

import click
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from tools.binarize_util import load_wav, get_curves
from tools.config_utils import load_yaml
from tools.dataset import IndexedDatasetBuilder
from tools.encoder import UnitsEncoder
from tools.get_melspec import MelSpecExtractor
from tools.multiprocess_utils import chunked_multiprocess_run


class BaseBinarizer:
    def __init__(self, binary_config_path):
        self.binary_config = load_yaml(binary_config_path)
        self.mel_spec_config = self.binary_config['mel_spec_config']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_mel_spec = MelSpecExtractor(**self.mel_spec_config, device=self.device)
        self.unitsEncoder = UnitsEncoder(
            self.binary_config['hubert_config'],
            self.binary_config['mel_spec_config'],
            device=self.device
        )

        self.datasets = self.load_datasets()
        self.binary_folder = pathlib.Path(self.binary_config['binary_folder'])
        self.binary_folder.mkdir(parents=True, exist_ok=True)

        self.multiprocess_works = self.binary_config['multiprocess_works']
        self.multiprocess_max_size = self.binary_config['multiprocess_max_size']
        self.multiprocess_start_size = self.binary_config['multiprocess_start_size']

        self.valid_sets = []
        self.valid_set_size = self.binary_config['valid_set_size']

        self.hop_size = self.mel_spec_config["hop_size"]
        self.window_size = self.mel_spec_config["window_size"]
        self.sample_rate = self.mel_spec_config["sample_rate"]
        self.frame_length = self.hop_size / self.sample_rate
        self.max_length = self.binary_config['max_length']

        self.aug_args = self.binary_config['augmentation_args']
        aug_enabled = self.aug_args['enabled']
        self.aug_num = 1 + (
            (self.aug_args['random_pitch_shifting']['num'] + self.aug_args['blank_padding']['num'])
            if aug_enabled else 0
        )

        shutil.copy(binary_config_path, self.binary_folder / 'config.yaml')
        self.export_config(self.binary_folder / 'datasets.yaml',
                           {"aug_num": self.aug_num, "datasets": self.datasets})

    def load_datasets(self):
        datasets_config_paths = self.binary_config["datasets_config_paths"]
        if not isinstance(datasets_config_paths, list):
            raise ValueError(f"{datasets_config_paths} is not a list.")

        valid_keys = {'language', 'label_type', 'raw_data_dir', 'test_prefixes'}
        valid_datasets = []

        for dataset_config_path in datasets_config_paths:
            if not pathlib.Path(dataset_config_path).exists():
                raise FileNotFoundError(f"{dataset_config_path} does not exist.")

            datasets = load_yaml(dataset_config_path)["datasets"]
            for item in datasets:
                if not isinstance(item, dict):
                    raise ValueError(f"{item} is not a dict.")

                missing_keys = valid_keys - item.keys()
                if missing_keys:
                    raise ValueError(f"datasets missing keys {missing_keys}: [{item}]")

                if not pathlib.Path(item['raw_data_dir']).exists():
                    raise FileNotFoundError(f"{item['raw_data_dir']} does not exist.")

                if item['label_type'] not in ['full', 'evaluate']:
                    raise ValueError(f"{item['label_type']} not in ['full','evaluate'].")

                valid_datasets.append(item)

        if not valid_datasets:
            raise ValueError("datasets are empty.")

        return valid_datasets

    @staticmethod
    def export_config(file_path, config):
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

    def binarize(self, prefix, meta_data, binary_data_folder):
        print(f"Binarizing {prefix} set...")
        train = prefix == "train"

        args = [(row, train) for _, row in meta_data.iterrows()]
        builder = IndexedDatasetBuilder(binary_data_folder, aug_num=self.aug_num, prefix=prefix)

        try:
            if self.multiprocess_works > 0 and len(args) > self.multiprocess_start_size:
                for item in tqdm(
                        chunked_multiprocess_run(self.process_item, args,
                                                 num_workers=self.multiprocess_works,
                                                 q_max_size=self.multiprocess_max_size),
                        total=len(args)
                ):
                    if item:
                        builder.add_item(item)
            else:
                for a in tqdm(args):
                    item = self.process_item(*a)
                    if item:
                        builder.add_item(item)
        except KeyboardInterrupt:
            builder.finalize()
            raise

        builder.finalize()
        total_time = sum(builder.aug_wav_lengths)
        print(
            f"Successfully binarized {prefix} set, total time {total_time:.2f}s ({total_time / 3600:.2f}h), saved to {builder.path}")

    def process(self):
        meta_data_df = self.get_meta_data()
        meta_data_df = self.post_process_meta_data(meta_data_df)

        meta_data_evaluate = meta_data_df[meta_data_df["label_type"] == "evaluate"]
        meta_data_df = meta_data_df.drop(meta_data_evaluate.index)

        valid_set_size = int(self.valid_set_size)
        if self.valid_set_size > 0:
            meta_data_valid = (meta_data_df[meta_data_df["label_type"] == "full"]
            .sample(frac=1).iloc[:valid_set_size])
        else:
            meta_data_valid = meta_data_df[meta_data_df["validation"]]

        meta_data_train = meta_data_df.drop(meta_data_valid.index).reset_index(drop=True)
        meta_data_valid = meta_data_valid.reset_index(drop=True)
        meta_data_evaluate = meta_data_evaluate.reset_index(drop=True)

        if len(meta_data_valid) == 0:
            raise ValueError("No valid data found.")

        for name, data in [
            ("evaluate", meta_data_evaluate),
            ("valid", meta_data_valid),
            ("train", meta_data_train),
        ]:
            self.binarize(name, data, self.binary_folder)

    def process_item(self, **kwargs):
        pass

    def get_meta_data(self):
        print("Loading metadata...")
        meta_data_frames = []

        for dataset in self.datasets:
            test_prefixes = dataset['test_prefixes']
            tuple_prefixes = tuple(filter(None, test_prefixes)) if test_prefixes else ()

            raw_data_dir = pathlib.Path(dataset['raw_data_dir'])
            csv_path = raw_data_dir / "transcriptions.csv"
            wav_folder = raw_data_dir / "wavs"

            if not (csv_path.exists() and wav_folder.exists()):
                raise FileNotFoundError(f"{csv_path} or {wav_folder} does not exist.")

            df = pd.read_csv(csv_path, dtype=str)
            original_count = len(df)

            df['name'] = df['name'].astype(str)
            invalid_names = df['name'].apply(lambda x: not isinstance(x, str) or x == 'nan' or x.strip() == '')

            if invalid_names.any():
                invalid_count = invalid_names.sum()
                print(f"Warning: Found {invalid_count} invalid name(s) in {csv_path}. Ignored these rows.")
                print(f"Invalid names: {df[invalid_names]['name'].tolist()}")
                df = df[~invalid_names].copy()
                print(f"Ignored {invalid_count} invalid rows. Remaining: {len(df)}/{original_count}")

            if len(df) == 0:
                print(f"Error: No valid data remaining in {csv_path} after cleaning!")
                continue

            required_cols = {'ph_seq', 'ph_dur'}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(f"{csv_path} missing columns: {missing}")

            df['language'] = dataset['language']
            df["label_type"] = dataset['label_type']
            df["wav_path"] = df["name"].apply(lambda name: str(wav_folder / f"{name}.wav"))
            df["validation"] = df["name"].str.startswith(tuple_prefixes) if tuple_prefixes else False

            meta_data_frames.append(df)

        if not meta_data_frames:
            raise ValueError("No valid metadata found.")

        meta_data_df = pd.concat(meta_data_frames, ignore_index=True)

        meta_data_df["ph_seq"] = meta_data_df["ph_seq"].apply(
            lambda s: s.split() if isinstance(s, str) else []
        )
        meta_data_df["ph_dur"] = meta_data_df["ph_dur"].apply(
            lambda s: list(map(float, s.split())) if isinstance(s, str) else []
        )

        meta_data_df = meta_data_df.sort_values(by="label_type").reset_index(drop=True)
        print(f"Final metadata contains {len(meta_data_df)} valid rows.")
        return meta_data_df

    def post_process_meta_data(self, meta_data_df):
        return meta_data_df


class NonLexicalLabelBinarizer(BaseBinarizer):
    def __init__(self, binary_config):
        super().__init__(binary_config)
        self.non_lexical_phonemes = self.binary_config['non_lexical_phonemes']

        if len(self.non_lexical_phonemes) < 1:
            raise ValueError("non_lexical_phonemes must have at least one phoneme.")

        self.non_lexical_phonemes_dict = {"None": 0} | {
            ph: i + 1 for i, ph in enumerate(self.non_lexical_phonemes)
        }

        self.vocab = self.get_vocab()
        self.export_config(self.binary_folder / 'vocab.yaml', self.vocab)

    def get_vocab(self):
        print("Generating vocab...")
        return {
            "non_lexical_phonemes": self.non_lexical_phonemes,
            "non_lexical_phonemes_dict": self.non_lexical_phonemes_dict
        }

    def make_non_lexical_ph_data(self, frames, ph_id_seq, ph_duration):
        if not ph_id_seq:
            return np.zeros((len(self.vocab) + 1, frames), dtype=np.int32), []

        ph_id_seq = np.array(ph_id_seq, dtype=np.int32)
        ph_dur = np.array(ph_duration, dtype=np.float32)

        ph_frame = ph_dur.cumsum() / self.frame_length
        ph_frame_int = np.round(np.concatenate(([0], ph_frame))).astype(np.int32)
        ph_frame_int = np.clip(ph_frame_int, 0, frames)

        num_phones = len(self.non_lexical_phonemes) + 1
        non_lexical_target = np.zeros((num_phones, frames), dtype=np.int32)
        non_lexical_intervals = []

        for i, ph_id in enumerate(ph_id_seq):
            start, end = ph_frame_int[i], ph_frame_int[i + 1]
            if start < end:
                non_lexical_target[ph_id, start:end] = 1
                if ph_id > 0:
                    non_lexical_intervals.append([start, end])

        return non_lexical_target, np.array(non_lexical_intervals)

    @torch.no_grad()
    def process_item(self, _item, train):
        try:
            wav_path = _item['wav_path']
            if not pathlib.Path(wav_path).exists():
                print(f"Skipping {wav_path}, because it doesn't exist")
                return None

            waveform, wav_length, n_frames = load_wav(
                wav_path, self.sample_rate, self.hop_size, self.device
            )

            if wav_length > self.max_length:
                print(f"Item {wav_path} has length {wav_length}s, too long, skip.")
                return None

            if not _item.ph_seq or len(_item.ph_dur) != len(_item.ph_seq):
                return None

            non_lexical_target, non_lexical_intervals = self.make_non_lexical_ph_data(
                n_frames, _item.non_lexical_phonemes_id_seq, _item.ph_dur
            )

            if non_lexical_target is None:
                print(f"Skipping {wav_path}, make non_lexical_ph data failed.")
                return None

            units = self.unitsEncoder.forward(
                waveform.unsqueeze(0), self.sample_rate, self.hop_size,
                aug=self.binary_config['augmentation_args']['enabled'] and train,
                aug_args=self.binary_config['augmentation_args']
            )

            mel_spec = self.get_mel_spec(waveform).cpu().numpy() if not train else np.array([[[0]]])
            B, T, C = units.shape

            if not (B == (self.aug_num if train else 1) and T > 0 and T == n_frames):
                raise ValueError(f"Shape mismatch: B={B}, T={T}, n_frames={n_frames}")

            repeat_vals = {
                'name': [str(_item["name"])] * B,
                'input_feature': units.transpose(1, 2).cpu().numpy().astype(np.float16),
                'mel_spec': np.repeat(mel_spec, B, axis=0).astype(np.float32),
                "non_lexical_target": np.repeat(non_lexical_target[np.newaxis], B, axis=0).astype(np.int32),
                "non_lexical_intervals": np.repeat(non_lexical_intervals[np.newaxis], B, axis=0).astype(np.int32),
                "wav_length": np.full(B, wav_length, dtype=np.float32)
            }

            return repeat_vals

        except Exception as e:
            print(f"error: {_item.get('name', 'unknown')}: {e}")
            return None

    def post_process_meta_data(self, meta_data_df):
        meta_data_df["non_lexical_phonemes_id_seq"] = meta_data_df["ph_seq"].apply(
            lambda seq: [self.vocab['non_lexical_phonemes_dict'].get(ph, 0) for ph in seq]
        )
        return meta_data_df


class ForcedAlignmentBinarizer(BaseBinarizer):
    def __init__(self, binary_config):
        super().__init__(binary_config)
        self.extra_phonemes = self.binary_config['extra_phonemes']
        self.silent_phonemes = self.binary_config['silent_phonemes']
        self.language_prefix = self.binary_config['language_prefix']
        self.dictionaries = self.binary_config['dictionaries']
        self.merged_phoneme_groups = (
            self.binary_config['merged_phoneme_groups']
            if self.binary_config['merged_phoneme'] else []
        )
        self.hubert_channel = self.binary_config['hubert_config']["channel"]

        self.vocab = self.get_vocab()
        self.export_config(self.binary_folder / 'vocab.yaml', self.vocab)

        for dict_path in self.dictionaries.values():
            shutil.copy(dict_path, self.binary_folder)

    def get_vocab(self):
        print("Generating vocab...")
        dataset_phonemes = set(self.extra_phonemes)

        for dataset in self.datasets:
            csv_path = pathlib.Path(dataset["raw_data_dir"]) / "transcriptions.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"{csv_path} does not exist.")

            df = pd.read_csv(csv_path)
            ph_seq = set(" ".join(df["ph_seq"]).split())

            dataset_phonemes.update(
                ph if ph in self.silent_phonemes or "/" in ph or not self.language_prefix
                else f"{dataset['language']}/{ph}"
                for ph in ph_seq
            )

        dict_phonemes = set()
        for lang, dict_path in self.dictionaries.items():
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    _word, _phonemes = line.strip().split("\t", 1)
                    for ph in _phonemes.split():
                        if '/' in ph:
                            raise ValueError(f"Invalid phoneme '{ph}' in {dict_path}: contains '/'")

                        dict_phonemes.add(
                            ph if ph in self.silent_phonemes or "/" in ph or not self.language_prefix
                            else f"{lang}/{ph}"
                        )

        dataset_phonemes -= set(self.silent_phonemes)
        dataset_phonemes = sorted(["SP", *dataset_phonemes])

        self.merged_phoneme_groups.insert(0, list({"SP", *self.silent_phonemes}))

        vocab = dict(zip(dataset_phonemes, range(len(dataset_phonemes))))
        for i, group in enumerate(self.merged_phoneme_groups):
            vocab.update({ph: i for ph in group})

        vocab.update({ph: len(vocab) for ph in dataset_phonemes if ph not in vocab})

        vocab_dict = {
            "vocab": vocab,
            "vocab_size": len(dataset_phonemes),
            "language_prefix": self.language_prefix,
            "silent_phonemes": list({"SP", *self.silent_phonemes}),
            "merged_phoneme_groups": self.merged_phoneme_groups,
            "dictionaries": {k: pathlib.Path(v).name for k, v in self.dictionaries.items()},
        }

        print(f"vocab_size is {len(dataset_phonemes)}:")
        only_in_dataset = set(dataset_phonemes) - dict_phonemes - set(self.silent_phonemes)
        only_in_dict = dict_phonemes - set(dataset_phonemes) - set(self.silent_phonemes)

        if only_in_dataset:
            print(f"+ {sorted(only_in_dataset)}")
        if only_in_dict:
            print(f"- {sorted(only_in_dict)}")

        return vocab_dict

    def make_ph_data(self, vocab, frames, raw_ph_id_seq, raw_ph_dur):
        ph_id_seq = np.array(raw_ph_id_seq, dtype=np.int32)
        not_sp_idx = ph_id_seq != 0
        ph_id_seq = ph_id_seq[not_sp_idx]

        # ph_edge: [T]
        ph_dur = np.array(raw_ph_dur).astype("float32")
        ph_time = np.array(np.concatenate(([0], ph_dur))).cumsum()
        ph_frame = ph_time / self.frame_length
        ph_interval = np.stack((ph_frame[:-1], ph_frame[1:]))
        ph_time = ph_time[:-1]
        ph_time = ph_time[not_sp_idx]

        ph_interval = ph_interval[:, not_sp_idx]
        ph_id_seq = ph_id_seq
        ph_frame = np.unique(ph_interval.flatten())
        if ph_frame[-1] >= frames:
            ph_frame = ph_frame[:-1]

        if len(ph_id_seq) <= 0:
            return None, None, None, None, None

        ph_edge = np.zeros([frames], dtype=np.float32)
        if len(ph_id_seq) > 0:
            if ph_frame[-1] + 0.5 > frames:
                ph_frame = ph_frame[:-1]
            if ph_frame[0] - 0.5 < 0:
                ph_frame = ph_frame[1:]
            ph_time_int = np.round(ph_frame).astype("int32")
            ph_time_fractional = ph_frame - ph_time_int

            ph_edge[ph_time_int] = 0.5 + ph_time_fractional
            ph_edge[ph_time_int - 1] = 0.5 - ph_time_fractional

        # ph_frame: [T]
        ph_frame = np.zeros(frames, dtype=np.int32)
        if len(ph_id_seq) > 0:
            for ph_id, st, ed in zip(
                    ph_id_seq, ph_interval[0], ph_interval[1]
            ):
                if st < 0:
                    st = 0
                if ed > frames:
                    ed = frames
                ph_frame[int(np.round(st)): int(np.round(ed))] = ph_id

        ph_mask = np.zeros(vocab["vocab_size"], dtype=np.int32)
        if len(ph_id_seq) > 0:
            ph_mask[ph_id_seq] = 1
        ph_mask[0] = 1

        return ph_id_seq, ph_edge, ph_frame, ph_mask, ph_time

    @torch.no_grad()
    def process_item(self, _item, train):
        try:
            wav_path = _item["wav_path"]
            if not pathlib.Path(wav_path).exists():
                print(f"Skipping {wav_path}, because it doesn't exist")
                return None

            waveform, wav_length, n_frames = load_wav(
                wav_path, self.sample_rate, self.hop_size, self.device
            )

            if wav_length > self.max_length:
                print(f"Item {wav_path} has length {wav_length}s, too long, skip.")
                return None

            curves = get_curves(waveform, n_frames, self.window_size, self.hop_size)

            if not _item.ph_id_seq or len(_item.ph_dur) != len(_item.ph_id_seq):
                return None

            ph_data = self.make_ph_data(self.vocab, n_frames, _item.ph_id_seq, _item.ph_dur)
            if ph_data[0] is None:
                print(f"Skipping {wav_path}, make ph data failed.")
                return None

            ph_id_seq, ph_edge, ph_frame, ph_mask, ph_time = ph_data

            units = self.unitsEncoder.forward(
                waveform.unsqueeze(0), self.sample_rate, self.hop_size,
                aug=self.binary_config['augmentation_args']['enabled'] and train,
                aug_args=self.binary_config['augmentation_args']
            )

            mel_spec = self.get_mel_spec(waveform).cpu().numpy() if not train else np.array([[[0]]])
            B, T, C = units.shape

            if not (B == (self.aug_num if train else 1) and T > 0 and T == n_frames and C == self.hubert_channel):
                raise ValueError(
                    f"Shape mismatch: B={B}, T={T}, C={C}, n_frames={n_frames}, hubert_channel={self.hubert_channel}")

            repeat_data = {
                'name': [str(_item["name"])] * B,
                'input_feature': units.transpose(1, 2).cpu().numpy().astype(np.float16),  # [B, C, T]
                'curves': np.repeat(curves.cpu().numpy(), B, axis=0).astype(np.float16),  # [B, 1, T]
                'mel_spec': np.repeat(mel_spec, B, axis=0).astype(np.float32),  # [B, C, T]
                'ph_id_seq': np.repeat([ph_id_seq], B, axis=0).astype(np.int32),  # [B, N]
                'ph_edge': np.repeat([ph_edge], B, axis=0).astype(np.float32),  # [B, T]
                'ph_frame': np.repeat([ph_frame], B, axis=0).astype(np.int32),  # [B, T]
                'ph_mask': np.repeat([ph_mask], B, axis=0).astype(np.int32),  # [B, T]
                'ph_time': np.repeat([ph_time], B, axis=0).astype(np.float32),  # [B, N]
                'ph_time_raw': np.concatenate(([0], _item.ph_dur)).cumsum()[:-1].astype(np.float32),  # [B, N]
                'ph_seq_raw': _item.ph_seq,
                'ph_seq': [[ph for ph in _item.ph_seq if self.vocab["vocab"][ph] != 0]] * B,
                "wav_length": np.full(B, wav_length, dtype=np.float32)
            }

            return repeat_data

        except Exception as e:
            print(f"error: {_item.get('name', 'unknown')}: {e}")
            return None

    def post_process_meta_data(self, meta_data_df):
        def format_phonemes(row):
            return [
                ph if ph in self.silent_phonemes or "/" in ph or not self.language_prefix
                else f"{row['language']}/{ph}"
                for ph in row["ph_seq"]
            ]

        meta_data_df["ph_seq"] = meta_data_df.apply(format_phonemes, axis=1)
        meta_data_df["ph_id_seq"] = meta_data_df["ph_seq"].apply(
            lambda seq: [self.vocab['vocab'][ph] for ph in seq]
        )
        return meta_data_df


@click.command()
@click.option("--config", "-c", type=str, required=True, help="binarize config path")
@click.option("--model", "-m", type=str, required=True,
              help="model type: nll[non_lexical_labeler] fa[forced_alignment]")
def binarize(config: str, model: str):
    if model not in ['nll', 'fa']:
        raise ValueError("model type must be 'nll' or 'fa'")

    binarizer_class = NonLexicalLabelBinarizer if model == "nll" else ForcedAlignmentBinarizer
    binarizer_class(config).process()


if __name__ == "__main__":
    binarize()
