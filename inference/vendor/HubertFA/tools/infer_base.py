import os
import pathlib
import warnings
from collections import defaultdict

import librosa
import numpy as np
from tqdm import tqdm

from tools.align_word import WordList, Word, Phoneme
from tools.decoder import AlignmentDecoder, NonLexicalDecoder
from tools.export_tool import Exporter
from tools.g2p import DictionaryG2P, PhonemeG2P


def find_all_duplicate_phonemes(ph_list):
    if len(ph_list) == 1:
        return [0]
    index_dict = defaultdict(list)
    for idx, sublist in enumerate(ph_list):
        key = tuple(sublist)
        index_dict[key].append(idx)

    duplicate_phonemes = {key: indices for key, indices in index_dict.items() if len(indices) > 1}

    if not duplicate_phonemes:
        raise Exception("No duplicate groups")

    sorted_groups = sorted(duplicate_phonemes.items(), key=lambda x: (-len(x[1]), -len(x[0])))
    best_key, best_indices = sorted_groups[0]
    return best_indices


def median_abs_deviation(x, axis=0, center=np.median, scale=1.0):
    """
    Compute the median absolute deviation of the data along the given axis.

    This is a NumPy implementation of scipy.stats.median_abs_deviation.
    """
    if isinstance(scale, str):
        if scale.lower() == 'normal':
            scale = 0.6744897501960817
        else:
            raise ValueError(f"{scale} is not a valid scale value.")

    x = np.asarray(x)

    if x.size == 0:
        if axis is None:
            return np.nan
        if axis is not None:
            nan_shape = list(x.shape)
            del nan_shape[axis]
            nan_shape = tuple(nan_shape)
            if nan_shape == ():
                return np.nan
            return np.full(nan_shape, np.nan)
        return np.nan

    contains_nan = np.isnan(x).any()

    if contains_nan:
        if axis is None:
            return np.nan
        else:
            result_shape = list(x.shape)
            if axis is not None:
                del result_shape[axis]
            result = np.full(tuple(result_shape), np.nan)
            return result / scale
    else:
        if axis is None:
            med = center(x)
            mad = np.median(np.abs(x - med))
        else:
            med = center(x, axis=axis)
            med_expanded = np.expand_dims(med, axis=axis)
            mad = np.median(np.abs(x - med_expanded), axis=axis)

    return mad / scale


def remove_outliers_per_position(data_series, threshold=1.5):
    """
    使用中位数绝对偏差(MAD)方法去除离群值
    参数:
        data_series -- 每个位置的时间戳列表
        threshold -- MAD阈值
    返回: 处理后的平均值列表
    """
    processed_values = []
    for position_values in data_series:
        if not position_values:
            processed_values.append(0.0)
            continue
        med = np.median(position_values)
        mad_val = median_abs_deviation(position_values)

        # 处理MAD=0的情况（所有值相同）
        if mad_val == 0:
            processed_values.append(med)
            continue

        # 计算修正Z-score
        z_scores = np.abs((np.array(position_values) - med) / (mad_val * 1.4826))

        # 分离有效值和离群值
        retained_values = []
        filtered_out = []
        for x, z in zip(position_values, z_scores):
            if z <= threshold:
                retained_values.append(x)
            else:
                filtered_out.append(x)

        if len(retained_values) > 0:
            final_value = np.mean(retained_values)
        else:
            final_value = med

        processed_values.append(final_value)
    return processed_values


class InferenceBase:
    def __init__(self):
        self.vocab = None
        self.mel_cfg = None
        self.vocab_folder = None

        self.nll_decoder = None
        self.fa_decoder = None
        self.dataset = []
        self.predictions = []

    def load_config(self):
        pass

    def load_model(self):
        pass

    def _infer(self, padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx, wav_length, non_lexical_phonemes):
        return NotImplementedError

    def init_decoder(self):
        self.nll_decoder = NonLexicalDecoder(vocab=self.vocab,
                                             class_names=['None', *self.vocab['non_lexical_phonemes']],
                                             sample_rate=self.mel_cfg["sample_rate"], hop_size=self.mel_cfg["hop_size"])
        self.fa_decoder = AlignmentDecoder(vocab=self.vocab, sample_rate=self.mel_cfg["sample_rate"],
                                           hop_size=self.mel_cfg["hop_size"])

    def get_dataset(self, wav_folder, language, g2p="dictionary", dictionary_path=None, in_format="lab"):
        if dictionary_path is None:
            dictionary_path = self.vocab_folder / self.vocab["dictionaries"].get(language, "")
        language = language if self.vocab['language_prefix'] else None

        if g2p == "dictionary":
            assert os.path.exists(dictionary_path), f"{pathlib.Path(dictionary_path).absolute()} does not exist."
            g2p = DictionaryG2P(language, dictionary_path)
        elif g2p == "phoneme":
            g2p = PhonemeG2P(language)
        else:
            raise f"g2p - {g2p} is not supported, which should be 'dictionary' or 'phoneme'."

        wav_paths = pathlib.Path(wav_folder).rglob("*.wav")
        for wav_path in wav_paths:
            try:
                lab_path = wav_path.with_suffix("." + in_format)
                if lab_path.exists():
                    with open(lab_path, "r", encoding="utf-8") as f:
                        lab_text = f.read().strip()
                    ph_seq, word_seq, ph_idx_to_word_idx = g2p(lab_text)
                    self.dataset.append((wav_path, ph_seq, word_seq, ph_idx_to_word_idx))
                else:
                    warnings.warn(f"{pathlib.Path(wav_path).absolute()} does not exist.")
            except Exception as e:
                e.args = (f" Error when processing {wav_path}: {e} ",)
        print(f"Loaded {len(self.dataset)} samples.")

    def infer(self, non_lexical_phonemes, pad_times=1, pad_length=5):
        non_lexical_phonemes = [ph.strip() for ph in non_lexical_phonemes.split(",") if ph.strip()]
        assert set(non_lexical_phonemes).issubset(set(self.vocab['non_lexical_phonemes'])), \
            f"The non_lexical_phonemes contain elements that are not included in the vocab."

        pad_lengths = [round(pad_length / pad_times * i, 1) for i in range(0, pad_times)] if pad_times > 1 else [0]

        for i in tqdm(range(len(self.dataset)), desc="Processing", unit="it"):
            wav_path, ph_seq, word_seq, ph_idx_to_word_idx = self.dataset[i]

            # Load and resample audio
            wav, sr = librosa.load(wav_path, sr=self.mel_cfg['sample_rate'], mono=True)
            wav_length = len(wav) / self.mel_cfg['sample_rate']

            words_list: list[WordList] = []
            for pl in pad_lengths:
                padded_samples = int(pl * self.mel_cfg['sample_rate'])
                padded_frames = int(padded_samples / self.mel_cfg['hop_size'])
                padded_wav = np.pad(wav, (padded_samples, 0), mode='constant', constant_values=0)

                words, non_lexical_words = self._infer(padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx,
                                                       wav_length, non_lexical_phonemes)
                for _words in non_lexical_words:
                    for word in _words:
                        words.add_AP(word)
                words.clear_language_prefix()
                words_list.append(words)

            ph_list = [words.phonemes for words in words_list]
            words_list = [words_list[i] for i in find_all_duplicate_phonemes(ph_list)]

            phonemes_all = []
            result_word = WordList()
            for w_idx in range(len(words_list[0])):
                phonemes = []
                for ph_idx in range(len(words_list[0][w_idx].phonemes)):
                    ph_start = \
                        remove_outliers_per_position([[words[w_idx].phonemes[ph_idx].start for words in words_list]])[0]
                    ph_end = \
                        remove_outliers_per_position([[words[w_idx].phonemes[ph_idx].end for words in words_list]])[0]
                    ph_start = max(ph_start, phonemes_all[-1].end if len(phonemes_all) > 0 else 0)
                    ph_end = max(ph_start + 0.0001, ph_end)
                    phonemes.append(Phoneme(ph_start, ph_end, words_list[0][w_idx].phonemes[ph_idx].text))
                    phonemes_all.append(Phoneme(ph_start, ph_end, words_list[0][w_idx].phonemes[ph_idx].text))
                word = Word(phonemes[0].start, phonemes[-1].end, words_list[0][w_idx].text)
                for ph in phonemes:
                    word.append_phoneme(ph)
                result_word.append(word)
            result_word.fill_small_gaps(wav_length)
            result_word.add_SP(wav_length)
            warning_log = result_word.log()
            if warning_log:
                warnings.warn(warning_log)
            self.predictions.append((wav_path, wav_length, result_word))

    def export(self, output_folder, output_format=None):
        if output_format is None:
            output_format = ['textgrid']
        Exporter(self.predictions, output_folder).export(output_format)
        print("Output files are saved to the same folder as the input wav files.")
