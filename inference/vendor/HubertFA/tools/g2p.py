import pathlib
import warnings


class BaseG2P:
    def __init__(self, language):
        self.language = language

    def _g2p(self, input_text):
        # input text, return phoneme sequence, word sequence, and phoneme index to word index mapping
        # ph_seq: list of phonemes, SP is the silence phoneme.
        # word_seq: list of words.
        # ph_idx_to_word_idx: ph_idx_to_word_idx[i] = j means the i-th phoneme belongs to the j-th word.
        # if ph_idx_to_word_idx[i] = -1, the i-th phoneme is a silence phoneme.
        # example: ph_seq = ['SP', 'ay', 'SP', 'ae', 'm', 'SP', 'ah', 'SP', 's', 't', 'uw', 'd', 'ah', 'n', 't', 'SP']
        #          word_seq = ['I', 'am', 'a', 'student']
        #          ph_idx_to_word_idx = [-1, 0, -1, 1, 1, -1, 2, -1, 3, 3, 3, 3, 3, 3, 3, -1]
        raise NotImplementedError

    def __call__(self, text):
        ph_seq, word_seq, ph_idx_to_word_idx = self._g2p(text)

        # The first and last phonemes should be `SP`,
        # and there should not be more than two consecutive `SP`s at any position.
        assert ph_seq[0] == "SP" and ph_seq[-1] == "SP"
        assert all(
            ph_seq[i] != "SP" or ph_seq[i + 1] != "SP" for i in range(len(ph_seq) - 1)
        )
        ph_seq = [f"{self.language}/{ph}" if self.language is not None and ph != "SP" else ph for ph in ph_seq]
        return ph_seq, word_seq, ph_idx_to_word_idx


class PhonemeG2P(BaseG2P):
    def __init__(self, language):
        super().__init__(language)

    def _g2p(self, input_text):
        _word_seq = input_text.strip().split(" ")
        _word_seq = [ph for ph in _word_seq if ph != "SP"]
        _ph_seq = ["SP"]
        _ph_idx_to_word_idx = [-1]
        for word_idx, word in enumerate(_word_seq):
            _ph_seq.append(word)
            _ph_idx_to_word_idx.append(word_idx)
            _ph_seq.append("SP")
            _ph_idx_to_word_idx.append(-1)
        return _ph_seq, _word_seq, _ph_idx_to_word_idx


class DictionaryG2P(BaseG2P):
    def __init__(self, language, dict_path: str | pathlib.Path):
        super().__init__(language)
        with open(dict_path, "r") as f:
            dictionary = f.read().strip().split("\n")
        self.dictionary = {
            item.split("\t")[0].strip(): item.split("\t")[1].strip().split(" ")
            for item in dictionary
        }

    def _g2p(self, input_text):
        word_seq_raw = input_text.strip().split(" ")
        _word_seq = []
        word_seq_idx = 0
        _ph_seq = ["SP"]
        _ph_idx_to_word_idx = [-1]
        for word in word_seq_raw:
            if word not in self.dictionary:
                warnings.warn(f"Word '{word}' is not in the dictionary. Ignored.")
                continue
            _word_seq.append(word)
            phones = self.dictionary[word]
            for i, ph in enumerate(phones):
                if (i == 0 or i == len(phones) - 1) and ph == "SP":
                    warnings.warn(
                        f"The first or last phoneme of word {word} is SP, which is not allowed. "
                        "Please check your dictionary."
                    )
                    continue
                _ph_seq.append(ph)
                _ph_idx_to_word_idx.append(word_seq_idx)
            if _ph_seq[-1] != "SP":
                _ph_seq.append("SP")
                _ph_idx_to_word_idx.append(-1)
            word_seq_idx += 1

        return _ph_seq, _word_seq, _ph_idx_to_word_idx
