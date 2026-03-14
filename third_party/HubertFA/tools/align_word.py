import warnings
from dataclasses import dataclass, field


@dataclass
class Phoneme:
    start: float
    end: float
    text: str

    def __init__(self, start: float, end: float, text: str):
        self.start = max(0.0, start)
        self.end = end
        self.text = text

        if not (self.start < self.end):
            error_msg = f"Phoneme Invalid: text={self.text} start={self.start}, end={self.end}"
            raise ValueError(error_msg)


@dataclass
class Word:
    start: float
    end: float
    text: str
    phonemes: list[Phoneme] = field(default_factory=list)

    def __init__(self, start: float, end: float, text: str, init_phoneme: bool = False):
        self.start = max(0.0, start)
        self.end = end
        self.text = text

        if not (self.start < self.end):
            error_msg = f"Word Invalid: text={self.text} start={self.start}, end={self.end}"
            raise ValueError(error_msg)

        self.phonemes: list[Phoneme] = []
        if init_phoneme:
            self.phonemes.append(Phoneme(self.start, self.end, self.text))

    @property
    def dur(self) -> float:
        return self.end - self.start

    def add_phoneme(self, phoneme, log_list: list = None):
        if phoneme.start == phoneme.end:
            warning_msg = f"{phoneme.text} phoneme长度为0，非法"
            if log_list is not None:
                log_list.append(f"WARNING: {warning_msg}")
            else:
                warnings.warn(warning_msg)
            return
        if phoneme.start >= self.start and phoneme.end <= self.end:
            self.phonemes.append(phoneme)
        else:
            warning_msg = f"{phoneme.text}: phoneme边界超出word，添加失败"
            if log_list is not None:
                log_list.append(f"WARNING: {warning_msg}")
            else:
                warnings.warn(warning_msg)

    def append_phoneme(self, phoneme, log_list: list = None):
        if phoneme.start == phoneme.end:
            warning_msg = f"{phoneme.text} phoneme长度为0，非法"
            if log_list is not None:
                log_list.append(f"WARNING: {warning_msg}")
            else:
                warnings.warn(warning_msg)
            return
        if len(self.phonemes) == 0:
            if phoneme.start == self.start:
                self.phonemes.append(phoneme)
                self.end = phoneme.end
            else:
                warning_msg = f"{phoneme.text}: phoneme左边界超出word，添加失败"
                if log_list is not None:
                    log_list.append(f"WARNING: {warning_msg}")
                else:
                    warnings.warn(warning_msg)
        else:
            if phoneme.start == self.phonemes[-1].end:
                self.phonemes.append(phoneme)
                self.end = phoneme.end
            else:
                warning_msg = f"{phoneme.text}: phoneme添加失败"
                if log_list is not None:
                    log_list.append(f"WARNING: {warning_msg}")
                else:
                    warnings.warn(warning_msg)

    def move_start(self, new_start, log_list: list = None):
        if 0 <= new_start < self.phonemes[0].end:
            self.start = new_start
            self.phonemes[0].start = new_start
        else:
            warning_msg = f"{self.text}: start >= first_phone_end，无法调整word边界"
            if log_list is not None:
                log_list.append(f"WARNING: {warning_msg}")
            else:
                warnings.warn(warning_msg)

    def move_end(self, new_end, log_list: list = None):
        if new_end > self.phonemes[-1].start >= 0:
            self.end = new_end
            self.phonemes[-1].end = new_end
        else:
            warning_msg = f"{self.text}: new_end <= first_phone_start，无法调整word边界"
            if log_list is not None:
                log_list.append(f"WARNING: {warning_msg}")
            else:
                warnings.warn(warning_msg)


class WordList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self._log = []  # 新增：日志列表

    def _add_log(self, message: str):
        """内部方法：添加日志信息"""
        self._log.append(message)

    def log(self) -> str:
        """将日志输出为字符串"""
        return "\n".join(self._log)

    def clear_log(self):
        """清空日志"""
        self._log.clear()

    def overlapping_words(self, new_word: Word):
        overlapping_words = []
        for word in self:
            if not isinstance(word, Word):
                continue
            if not (new_word.end <= word.start or new_word.start >= word.end):
                overlapping_words.append(word)
        return overlapping_words

    def append(self, word: Word):
        if len(word.phonemes) == 0:
            warning_msg = f"{word}: phones为空，非法word"
            self._add_log(f"WARNING: {warning_msg}")
            return

        if len(self) == 0:
            super().append(word)
            return

        if not self.overlapping_words(word):
            super().append(word)
        else:
            warning_msg = f"{word}: 区间重叠，无法添加word"
            self._add_log(f"WARNING: {warning_msg}")

    @staticmethod
    def remove_overlapping_intervals(raw_interval, remove_interval):
        r_start, r_end = raw_interval
        m_start, m_end = remove_interval

        if not (r_start < r_end):
            raise ValueError(f"raw_interval.start must be smaller than raw_interval.end")
        if not (m_start < m_end):
            raise ValueError(f"remove_interval.start must be smaller than remove_interval.end")

        overlap_start = max(r_start, m_start)
        overlap_end = min(r_end, m_end)

        if overlap_start >= overlap_end:
            return [raw_interval]

        result = []
        if r_start < overlap_start:
            result.append((r_start, overlap_start))

        if overlap_end < r_end:
            result.append((overlap_end, r_end))

        return result

    def add_AP(self, new_word: Word, min_dur=0.1):
        try:
            if len(new_word.phonemes) == 0:
                warning_msg = f"{new_word.text} phonemes为空，非法word"
                self._add_log(f"WARNING: {warning_msg}")
                return

            if len(self) == 0:
                self.append(new_word)
                return

            overlapping = self.overlapping_words(new_word)
            if not overlapping:
                self.append(new_word)
                self.sort(key=lambda w: w.start)
                return

            ap_intervals = [(new_word.start, new_word.end)]
            for word in self:
                temp_res = []
                for _ap in ap_intervals:
                    temp_res.extend(self.remove_overlapping_intervals(_ap, (word.start, word.end)))
                ap_intervals = temp_res
            ap_intervals = [_ap for _ap in ap_intervals if _ap[1] - _ap[0] >= min_dur]

            for _ap in ap_intervals:
                try:
                    self.append(Word(_ap[0], _ap[1], new_word.text, True))
                except ValueError as e:
                    self._add_log(f"ERROR: {e}")
                    continue

            self.sort(key=lambda w: w.start)
        except Exception as e:
            self._add_log(f"ERROR in add_AP: {e}")

    def fill_small_gaps(self, wav_length: float, gap_length: float = 0.1):
        try:
            if self[0].start < 0:
                self[0].start = 0

            if self[0].start > 0:
                if abs(self[0].start) < gap_length < self[0].dur:
                    self[0].move_start(0, self._log)

            if self[-1].end >= wav_length - gap_length:
                self[-1].move_end(wav_length, self._log)

            for i in range(1, len(self)):
                if 0 < self[i].start - self[i - 1].end <= gap_length:
                    self[i - 1].move_end(self[i].start, self._log)
        except Exception as e:
            self._add_log(f"ERROR in fill_small_gaps: {e}")

    def add_SP(self, wav_length, add_phone="SP"):
        try:
            words_res = WordList()
            words_res._log = self._log  # 共享日志

            if self[0].start > 0:
                try:
                    words_res.append(Word(0, self[0].start, add_phone, init_phoneme=True))
                except ValueError as e:
                    self._add_log(f"ERROR: {e}")

            words_res.append(self[0])
            for i in range(1, len(self)):
                word = self[i]
                if word.start > words_res[-1].end:
                    try:
                        words_res.append(Word(words_res[-1].end, word.start, add_phone, init_phoneme=True))
                    except ValueError as e:
                        self._add_log(f"ERROR: {e}")
                words_res.append(word)

            if self[-1].end < wav_length:
                try:
                    words_res.append(Word(self[-1].end, wav_length, add_phone, init_phoneme=True))
                except ValueError as e:
                    self._add_log(f"ERROR: {e}")

            self.clear()
            self.extend(words_res)
            self.check()
        except Exception as e:
            self._add_log(f"ERROR in add_SP: {e}")

    @property
    def phonemes(self):
        phonemes = []
        for word in self:
            phonemes.extend([ph.text for ph in word.phonemes])
        return phonemes

    @property
    def intervals(self):
        return [[word.start, word.end] for word in self]

    def clear_language_prefix(self):
        for word in self:
            for phoneme in word.phonemes:
                phoneme.text = phoneme.text.split("/")[-1]

    def check(self):
        if len(self) == 0:
            return True

        for i, word in enumerate(self):
            if not isinstance(word, Word):
                warning_msg = f"Element at index {i} is not a Word instance"
                self._add_log(f"WARNING: {warning_msg}")
                return False

            if not (word.start < word.end):
                warning_msg = f"Word '{word.text}' has invalid time order: start={word.start}, end={word.end}"
                self._add_log(f"WARNING: {warning_msg}")
                return False

            if len(word.phonemes) == 0:
                warning_msg = f"Word '{word.text}' has no phonemes"
                self._add_log(f"WARNING: {warning_msg}")
                return False

            if word.phonemes[0].start != word.start:
                warning_msg = f"Word '{word.text}' first phoneme start({word.phonemes[0].start}) != word start({word.start})"
                self._add_log(f"WARNING: {warning_msg}")
                return False

            if word.phonemes[-1].end != word.end:
                warning_msg = f"Word '{word.text}' last phoneme end({word.phonemes[-1].end}) != word end({word.end})"
                self._add_log(f"WARNING: {warning_msg}")
                return False

            for j in range(len(word.phonemes)):
                if not (word.phonemes[j].start < word.phonemes[j].end):
                    warning_msg = f"Word '{word.text}' phoneme '{word.phonemes[j].text}' has invalid time order: start={word.phonemes[j].start}, end={word.phonemes[j].end}"
                    self._add_log(f"WARNING: {warning_msg}")
                    return False

                if j < len(word.phonemes) - 1 and word.phonemes[j].end != word.phonemes[j + 1].start:
                    warning_msg = f"Word '{word.text}' phoneme '{word.phonemes[j].text}' end({word.phonemes[j].end}) != next phoneme '{word.phonemes[j + 1].text}' start({word.phonemes[j + 1].start})"
                    self._add_log(f"WARNING: {warning_msg}")
                    return False

        for i in range(len(self) - 1):
            if self[i].end != self[i + 1].start:
                warning_msg = f"Word '{self[i].text}' end({self[i].end}) != next word '{self[i + 1].text}' start({self[i + 1].start})"
                self._add_log(f"WARNING: {warning_msg}")
                return False

        return True
