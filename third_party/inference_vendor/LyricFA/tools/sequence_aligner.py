from collections import Counter
from enum import IntEnum
from typing import List, Tuple, Optional


class EditOperation(IntEnum):
    MATCH = 0
    SUBSTITUTE = 1
    DELETE = 2
    INSERT = 3


class SequenceAligner:
    OVERLAP_THRESHOLD = 0.3

    def __init__(self, deletion_cost: int = 1, insertion_cost: int = 1, substitution_cost: int = 1) -> None:
        self.deletion_cost = deletion_cost
        self.insertion_cost = insertion_cost
        self.substitution_cost = substitution_cost

    def compute_alignment(self, seq1: List[str], seq2: List[str]) -> Tuple[int, List[str], List[str]]:
        len1, len2 = len(seq1), len(seq2)

        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        bt = [[EditOperation.MATCH] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(1, len1 + 1):
            dp[i][0] = i * self.deletion_cost
            bt[i][0] = EditOperation.DELETE

        for j in range(1, len2 + 1):
            dp[0][j] = j * self.insertion_cost
            bt[0][j] = EditOperation.INSERT

        for i in range(1, len1 + 1):
            s1_char = seq1[i - 1]
            for j in range(1, len2 + 1):
                s2_char = seq2[j - 1]

                if s1_char == s2_char:
                    dp[i][j] = dp[i - 1][j - 1]
                    bt[i][j] = EditOperation.MATCH
                else:
                    sub_cost = dp[i - 1][j - 1] + self.substitution_cost
                    del_cost = dp[i - 1][j] + self.deletion_cost
                    ins_cost = dp[i][j - 1] + self.insertion_cost

                    min_cost = sub_cost
                    op = EditOperation.SUBSTITUTE
                    if del_cost < min_cost:
                        min_cost = del_cost
                        op = EditOperation.DELETE
                    if ins_cost < min_cost:
                        min_cost = ins_cost
                        op = EditOperation.INSERT

                    dp[i][j] = min_cost
                    bt[i][j] = op

        aligned1, aligned2 = self._backtrack(seq1, seq2, bt)
        return dp[len1][len2], aligned1, aligned2

    @staticmethod
    def _backtrack(seq1: List[str], seq2: List[str],
                   bt: List[List[EditOperation]]) -> Tuple[List[str], List[str]]:
        i, j = len(seq1), len(seq2)
        max_len = i + j
        res1: List[Optional[str]] = [None] * max_len
        res2: List[Optional[str]] = [None] * max_len
        idx = max_len - 1

        while i > 0 or j > 0:
            if i > 0 and j > 0:
                op = bt[i][j]
                if op in (EditOperation.MATCH, EditOperation.SUBSTITUTE):
                    res1[idx] = seq1[i - 1]
                    res2[idx] = seq2[j - 1]
                    i -= 1
                    j -= 1
                elif op == EditOperation.DELETE:
                    res1[idx] = seq1[i - 1]
                    res2[idx] = "-"
                    i -= 1
                else:  # INSERT
                    res1[idx] = "-"
                    res2[idx] = seq2[j - 1]
                    j -= 1
            elif i > 0:
                res1[idx] = seq1[i - 1]
                res2[idx] = "-"
                i -= 1
            else:  # j > 0
                res1[idx] = "-"
                res2[idx] = seq2[j - 1]
                j -= 1
            idx -= 1

        start = idx + 1
        return res1[start:], res2[start:]  # type: ignore

    @staticmethod
    def compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
        if len(seq1) < len(seq2):
            seq1, seq2 = seq2, seq1
        m, n = len(seq1), len(seq2)

        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            c1 = seq1[i - 1]
            for j in range(1, n + 1):
                if c1 == seq2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, prev

        return prev[n]

    def find_best_match(
            self,
            input_seq: List[str],
            reference_seq: List[str],
            reference_text: Optional[List[str]] = None,
            max_window_scale: float = 1.3,
            extra_window: int = 8,
    ) -> Tuple[str, int, int, Optional[List[str]], Optional[List[str]], str]:
        if not input_seq:
            return "", -1, -1, None, None, "Input sequence is empty"
        if not reference_seq:
            return "", -1, -1, None, None, "Reference sequence is empty"

        input_len = len(input_seq)
        ref_len = len(reference_seq)
        long_input_note = ""
        if input_len > ref_len:
            # 允许输入长于参考：ASR 常会出现插入/幻觉词，
            # 这里继续走窗口+编辑距离流程，尽量回填出参考歌词。
            long_input_note = "Input longer than reference; attempted approximate alignment"

        direct_start = self._find_exact_match(input_seq, reference_seq)
        if direct_start != -1:
            return self._build_exact_match_result(
                direct_start, input_len, reference_seq, reference_text
            )

        window_size = self._determine_window_size(
            input_len, ref_len, max_window_scale, extra_window
        )

        best_start, _ = self._scan_windows(input_seq, reference_seq, window_size, input_len)
        if best_start == -1:
            if long_input_note:
                return "", -1, -1, None, None, f"{long_input_note}; no matching window found"
            return "", -1, -1, None, None, "No matching window found"

        matched_text, start, end, matched_phonetic_list, matched_text_list, reason = self._build_match_from_alignment(
            input_seq, reference_seq, reference_text, best_start, window_size
        )

        if long_input_note:
            reason = f"{long_input_note}; {reason}" if reason else long_input_note

        return matched_text, start, end, matched_phonetic_list, matched_text_list, reason

    @staticmethod
    def _find_exact_match(input_seq: List[str], reference_seq: List[str]) -> int:
        input_len = len(input_seq)
        ref_len = len(reference_seq)
        for start in range(ref_len - input_len + 1):
            if reference_seq[start:start + input_len] == input_seq:
                return start
        return -1

    @staticmethod
    def _build_exact_match_result(
            start: int,
            length: int,
            reference_seq: List[str],
            reference_text: Optional[List[str]],
    ) -> Tuple[str, int, int, List[str], List[str], str]:
        end = start + length
        matched_phonetic_list = reference_seq[start:end]
        matched_text_list = reference_text[start:end] if reference_text else []
        matched_text = " ".join(matched_text_list) if matched_text_list else ""
        return matched_text, start, end, matched_phonetic_list, matched_text_list, ""

    @staticmethod
    def _determine_window_size(
            input_len: int, ref_len: int, max_window_scale: float, extra_window: int
    ) -> int:
        window_size = min(input_len + extra_window, int(input_len * max_window_scale))
        return min(window_size, ref_len)

    def compute_edit_distance(self, seq1: List[str], seq2: List[str]) -> int:
        len1, len2 = len(seq1), len(seq2)
        if len1 < len2:
            seq1, seq2 = seq2, seq1
            len1, len2 = len2, len1

        prev = [j * self.insertion_cost for j in range(len2 + 1)]
        curr = [0] * (len2 + 1)

        for i in range(1, len1 + 1):
            curr[0] = i * self.deletion_cost
            s1 = seq1[i - 1]
            for j in range(1, len2 + 1):
                s2 = seq2[j - 1]
                if s1 == s2:
                    curr[j] = prev[j - 1]
                else:
                    sub = prev[j - 1] + self.substitution_cost
                    dele = prev[j] + self.deletion_cost
                    ins = curr[j - 1] + self.insertion_cost
                    curr[j] = min(sub, dele, ins)
            prev, curr = curr, prev
        return prev[len2]

    def _scan_windows(
            self,
            input_seq: List[str],
            reference_seq: List[str],
            window_size: int,
            input_len: int,
    ) -> Tuple[int, float]:
        ref_len = len(reference_seq)
        candidates = []  # (approx_dist, start)
        input_freq = Counter(input_seq)

        for start in range(ref_len - window_size + 1):
            window = reference_seq[start:start + window_size]

            if not any(c in input_freq for c in window):
                continue

            window_freq = Counter(window)
            overlap = sum(min(input_freq[c], window_freq.get(c, 0)) for c in input_freq)
            coverage = overlap / input_len
            if coverage < self.OVERLAP_THRESHOLD:
                continue

            lcs_len = self.compute_lcs_length(input_seq, window)
            approx_dist = len(input_seq) + len(window) - 2 * lcs_len
            candidates.append((approx_dist, start))

        if not candidates:
            return -1, float('inf')

        candidates.sort(key=lambda x: x[0])

        total = len(candidates)
        num_to_keep = max(10, int(0.3 * total))
        if num_to_keep > total:
            num_to_keep = total

        best_start = -1
        min_edit_dist = float('inf')

        for approx_dist, start in candidates[:num_to_keep]:
            window = reference_seq[start:start + window_size]
            edit_dist = self.compute_edit_distance(input_seq, window)
            if edit_dist < min_edit_dist:
                min_edit_dist = edit_dist
                best_start = start
                if edit_dist == 0:
                    break

        return best_start, min_edit_dist

    def _build_match_from_alignment(
            self,
            input_seq: List[str],
            reference_seq: List[str],
            reference_text: Optional[List[str]],
            best_start: int,
            window_size: int,
    ) -> Tuple[str, int, int, Optional[List[str]], Optional[List[str]], str]:
        window_end = best_start + window_size
        window_seq = reference_seq[best_start:window_end]
        _, aligned_input, aligned_window = self.compute_alignment(input_seq, window_seq)

        matched_phonetic_list: List[str] = []
        matched_text_list: List[str] = []
        win_idx = 0

        for win_char, inp_char in zip(aligned_window, aligned_input):
            if win_char != '-':
                if inp_char != '-':
                    matched_phonetic_list.append(win_char)
                    if reference_text and win_idx < len(reference_text[best_start:window_end]):
                        matched_text_list.append(reference_text[best_start + win_idx])
                win_idx += 1

        if not matched_phonetic_list:
            return "", -1, -1, None, None, "Alignment produced empty result"

        matched_text = " ".join(matched_text_list)
        return matched_text, best_start, window_end, matched_phonetic_list, matched_text_list, ""

    def find_best_match_and_return_lyrics(
            self,
            input_pronunciation: List[str],
            reference_text: List[str],
            reference_pronunciation: List[str],
    ) -> Tuple[str, str, int, int, str]:
        matched_text, start, end, matched_phonetic_list, _, reason = self.find_best_match(
            input_seq=input_pronunciation,
            reference_seq=reference_pronunciation,
            reference_text=reference_text,
        )
        matched_phonetic = " ".join(matched_phonetic_list) if matched_phonetic_list else ""
        return matched_text, matched_phonetic, start, end, reason


def calculate_difference_count(seq1: List[str], seq2: List[str]) -> int:
    min_len = min(len(seq1), len(seq2))
    diff = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a != b)
    diff += abs(len(seq1) - len(seq2))
    return diff


class SmartHighlighter:
    def __init__(self, aligner: SequenceAligner):
        self.aligner = aligner

    def highlight_differences(
            self, asr_result: str, match_phonetic: str, match_text: str
    ) -> Tuple[str, str, str, int]:
        asr_tokens = asr_result.split()
        match_phonetic_tokens = match_phonetic.split()
        match_text_tokens = match_text.split()

        if not match_phonetic_tokens:
            asr_highlighted = [f"({t})" for t in asr_tokens]
            return " ".join(asr_highlighted), "", "", len(asr_tokens)

        edit_distance, aligned_asr, aligned_match = self.aligner.compute_alignment(
            asr_tokens, match_phonetic_tokens
        )

        asr_highlighted: List[str] = []
        phonetic_highlighted: List[str] = []
        text_highlighted: List[str] = []

        text_index = 0
        text_tokens_len = len(match_text_tokens)

        for asr_token, match_token in zip(aligned_asr, aligned_match):
            text_token = ""
            if match_token != '-':
                if text_index < text_tokens_len:
                    text_token = match_text_tokens[text_index]
                    text_index += 1

            if asr_token == '-':
                phonetic_highlighted.append(f"({match_token})")
                asr_highlighted.append("")
                if text_token:
                    text_highlighted.append(f"({text_token})")
            elif match_token == '-':
                asr_highlighted.append(f"({asr_token})")
                phonetic_highlighted.append("")
                text_highlighted.append("")
            elif asr_token != match_token:
                asr_highlighted.append(f"({asr_token})")
                phonetic_highlighted.append(f"({match_token})")
                if text_token:
                    text_highlighted.append(f"({text_token})")
            else:
                asr_highlighted.append(asr_token)
                phonetic_highlighted.append(match_token)
                if text_token:
                    text_highlighted.append(text_token)

        asr_result_str = " ".join([s for s in asr_highlighted if s])
        phonetic_result_str = " ".join([s for s in phonetic_highlighted if s])
        text_result_str = " ".join([s for s in text_highlighted if s])

        return asr_result_str, phonetic_result_str, text_result_str, edit_distance
