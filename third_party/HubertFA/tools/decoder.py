import numpy as np

from tools.align_word import Phoneme, Word, WordList
from tools.plot import plot_force_alignment_prob, plot_non_lexical_phonemes


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


class AlignmentDecoder:
    def __init__(self, vocab, sample_rate, hop_size):
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_length = hop_size / sample_rate
        self.T = self.ph_seq_id = self.ph_idx_seq = self.ph_frame_pred = self.ph_time_int_pred = None
        self.ph_seq_pred = self.ph_intervals_pred = self.edge_prob = self.pred_words = self.frame_confidence = None

    def decode(self,
               ph_frame_logits,  # [vocab_size, T]
               ph_edge_logits,  # [B, T]
               wav_length: float,
               ph_seq: list[str],
               word_seq: list[str] = None,
               ph_idx_to_word_idx: list[int] = None,
               ignore_sp: bool = True,
               ):
        ph_frame_logits = ph_frame_logits[0]  # [vocab_size, T]
        ph_edge_logits = ph_edge_logits[0]  # [T]
        ph_seq_id = np.array([self.vocab["vocab"][ph] for ph in ph_seq])
        self.ph_seq_id = ph_seq_id

        ph_mask = np.full(self.vocab["vocab_size"], 1e9)
        ph_mask[ph_seq_id], ph_mask[0] = 0, 0

        if word_seq is None:
            word_seq = ph_seq
            ph_idx_to_word_idx = np.arange(len(ph_seq))

        num_frames = int((wav_length * self.sample_rate + 0.5) / self.hop_size)
        ph_frame_logits, ph_edge_logits = ph_frame_logits[:, :num_frames], ph_edge_logits[:num_frames]

        ph_frame_logits_adjusted = ph_frame_logits - ph_mask[:, np.newaxis]  # [vocab_size, 1]
        ph_frame_pred = softmax(ph_frame_logits_adjusted, axis=0).astype("float32")  # [vocab_size, T]
        ph_prob_log = log_softmax(ph_frame_logits_adjusted, axis=0).astype("float32")  # [vocab_size, T]
        ph_edge_pred = np.clip(sigmoid(ph_edge_logits), 0.0, 1.0).astype("float32")  # [T]
        self.ph_frame_pred = ph_frame_pred  # [vocab_size, T]
        vocab_size, self.T = ph_frame_pred.shape

        # decode
        edge_diff = np.concatenate((np.diff(ph_edge_pred, axis=0), [0]), axis=0)  # [T]
        self.edge_prob = (ph_edge_pred + np.concatenate(([0], ph_edge_pred[:-1]))).clip(0, 1)  # [T]

        (
            ph_idx_seq,
            ph_time_int_pred,
            frame_confidence,  # [T]
        ) = self._decode(
            ph_seq_id,  # [ph_seq_len]
            ph_prob_log,  # [vocab_size, T]
            self.edge_prob,  # [T]
        )
        total_confidence = np.exp(np.mean(np.log(frame_confidence + 1e-6)) / 3)

        self.ph_idx_seq = ph_idx_seq
        self.ph_time_int_pred = np.array(ph_time_int_pred, dtype="int32")
        self.frame_confidence = frame_confidence

        # postprocess
        ph_time_fractional = (edge_diff[self.ph_time_int_pred] / 2).clip(-0.5, 0.5)
        ph_time_pred = self.frame_length * (
            np.concatenate([self.ph_time_int_pred.astype("float32") + ph_time_fractional, [self.T]])
        )
        ph_time_pred = np.clip(ph_time_pred, 0, None)
        ph_intervals = np.stack([ph_time_pred[:-1], ph_time_pred[1:]], axis=1)

        word = None
        words: WordList = WordList()

        ph_seq_decoded = []
        word_idx_last = -1
        for i, ph_idx in enumerate(ph_idx_seq):
            ph_seq_decoded.append(ph_seq[ph_idx])
            # ph_idx只能用于两种情况：ph_seq和ph_idx_to_word_idx
            if ph_seq[ph_idx] == 'SP' and ignore_sp:
                continue
            phoneme = Phoneme(ph_intervals[i, 0], ph_intervals[i, 1], ph_seq[ph_idx])

            word_idx = ph_idx_to_word_idx[ph_idx]
            if word_idx == word_idx_last:
                word.append_phoneme(phoneme)
            else:
                word = Word(ph_intervals[i, 0], ph_intervals[i, 1], word_seq[word_idx])
                word.add_phoneme(phoneme)
                words.append(word)
                word_idx_last = word_idx
        self.ph_seq_pred, self.ph_intervals_pred, self.pred_words = words.phonemes, words.intervals, words
        return words, total_confidence

    def plot(self, melspec, ph_time_gt=None):
        ph_idx_frame = np.zeros(self.T, dtype="int32")
        ph_intervals_pred_int = (np.array(self.ph_intervals_pred) / self.frame_length).round().astype("int32")
        ph_time_gt_int = (ph_time_gt / self.frame_length).round().astype("int32") if ph_time_gt is not None else None
        last_ph_idx = 0
        for ph_idx, ph_time in zip(self.ph_idx_seq, self.ph_time_int_pred):
            ph_idx_frame[ph_time] += ph_idx - last_ph_idx
            last_ph_idx = ph_idx
        ph_idx_frame = np.cumsum(ph_idx_frame)
        return plot_force_alignment_prob(
            melspec=melspec[:, :self.T], ph_seq=self.pred_words.phonemes, ph_intervals=ph_intervals_pred_int,
            frame_confidence=self.frame_confidence, edge_prob=self.edge_prob,
            ph_frame_prob=self.ph_frame_pred[self.ph_seq_id, :], ph_frame_id_gt=ph_idx_frame,
            ph_time_gt=ph_time_gt_int
        )

    @staticmethod
    def forward_pass(T, S, prob_log, edge_prob, curr_ph_max_prob_log, dp, ph_seq_id, prob3_pad_len=2):
        backtrack_s = np.full_like(dp, -1, dtype=np.int32)  # [S, T]
        edge_prob_log, not_edge_prob_log = np.log(edge_prob + 1e-6), np.log(1 - edge_prob + 1e-6)
        mask_reset = (ph_seq_id == 0)  # [S]

        # 预分配数组
        prob1 = np.empty(S, dtype=np.float32)
        prob2 = np.full(S, -np.inf, dtype=np.float32)
        prob3 = np.full(S, -np.inf, dtype=np.float32)

        # 预计算prob3索引和掩码
        i_vals_prob3 = np.arange(prob3_pad_len, S)
        idx_arr = np.clip(i_vals_prob3 - prob3_pad_len + 1, 0, S - 1)
        mask_cond_prob3 = (idx_arr >= S - 1) | (ph_seq_id[idx_arr] == 0)

        # 循环时间帧
        for t in range(1, T):
            prob_log_t, edge_log_t, not_edge_log_t = prob_log[:, t], edge_prob_log[t], not_edge_prob_log[t]
            dp_prev = dp[:, t - 1]

            # 类型1转移: 停留在当前音素
            prob1[:] = dp_prev + prob_log_t + not_edge_log_t

            # 类型2转移: 移动到下一个音素
            prob2[1:] = dp_prev[:S - 1] + prob_log_t[:S - 1] + edge_log_t + curr_ph_max_prob_log[:S - 1] * (T / S)

            # 类型3转移: 跳转到后续音素
            candidate_vals = dp_prev[:S - prob3_pad_len] + prob_log_t[
                :S - prob3_pad_len] + edge_log_t + curr_ph_max_prob_log[:S - prob3_pad_len] * (T / S)
            prob3[i_vals_prob3] = np.where(mask_cond_prob3, candidate_vals, -np.inf)

            # 组合概率并找出最佳转移
            stacked_probs = np.vstack((prob1, prob2, prob3))
            max_indices = np.argmax(stacked_probs, axis=0)
            dp[:, t], backtrack_s[:, t] = stacked_probs[max_indices, np.arange(S)], max_indices

            # 更新当前音素最大概率
            mask_type0 = (max_indices == 0)
            np.maximum(curr_ph_max_prob_log, prob_log_t, out=curr_ph_max_prob_log, where=mask_type0)
            np.copyto(curr_ph_max_prob_log, prob_log_t, where=~mask_type0)
            curr_ph_max_prob_log[mask_reset] = 0.0

            prob2[1:], prob3[i_vals_prob3] = -np.inf, -np.inf
        return dp, backtrack_s, curr_ph_max_prob_log

    def _decode(self,
                ph_seq_id,  # [S]
                ph_prob_log,  # [vocab_size, T]
                edge_prob):  # [T]
        vocab_size, T = ph_prob_log.shape
        S = len(ph_seq_id)
        prob_log = ph_prob_log[ph_seq_id, :]  # [S, T]

        # init
        curr_ph_max_prob_log = np.full(S, -np.inf)  # [S]
        dp = np.full((S, T), -np.inf, dtype="float32")  # [S, T]

        # 如果mode==forced，只能从SP开始或者从第一个音素开始
        dp[0, 0] = prob_log[0, 0]
        curr_ph_max_prob_log[0] = prob_log[0, 0]
        if ph_seq_id[0] == 0 and S > 1:
            dp[1, 0] = prob_log[1, 0]
            curr_ph_max_prob_log[1] = prob_log[1, 0]

        # forward
        dp, backtrack_s, curr_ph_max_prob_log = self.forward_pass(
            T, S, prob_log, edge_prob, curr_ph_max_prob_log, dp, ph_seq_id,
            prob3_pad_len=2 if S >= 2 else 1
        )

        # backward
        ph_idx_seq, ph_time_int, frame_confidence = [], [], []

        # 确定结束状态
        if S == 1:
            s = 0
        else:
            s = S - 2 if dp[-2, -1] > dp[-1, -1] and ph_seq_id[-1] == 0 else S - 1

        # 回溯路径
        for t in np.arange(T - 1, -1, -1):
            assert backtrack_s[s, t] >= 0 or t == 0
            frame_confidence.append(dp[s, t])

            if backtrack_s[s, t] != 0:
                ph_idx_seq.append(s)
                ph_time_int.append(t)

                # 根据转移类型更新s
                if backtrack_s[s, t] == 1:
                    s -= 1  # 类型2转移
                elif backtrack_s[s, t] == 2:
                    s -= 2  # 类型3转移，假设prob3_pad_len=2

        ph_idx_seq.reverse()
        ph_time_int.reverse()
        frame_confidence.reverse()
        frame_confidence = np.exp(np.diff(np.pad(frame_confidence, (1, 0), "constant", constant_values=0.0), 1))
        return ph_idx_seq, ph_time_int, frame_confidence


class NonLexicalDecoder:
    def __init__(self, vocab, class_names: list[str], sample_rate: int, hop_size: int):
        self.vocab = vocab
        self.non_lexical_phs = class_names
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_length = hop_size / sample_rate
        self.cvnt_probs = None

    def decode(self,
               cvnt_logits,
               wav_length: float | None = None,
               non_lexical_phonemes: list[str] = None,
               ) -> list[WordList]:
        non_lexical_phonemes = non_lexical_phonemes or []
        if wav_length is not None:
            num_frames = int((wav_length * self.sample_rate + 0.5) / self.hop_size)
            cvnt_logits = cvnt_logits[:, :, :num_frames]
        self.cvnt_probs = softmax(cvnt_logits, axis=1)[0]

        non_lexical_words = []
        for ph in non_lexical_phonemes:
            i = self.non_lexical_phs.index(ph)
            tag_words: list[Word] = self.non_lexical_words(self.cvnt_probs[i], tag=ph)
            non_lexical_words.append(tag_words)
        return non_lexical_words

    def plot(self, mel_spec, non_lexical_target=None):
        return plot_non_lexical_phonemes(
            mel_spec=mel_spec, cvnt_prob=self.cvnt_probs,
            label=self.non_lexical_phs, frame_duration=self.frame_length, non_lexical_target=non_lexical_target
        )

    def non_lexical_words(self, prob, threshold=0.5, max_gap=5, mix_frames=10, tag=""):
        words, start, gap_count = [], None, 0

        for i in range(len(prob)):
            if prob[i] >= threshold:
                if start is None:
                    start = i
                gap_count = 0
            elif start is not None:
                if gap_count < max_gap:
                    gap_count += 1
                else:
                    end = i - gap_count - 1
                    if end > start and (end - start) >= mix_frames:
                        word = Word(start * self.frame_length, end * self.frame_length, tag)
                        word.add_phoneme(Phoneme(start * self.frame_length, end * self.frame_length, tag))
                        words.append(word)
                    start, gap_count = None, 0

        # Handle the case where the array ends with a segment
        if start is not None and (len(prob) - start) >= mix_frames:
            word = Word(start * self.frame_length, (len(prob) - 1) * self.frame_length, tag)
            word.add_phoneme(Phoneme(start * self.frame_length, (len(prob) - 1) * self.frame_length, tag))
            words.append(word)
        return words
