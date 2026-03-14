from bisect import bisect_left

import numpy as np
import textgrid as tg


class CustomPointTier(tg.PointTier):
    def addPoint(self, point):
        i = bisect_left(self.points, point)
        self.points.insert(i, point)


class Metric:
    """
    A torchmetrics.Metric-like class with similar methods but lowered computing overhead.
    """

    def update(self, pred, target):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class VlabelerEditsCount(Metric):
    def __init__(self, move_min_frames=1, move_max_frames=2):
        # Convert frame counts to seconds
        self.move_min = move_min_frames
        self.move_max = move_max_frames
        self.counts = 0

    def update(self, pred: tg.PointTier, target: tg.PointTier):
        m, n = len(pred), len(target)
        if m != n:
            min_len = min(m, n)
            pred = pred[:min_len]
            target = target[:min_len]
            m = n = min_len

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            dp[i][0] = i  # Deletion cost
        for j in range(1, n + 1):
            dp[0][j] = j * 2  # Insertion cost

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Insertion cost
                insert = dp[i][j - 1] + 1
                if j == 1 or target[j - 1].mark != target[j - 2].mark:
                    insert += 1

                # Deletion cost
                delete = dp[i - 1][j] + 1

                # Move/substitution cost
                move = dp[i - 1][j - 1]
                time_diff = abs(pred[i - 1].time - target[j - 1].time)

                # Only count if time difference is within specified frame range
                if self.move_min <= time_diff < self.move_max:
                    move += 1

                # Additional cost for mismatched labels
                if pred[i - 1].mark != target[j - 1].mark:
                    move += 1

                dp[i][j] = min(insert, delete, move)

        self.counts += dp[m][n]

    def compute(self):
        return self.counts

    def reset(self):
        self.counts = 0


class VlabelerEditRatio(Metric):
    """
    Edit distance divided by total length of target.
    """

    def __init__(self, move_min_frames=1, move_max_frames=2):
        self.edit_distance = VlabelerEditsCount(move_min_frames, move_max_frames)
        self.total = 0

    def update(self, pred: tg.PointTier, target: tg.PointTier):
        self.edit_distance.update(pred, target)
        # Total possible edits: 每个点可能有插入、删除、移动三种操作
        # 简化计算：使用目标序列长度的两倍作为最大可能编辑数
        self.total += 2 * len(target)

    def compute(self):
        if self.total == 0:
            return 1.0
        return round(self.edit_distance.compute() / self.total, 6)

    def reset(self):
        self.edit_distance.reset()
        self.total = 0


class IntersectionOverUnion(Metric):
    """
    所有音素的交并比
    Intersection over union of all phonemes.
    """

    def __init__(self):
        self.intersection = {}
        self.sum = {}

    def update(self, pred: tg.PointTier, target: tg.PointTier):
        len_pred = len(pred) - 1
        len_target = len(target) - 1
        for i in range(len_pred):
            if pred[i].mark not in self.sum:
                self.sum[pred[i].mark] = pred[i + 1].time - pred[i].time
                self.intersection[pred[i].mark] = 0
            else:
                self.sum[pred[i].mark] += pred[i + 1].time - pred[i].time
        for j in range(len_target):
            if target[j].mark not in self.sum:
                self.sum[target[j].mark] = target[j + 1].time - target[j].time
                self.intersection[target[j].mark] = 0
            else:
                self.sum[target[j].mark] += target[j + 1].time - target[j].time

        i = 0
        j = 0
        while i < len_pred and j < len_target:
            if pred[i].mark == target[j].mark:
                intersection = min(pred[i + 1].time, target[j + 1].time) - max(
                    pred[i].time, target[j].time
                )
                self.intersection[pred[i].mark] += (
                    intersection if intersection > 0 else 0
                )

            if pred[i + 1].time < target[j + 1].time:
                i += 1
            elif pred[i + 1].time > target[j + 1].time:
                j += 1
            else:
                i += 1
                j += 1

    def compute(self, phonemes=None):
        if phonemes is None:
            return {
                k: round(v / (self.sum[k] - v), 6) if self.sum[k] != v else 0.0
                for k, v in self.intersection.items()
            }

        if isinstance(phonemes, str):
            if phonemes in self.intersection:
                return round(
                    self.intersection[phonemes]
                    / (self.sum[phonemes] - self.intersection[phonemes]),
                    6,
                )
            else:
                return None
        else:
            return {
                ph: (
                    round(
                        self.intersection[ph] / (self.sum[ph] - self.intersection[ph]),
                        6,
                    )
                    if ph in self.intersection
                    else None
                )
                for ph in phonemes
            }

    def reset(self):
        self.intersection = {}
        self.sum = {}


def compute_lcs_matches(pred, target):
    s1 = [point.mark for point in pred.points]
    s2 = [point.mark for point in target.points]
    m, n = len(s1), len(s2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    i, j = m, n
    matches = []
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches


def get_matched_pairs(pred_tier: tg.PointTier, target_tier: tg.PointTier):
    matches = compute_lcs_matches(pred_tier, target_tier)

    pred_matched = [pred_tier.points[i] for i, _ in matches]

    target_matched = [target_tier.points[j] for _, j in matches]

    return pred_matched, target_matched


class BoundaryEditDistance(Metric):
    """
    The total moving distance from the predicted boundaries to the target boundaries.
    """

    def __init__(self):
        self.distance = 0.0
        self.phonemes = 0
        self.error_phonemes = 0

    def update(self, pred: tg.PointTier, target: tg.PointTier):
        if len(pred) != len(target):
            pred_lcs, target_lcs = get_matched_pairs(pred, target)
            self.error_phonemes += abs(len(pred_lcs) - len(target))
            pred = pred_lcs
            target = target_lcs

        for i in range(len(pred)):
            if pred[i].mark != target[i].mark:
                return False

        pred_times = np.array([p.time for p in pred])
        target_times = np.array([t.time for t in target])
        time_diffs = np.abs(pred_times - target_times)

        self.distance += np.sum(time_diffs)
        self.phonemes += len(target_times)
        return True

    def compute(self):
        return round(self.distance, 6)

    def reset(self):
        self.distance = 0.0
        self.phonemes = 0


class BoundaryEditRatio(Metric):
    """
    The boundary edit distance divided by the total duration of target intervals.
    """

    def __init__(self):
        self.distance_metric = BoundaryEditDistance()
        self.duration = 0.0

    def update(self, pred: tg.PointTier, target: tg.PointTier):
        if self.distance_metric.update(pred, target):
            self.duration += target[-1].time - target[0].time

    def compute(self):
        if self.duration == 0.0:
            return 1.0
        return round(self.distance_metric.compute() / self.duration, 6)


class BoundaryEditRatioWeighted(Metric):
    """
    The boundary edit distance divided by the total duration of target intervals.
    """

    def __init__(self):
        self.distance_metric = BoundaryEditDistance()
        self.duration = 0.0
        self.counts = 0
        self.error = 0

    def update(self, pred: tg.PointTier, target: tg.PointTier):
        self.counts += 1
        if self.distance_metric.update(pred, target):
            self.duration += target[-1].time - target[0].time
        else:
            self.error += 1

    def compute(self):
        if self.duration == 0.0 or self.distance_metric.phonemes == 0.0 or self.counts == 0.0:
            return 1.0
        if (1 - self.distance_metric.error_phonemes / self.distance_metric.phonemes) + (
                self.error / self.counts) * 0.2 == 0.0:
            return 1.0
        return round(
            (self.distance_metric.compute() / self.duration) /
            (1 - self.distance_metric.error_phonemes / self.distance_metric.phonemes) +
            (self.error / self.counts) * 0.2,
            6
        )
