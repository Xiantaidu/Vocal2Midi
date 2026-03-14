import torch
import torchmetrics
from torch import Tensor


class NoteOverlapMetricCollection(torchmetrics.Metric):
    """
    This metric collection computes a set of measures for note overlap rate,
    where each note is considered as a rectangle in the time-pitch space.
    Arguments:
        pitch_width: float, the width of the note in the pitch dimension.
    Inputs:
        - pred_scores: Tensor of shape [..., T], predicted pitch scores.
        - pred_presence: Boolean tensor of shape [..., T], predicted presence indicators, 0 means no score or unvoiced.
        - target_scores: Tensor of shape [..., T], target pitch scores.
        - target_presence: Boolean tensor of shape [..., T], target presence indicators, 0 means no score or unvoiced.
        - mask: Optional Tensor of shape [..., T], mask to apply on the frames.
    Outputs:
        - overlap_precision: Ratio between overlapped area and total predicted area.
        - overlap_recall: Ratio between overlapped area and total target area.
    """
    def __init__(self, pitch_width: float, postfix: str = "", **kwargs):
        super().__init__(**kwargs)
        self.pitch_width = pitch_width
        self.postfix = postfix
        self.add_state("overlap", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total_pred", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(
            self, pred_scores: Tensor, pred_presence: Tensor,
            target_scores: Tensor, target_presence: Tensor,
            mask: Tensor = None
    ) -> None:
        if mask is not None:
            pred_presence = pred_presence & mask
            target_presence = target_presence & mask
        time_overlap = (pred_presence & target_presence).float()
        pitch_overlap = torch.clamp(self.pitch_width - torch.abs(pred_scores - target_scores), min=0.0)
        overlap = (time_overlap * pitch_overlap).sum()
        total_pred = (pred_presence.float() * self.pitch_width).sum()
        total_target = (target_presence.float() * self.pitch_width).sum()
        self.overlap += overlap
        self.total_pred += total_pred
        self.total_target += total_target

    def compute(self) -> dict[str, Tensor]:
        precision = self.overlap / (self.total_pred + 1e-6)
        recall = self.overlap / (self.total_target + 1e-6)
        return {
            f"overlap_precision{self.postfix}": precision,
            f"overlap_recall{self.postfix}": recall,
        }
