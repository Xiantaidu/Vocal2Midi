import torch
import torch.nn.functional as F
from torch import nn, Tensor


class GaussianBlurredBinsLoss(nn.Module):
    """
    This loss maps ground truth scores to a set of Gaussian-blurred bins, and computes the BCEWithLogitsLoss
    between the predicted logits and the blurred targets.
    Arguments:
        min_val: float, minimum value of the score range.
        max_val: float, maximum value of the score range.
        num_bins: int, number of bins (N) to quantize the score range.
        std: float, standard deviation of the Gaussian blur in the original score scale.
    Inputs:
        - logits: Tensor of shape [..., T, N], predicted logits for each bin.
        - scores: Tensor of shape [..., T], target scores.
        - presence: Tensor of shape [..., T], target presence indicators, 0 means no score.
        - mask: Optional Tensor of shape [..., T], mask to apply on the loss.
    Outputs:
        Scalar tensor representing the Gaussian blurred bins loss.
    """

    def __init__(self, min_val: float, max_val: float, num_bins: int, std: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins
        self.std = std / (max_val - min_val) * (num_bins - 1)
        centers = torch.linspace(min_val, max_val, num_bins)
        self.register_buffer("centers", centers, persistent=False)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def get_gaussian_blurred_bins(self, scores: Tensor, presence: Tensor) -> Tensor:
        B = (1,) * (scores.ndim - 2)
        centers = self.centers.reshape(*B, 1, -1)  # [..., 1, N]
        diffs = scores.unsqueeze(-1) - centers  # [..., T, N]
        gaussians = torch.exp(-0.5 * (diffs / self.std) ** 2)  # [..., T, N]
        targets = gaussians * presence.unsqueeze(-1)  # zero out where no presence
        return targets

    def forward(self, logits: Tensor, scores: Tensor, presence: Tensor, mask=None) -> Tensor:
        if mask is not None:
            mask = mask.unsqueeze(-1).float().expand_as(logits)
        targets = self.get_gaussian_blurred_bins(scores, presence)  # [..., T, N]
        loss = self.criterion(logits, targets)
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss


class CascadedDialCaliperLoss(nn.Module):
    """
    This loss constraints a series of cascaded caliper dials of different periods to point to the target values.
    These dials can be used to refine a coarse beam value to a more precise score.
    Arguments:
        periods: list of float, periods for each dial.
    Inputs:
        - dials: Tensor of shape [..., N, 2], predicted dial pointers in Cartesian coordinates.
        - targets: Tensor of shape [...], target scores.
        - mask: Optional Tensor of shape [...], mask to apply on the loss.
    Outputs:
        Scalar tensor representing the cascaded dial caliper loss.
    """
    def __init__(self, periods: list[float]):
        super().__init__()
        self.num_periods = len(periods)
        self.register_buffer(
            "periods",
            torch.tensor(periods, dtype=torch.float32).unsqueeze(0),
            persistent=False
        )

    def forward(self, dials: Tensor, targets: Tensor, mask: Tensor = None) -> Tensor:
        if mask is not None:
            dials = dials[mask]  # [X, N, 2]
            targets = targets[mask]  # [X]
        else:
            dials = dials.view(-1, self.num_periods, 2)  # [X, N, 2]
            targets = targets.view(-1)  # [X]

        target_angles = (targets.unsqueeze(-1) / self.periods) * (2 * torch.pi)  # [X, N]
        target_vectors = torch.stack([
            torch.cos(target_angles),
            torch.sin(target_angles),
        ], dim=-1)  # [X, N, 2]
        pred_vectors = F.normalize(dials, p=2, dim=-1)
        loss = (1.0 - (pred_vectors * target_vectors).sum(dim=-1)).sum(dim=-1).mean()

        return loss
