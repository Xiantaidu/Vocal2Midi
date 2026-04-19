import torch
from torch import nn, Tensor
from torch.nn import functional as F


class GaussianSoftBoundaryLoss(nn.Module):
    """
    Loss between predicted boundaries with gaussian-softened target boundaries.
    Arguments:
        std: float, standard deviation for gaussian softening. Larger std means softer targets.
    Inputs:
        - logits: Tensor of shape [..., T], predicted boundary logits (before sigmoid).
        - boundaries: Tensor of shape [..., T], ground truth boundary indicators, 0 = no boundary, 1 = boundary.
        - mask: Optional Tensor of shape [..., T], mask to apply on the loss.
    Outputs:
        Scalar tensor representing the loss.
    """

    def __init__(self, std: float = 1.0):
        super().__init__()
        self.std = std
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: Tensor, boundaries: Tensor, mask=None):
        if mask is not None:
            boundaries = boundaries & mask
            mask = mask.float()
        targets = gaussian_soften_boundaries(boundaries, std=self.std)
        loss = self.criterion(logits, targets)
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss


class EarthMoversDistanceLoss(nn.Module):
    """
    This loss computes the Earth Mover's Distance (EMD) between predicted and ground truth boundary sequences.
    Arguments:
        bidirectional: bool, if True, computes EMD in both forward and backward directions and averages the results.
    Inputs:
        - boundaries_pred: Tensor of shape [..., T], predicted boundary probabilities.
        - boundaries_target: Tensor of shape [..., T], ground truth boundary indicators, 0 = no boundary, 1 = boundary.
    Outputs:
        Scalar tensor representing the EMD loss.
    """

    def __init__(self, bidirectional=False):
        super().__init__()
        self.criterion = nn.L1Loss(reduction="sum")
        self.bidirectional = bidirectional

    def forward(self, boundaries_pred: Tensor, boundaries_target: Tensor, mask=None):
        boundaries_target = boundaries_target.float()
        if mask is None:
            numel = boundaries_target.numel()
        else:
            numel = mask.float().sum()
            boundaries_pred = torch.where(mask, boundaries_pred, 0.0)
            boundaries_target = torch.where(mask, boundaries_target, 0.0)
        scale = boundaries_target.sum(dim=-1, keepdim=True).clamp(min=1.0)
        loss = self.criterion(
            boundaries_pred.cumsum(dim=-1) / scale,
            boundaries_target.cumsum(dim=-1) / scale,
        ) / numel
        if self.bidirectional:
            loss_flip = self.criterion(
                boundaries_pred.flip(-1).cumsum(dim=-1) / scale,
                boundaries_target.flip(-1).cumsum(dim=-1) / scale,
            ) / numel
            loss = (loss + loss_flip) / 2
        return loss


class ApproachingMomentumLoss(nn.Module):
    """
    This loss constraints the velocities on the time dimension to approach a set of boundaries. Steps:
        1. Apply Distance Transform to compute the distance between each position to its nearest boundary;
        2. Compute a momentum weight based on the distance;
        3. Use the momentum to perform a weighted gradient detach on the predicted velocity;
        4. Accumulate the velocity through time dimension to get the predicted distance;
        5. Compute the L1 loss between the predicted distance and ground truth distance.
    Arguments:
        radius: int, maximum distance to consider for Distance Transform.
         The velocities within this radius are towards the boundary, otherwise zeros.
        decay_start: int, regions where distance <= decay_start are applied with full momentum.
        decay_width: int, regions where distance > decay_start + decay_width have no momentum.
        decay_alpha: float, spacial scaling factor for the decay function.
        decay_power: float, power of the decay function.
    Inputs:
        - velocities: Tensor of shape [..., T], predicted velocities.
          Negative means approaching from left side, positive means approaching from right side.
        - boundaries: Tensor of shape [..., T], ground truth boundary indicators, 0 = no boundary, 1 = boundary.
        - mask: Optional Tensor of shape [..., T], mask to apply on the loss.
    Outputs:
        Scalar tensor representing the approaching momentum loss.
    """

    def __init__(
            self, radius: int = 20,
            decay_start: int = 20, decay_width: int = 20,
            decay_alpha: float = 0.5, decay_power: float = 2.0,
    ):
        super().__init__()
        self.radius = radius
        self.decay_start = decay_start
        self.decay_end = decay_start + decay_width
        self.decay_alpha = decay_alpha
        self.decay_power = decay_power
        self.criterion = nn.L1Loss(reduction="none")

    def get_momentum(self, distance: Tensor):
        momentum = torch.where(
            distance <= self.decay_start,
            1.0,
            torch.pow(1.0 + self.decay_alpha * (distance - self.decay_start), -self.decay_power)
        ).float()
        momentum *= (distance <= self.decay_end).float()
        return momentum

    def forward(self, velocities: Tensor, boundaries: Tensor, mask=None):
        if mask is not None:
            boundaries = boundaries | ~mask
            mask = mask.float()
            velocities = velocities * mask
        gt_distance = distance_transform(boundaries, max_distance=self.radius)
        momentum = self.get_momentum(gt_distance)
        velocities = velocities * momentum + (1.0 - momentum) * velocities.detach()
        pred_distance = velocities.cumsum(dim=-1)
        scale = gt_distance.amax(dim=-1, keepdim=True) + 1e-6
        loss = self.criterion(pred_distance / scale, gt_distance / scale)
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss


def distance_transform(
        boundaries: Tensor, max_distance: int = None, edges=True
) -> Tensor:
    """
    Compute the distance transform for binary boundary indicators.
    For each position, compute the distance to the nearest boundary (where boundary == 1).
    :param boundaries: [..., T], 1 = boundary, 0 = non-boundary
    :param max_distance: int, maximum distance to consider
    :param edges: bool, if True, both edges of the sequence are considered as boundaries;
        otherwise, infinite distance can appear if there are no boundaries in the sequence.
    :return: [..., T]
    """
    if edges:
        boundaries = F.pad(boundaries, (1, 1), mode="constant", value=1)
    indices = torch.arange(
        boundaries.shape[-1], dtype=torch.float32, device=boundaries.device,
    ).view([1] * (boundaries.ndim - 1) + [-1])  # [..., T]
    masked_indices = torch.where(boundaries, indices, torch.full_like(indices, fill_value=float("inf")))
    distance = torch.abs(indices.unsqueeze(-1) - masked_indices.unsqueeze(-2)).amin(dim=-1)
    if edges:
        distance = distance[..., 1:-1]
    if max_distance is not None:
        distance = torch.clamp(distance, max=max_distance)
    return distance


def gaussian_soften_boundaries(boundaries: Tensor, std: float = 1.0):
    distance = distance_transform(boundaries, edges=False)
    softened = torch.exp(-0.5 * (distance / std) ** 2)
    return softened
