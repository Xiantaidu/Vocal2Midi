import torch
import torch.nn as nn


def get_bin_index(tensor, num_bins):
    return (tensor * num_bins).floor().long().clamp(0, num_bins - 1)


def update_ema(ema, alpha, num_bins, hist):
    hist_normalized = hist / hist.sum().clamp(min=1e-10) * num_bins
    ema_updated = ema * alpha + (1 - alpha) * hist_normalized
    return ema_updated / ema_updated.sum().clamp(min=1e-10) * num_bins


class CTCGHMLoss(nn.Module):
    def __init__(self, num_bins=10, alpha=0.999):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss(reduction="none")
        self.num_bins = num_bins
        self.register_buffer("ema", torch.ones(num_bins))
        self.alpha = alpha

    def forward(self, log_probs, targets, input_lengths, target_lengths, valid=False):
        if log_probs.size(0) == 0:
            return torch.tensor(0.0, device=log_probs.device)

        device = log_probs.device
        try:
            raw_loss = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        except RuntimeError:
            raw_loss = self.ctc_loss_fn(
                log_probs.cpu(), targets.cpu(),
                input_lengths.cpu(), target_lengths.cpu()
            ).to(device)

        loss_for_ema = (-raw_loss / input_lengths).exp().clamp(1e-6, 1 - 1e-6)
        bin_indices = get_bin_index(loss_for_ema, self.num_bins)
        weights = self.ema[bin_indices].detach()
        loss_weighted = raw_loss / (weights + 1e-10)

        if not valid:
            hist = torch.histc(loss_for_ema, self.num_bins, min=0, max=1)
            self.ema.data = update_ema(self.ema, self.alpha, self.num_bins, hist)

        return loss_weighted.mean()


class MultiLabelGHMLoss(nn.Module):
    def __init__(self, num_classes: int, num_bins: int = 10, alpha: float = 0.999, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        self.register_buffer("GD_stat_ema", torch.ones(num_bins))
        self.register_buffer("label_stat_ema_each_class", torch.ones(num_classes * 3))

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self,
                pred_logits,  # [B, T]
                target_prob,  # [B, T]
                mask=None,  # [B, T]
                valid=False):
        if pred_logits.size(0) == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        mask = mask.expand_as(pred_logits) if mask is not None else 1.0
        target_prob = target_prob.clamp(self.label_smoothing, 1 - self.label_smoothing)

        raw_loss = self.loss_fn(pred_logits, target_prob)
        pred_prob = torch.sigmoid(pred_logits)

        grad_mag = (pred_prob - target_prob).abs()
        bin_indices = get_bin_index(grad_mag, self.num_bins)
        GD_weights = 1 / self.GD_stat_ema[bin_indices].detach()

        target_prob_index = (target_prob * 3).floor().long().clamp(0, 2)
        classes_weights = 1 / self.label_stat_ema_each_class[target_prob_index].detach()

        weights = torch.sqrt(GD_weights * classes_weights)
        loss_weighted = (raw_loss * weights * mask).sum() / mask.sum().clamp(min=1e-10)

        if not valid:
            GD_hist = torch.bincount(
                bin_indices.flatten(),
                weights=mask.flatten().float(),
                minlength=self.num_bins
            )
            label_hist = torch.bincount(
                target_prob_index.flatten(),
                weights=mask.flatten().float(),
                minlength=self.num_classes * 3
            )
            self.GD_stat_ema.data = update_ema(self.GD_stat_ema, self.alpha, self.num_bins, GD_hist)
            self.label_stat_ema_each_class.data = update_ema(
                self.label_stat_ema_each_class, self.alpha, self.num_classes * 3, label_hist
            )

        return loss_weighted


class GHMLoss(torch.nn.Module):
    def __init__(self, num_classes: int, num_bins: int = 10, alpha: float = 1 - 1e-6, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.num_bins = num_bins
        self.num_classes = num_classes

        self.register_buffer("GD_ema", torch.ones(num_bins))
        self.register_buffer("class_ema", torch.ones(num_classes))

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.label_smoothing = label_smoothing

    def forward(self, pred_logits,  # (B, C, T)
                target_label,  # (B, T)
                mask=None,  # (B, C, T)
                valid=False):
        if len(pred_logits) <= 0:
            return torch.tensor(0.0).to(pred_logits.device)

        B, C, T = pred_logits.shape
        assert C == self.num_classes
        target_label = target_label.long()

        # mask: [B, C, T]
        if mask is None:
            mask = torch.ones_like(pred_logits).to(pred_logits.device)
        time_mask = mask.any(dim=1)  # [B, T]
        valid_elements = time_mask.sum().float().clamp(min=1)

        pred_logits_masked = pred_logits - 1e9 * mask.logical_not().float()

        target_prob = nn.functional.one_hot(target_label, num_classes=self.num_classes).float()
        target_prob = target_prob.clamp(self.label_smoothing, 1 - self.label_smoothing)
        target_prob = target_prob.transpose(1, 2) * mask.float()  # [B, C, T]

        raw_loss = self.loss_fn(pred_logits_masked, target_prob)  # [B, T]
        pred_probs = torch.softmax(pred_logits_masked, dim=1).detach()  # [B, C, T]

        GD = (pred_probs - target_prob).abs()  # [B, C, T]
        GD_target = torch.gather(GD, 1, target_label.unsqueeze(1)).squeeze(1)  # [B, T]
        GD_index = get_bin_index(GD_target, self.num_bins)

        class_weights = self.class_ema[target_label].detach()  # [B, T]
        GD_weights = self.GD_ema[GD_index].detach()  # [B, T]
        weights = torch.sqrt(class_weights * GD_weights)  # [B, T]

        loss_weighted = (raw_loss / weights.clamp(min=1e-10)) * time_mask.float()  # [B, T]
        loss_final = loss_weighted.sum() / valid_elements

        if not valid:
            if valid_elements > 0:
                valid_target = target_label[time_mask]  # [N]
                valid_GD_index = GD_index[time_mask]  # [N]

                class_hist = torch.bincount(
                    valid_target.flatten(),
                    minlength=self.num_classes
                )
                GD_hist = torch.bincount(
                    valid_GD_index.flatten(),
                    minlength=self.num_bins
                )

                self.GD_ema = update_ema(self.GD_ema, self.alpha, self.num_bins, GD_hist)
                self.class_ema = update_ema(self.class_ema, self.alpha, self.num_classes, class_hist)

        return loss_final


if __name__ == "__main__":
    torch.manual_seed(42)
    loss_fn = MultiLabelGHMLoss(10, alpha=0.9)
    _input = torch.randn(3, 3, 10)  # logits
    target = (torch.randn(3, 3, 10) > 0).float()
    print("MultiLabelGHMLoss test:", loss_fn(_input, target))

    loss_fn2 = GHMLoss(num_classes=10, alpha=0.9)
    _input2 = torch.randn(2, 10, 5)  # B=2, C=10, T=5
    target2 = torch.randint(0, 10, (2, 5))  # B=2, T=5
    print("GHMLoss test:", loss_fn2(_input2, target2))
