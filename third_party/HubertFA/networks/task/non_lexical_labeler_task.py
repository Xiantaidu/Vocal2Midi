import random

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler_module

from networks.layer.backbone.cvnt import CVNT
from networks.optimizer.muon import Muon_AdamW
from tools.decoder import NonLexicalDecoder


class LitNonLexicalLabelerTask(pl.LightningModule):
    def __init__(
            self,
            vocab: dict,
            config: dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.validation_error_rates = []

        self.vocab: dict = vocab
        self.config: dict = config
        self.hubert_config: dict = config['hubert_config']
        self.mel_spec_config: dict = config["mel_spec_config"]
        self.optimizer_config: dict = config["optimizer_config"]
        self.loss_config: dict = config["loss_config"]

        self.hop_size: int = self.mel_spec_config["hop_size"]
        self.window_size: int = self.mel_spec_config["window_size"]
        self.sample_rate: int = self.mel_spec_config["sample_rate"]
        self.frame_length: float = self.hop_size / self.sample_rate

        self.non_lexical_phonemes: list = self.vocab["non_lexical_phonemes"]
        self.non_lexical_mask_ratio: float = self.config["cvnt_arg"]["mask_ratio"]

        self.aug_num: int = config['aug_num']
        self.class_names: list = ['None', *self.non_lexical_phonemes]
        self.num_classes: int = len(self.class_names)
        assert self.num_classes > 1, "non_lexical_phonemes must have at least one phoneme."

        self.cvnt = CVNT(config['cvnt_arg'], in_channels=self.hubert_config['channel'],
                         output_size=self.num_classes)

        self.losses_names = [
            "ce_loss",
            "focal_loss",
            "dice_loss",
            "consistency_loss",
            "total_loss",
        ]
        self.losses_weights = torch.tensor(self.loss_config["losses"]["weights"])

        # loss function
        self.MSE_loss_fn = nn.MSELoss()
        self.decoder: NonLexicalDecoder = NonLexicalDecoder(self.vocab, self.class_names, self.sample_rate,
                                                            self.hop_size)

        # validation_step_outputs
        self.validation_step_outputs = {"losses": [], "tiers-0": [], "tiers-1": []}

        self.loss_weights = config.get('loss_weights', [1.0] * self.num_classes)
        self.focal_gamma = config.get('focal_gamma', 3.0)
        self.dice_smooth = config.get('dice_smooth', 0.1)

        self.get_mel_spec = None

    def on_validation_start(self):
        self.on_train_start()

    def on_train_start(self):
        self.losses_weights = self.losses_weights.to(self.device)

    def make_non_lexical_mask(self, shape, non_lexical_intervals):
        B, C, T = shape
        non_lexical_mask = torch.zeros((B, 1, T), dtype=torch.bool, device=self.device)

        if self.non_lexical_mask_ratio > 0:
            mask_idxes = random.choices(range(B), k=int(B * self.non_lexical_mask_ratio))

            for mask_idx in mask_idxes:
                non_lexical_interval = non_lexical_intervals[mask_idx]
                if len(non_lexical_interval) > 0:
                    non_lexical_idx = random.choices(range(len(non_lexical_interval)),
                                                     k=int(len(non_lexical_interval) * self.non_lexical_mask_ratio))
                    for idx in non_lexical_idx:
                        start_frame = non_lexical_interval[idx][0]
                        end_frame = non_lexical_interval[idx][1]
                        ph_len = end_frame - start_frame

                        if ph_len > 0:
                            mask_len = max(1, int(self.non_lexical_mask_ratio * ph_len))
                            offset = random.randint(0, max(0, ph_len - mask_len))
                            non_lexical_mask[mask_idx, :, start_frame + offset:start_frame + offset + mask_len] = True
        return non_lexical_mask

    def cross_entropy_and_focal_loss(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        weights = torch.tensor(self.loss_weights, device=logits.device)

        ce_loss = F.nll_loss(log_probs, targets, weight=weights)
        ce_per_element = F.nll_loss(log_probs, targets, reduction='none')
        pt = torch.exp(-ce_per_element)
        focal_loss = ((1 - pt) ** self.focal_gamma * ce_per_element).mean()

        return ce_loss, focal_loss

    def dice_loss(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        batch_size, num_classes, seq_len = probs.shape

        target_mask = torch.zeros_like(probs)
        targets_expanded = targets.unsqueeze(1).expand(-1, num_classes, -1)
        target_mask.scatter_(1, targets_expanded, 1)

        target_mask = target_mask[:, 1:, :]
        pred_masks = probs[:, 1:, :]

        intersection = (pred_masks * target_mask).sum(dim=(0, 2))
        cardinality_pred = pred_masks.sum(dim=(0, 2))
        cardinality_target = target_mask.sum(dim=(0, 2))

        valid_classes = (cardinality_target > 0)
        if not valid_classes.any():
            return torch.tensor(0.0, device=logits.device)

        dice = (2 * intersection[valid_classes] + self.dice_smooth) / \
               (cardinality_pred[valid_classes] + cardinality_target[valid_classes] + self.dice_smooth)

        return (1 - dice).mean()

    def _get_consistency_loss(
            self,
            cvnt_logits  # [B, N, T]
    ):
        B, N, T = cvnt_logits.shape
        if self.aug_num <= 1:
            return torch.tensor(0.0, device=self.device)

        assert B % self.aug_num == 0, f"batch size must be divisible by aug_num - {self.aug_num}."

        batch_size = B // self.aug_num
        if batch_size == 0:
            return torch.tensor(0.0, device=self.device)

        grouped_logits = cvnt_logits.view(batch_size, self.aug_num, -1, T)  # [B, aug_num, N, T]

        original_logits = grouped_logits[:, 0]  # [B, N, T]
        augmented_logits = grouped_logits[:, 1:]  # [B, aug_num-1, N, T]

        consistency_losses = []
        for i in range(augmented_logits.size(1)):
            aug_logit = augmented_logits[:, i]  # [B, N, T]

            original_probs = F.softmax(original_logits, dim=1)
            aug_probs = F.softmax(aug_logit, dim=1)

            kl_loss = F.kl_div(aug_probs.log(), original_probs, reduction='batchmean', log_target=False)
            mse_loss = F.mse_loss(aug_probs, original_probs)

            combined_loss = 0.7 * kl_loss + 0.3 * mse_loss
            consistency_losses.append(combined_loss)

        if consistency_losses:
            consistency_loss = torch.stack(consistency_losses).mean()
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)

        return consistency_loss

    def _get_loss(
            self,
            cvnt_logits,  # [B, N, T]
            non_lexical_target,
            valid=False,
    ):
        target_indices = torch.argmax(non_lexical_target, dim=1)
        ce_loss, focal_loss = self.cross_entropy_and_focal_loss(cvnt_logits, target_indices)
        dice_loss = self.dice_loss(cvnt_logits, target_indices)
        consistency_loss = self._get_consistency_loss(cvnt_logits) if not valid else torch.tensor(0.0,
                                                                                                  device=self.device)

        losses = [
            ce_loss,
            focal_loss,
            dice_loss,
            consistency_loss
        ]

        return losses

    def forward(self,
                x,  # [B, C, T]
                ):
        cvnt_logits = self.cvnt(x)  # [B, N, T]
        return cvnt_logits

    def training_step(self, batch, batch_idx):
        (
            name,
            mel_spec,
            input_feature,  # [B, C, T]
            input_feature_lengths,  # (B)
            non_lexical_target,  # [B, N, T]
            non_lexical_intervals,
        ) = batch

        non_lexical_mask = self.make_non_lexical_mask(input_feature.shape, non_lexical_intervals)  # [B, 1, T]
        masked_input = torch.where(
            non_lexical_mask,
            torch.zeros_like(input_feature),
            input_feature
        )
        masked_non_lexical_target = torch.where(
            non_lexical_mask,
            torch.zeros_like(non_lexical_target),
            non_lexical_target
        )

        cvnt_logits = self.forward(masked_input)

        losses = self._get_loss(cvnt_logits, masked_non_lexical_target)
        total_loss = (torch.stack(losses) * self.losses_weights).sum()
        losses.append(total_loss)

        log_dict = {
            f"train/{k}": v
            for k, v in zip(self.losses_names, losses)
            if v != 0
        }
        self.log_dict(log_dict)

        precision_metrics = {}

        with torch.no_grad():
            probs = torch.softmax(cvnt_logits, dim=1)  # [valid_B, N, T]
            pred_classes = torch.argmax(probs, dim=1)  # [valid_B, T]

            for class_idx, class_name in enumerate(self.class_names):
                if class_idx == 0:
                    continue

                pred_mask = (pred_classes == class_idx)  # [valid_B, T]
                true_mask = masked_non_lexical_target[:, class_idx, :] > 0.5  # [valid_B, T]

                if true_mask.sum() > 0:
                    precision = (pred_mask & true_mask).sum() / pred_mask.sum().clamp_min(1e-8)
                    precision_metrics[f'non_lexical_phonemes/{class_name}'] = precision

        for metric_name, value in precision_metrics.items():
            self.log(metric_name, value)

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (
            name,
            mel_spec,
            input_feature,  # [B, C, T]
            input_feature_lengths,  # (B)
            non_lexical_target,
            non_lexical_intervals,  # [B,N,T]
        ) = batch

        cvnt_logits = self.forward(input_feature)  # [B,N,T]
        self.decoder.decode(
            cvnt_logits=cvnt_logits.float().cpu().numpy(),
            wav_length=None,
            non_lexical_phonemes=self.non_lexical_phonemes
        )

        with torch.no_grad():
            probs = torch.softmax(cvnt_logits, dim=1)  # [B, N, T]
            pred_classes = torch.argmax(probs, dim=1)  # [B, T]
            target_indices = torch.argmax(non_lexical_target, dim=1)  # [B, T]

            max_probs, _ = torch.max(probs, dim=1)  # [B, T]
            correct_frames = ((pred_classes == target_indices) & (max_probs > 0.65)).float()

            batch_error_rates = 1 - correct_frames.mean(dim=1)  # [B]
            frame_error_rate = batch_error_rates.mean()

        prefix = "unseen_evaluate" if dataloader_idx > 0 else "valid_evaluate"
        self.log(f"{prefix}/frame_error_rate", frame_error_rate, add_dataloader_idx=False)

        if dataloader_idx == 0:
            losses = self._get_loss(cvnt_logits, non_lexical_target, valid=True)
            total_loss = (torch.stack(losses) * self.losses_weights).sum()
            losses.append(total_loss)
            losses = torch.stack(losses)
            self.validation_step_outputs["losses"].append(losses)

        if ((dataloader_idx == 0 or self.config.get("draw_evaluate", False))
                and batch_idx < self.config.get("num_valid_plots", 20)):
            fig = self.decoder.plot(mel_spec.cpu().numpy(), non_lexical_target[0].cpu().numpy())
            self.logger.experiment.add_figure(f"{'evaluate' if dataloader_idx > 0 else 'valid'}/plot_{name[0]}", fig,
                                              self.global_step)

    def on_validation_epoch_end(self):
        losses = torch.stack(self.validation_step_outputs["losses"], dim=0)
        losses = (losses / ((losses > 0).sum(dim=0, keepdim=True) + 1e-6)).sum(dim=0)
        self.log_dict(
            {f"valid/{k}": v for k, v in zip(self.losses_names, losses) if v != 0}
        )

    def configure_optimizers(self):
        optimizer = Muon_AdamW(
            self,
            lr=self.optimizer_config["lr"],
            muon_args=self.optimizer_config["muon_args"],
            adamw_args=self.optimizer_config["adamw_args"],
            weight_decay=self.optimizer_config["muon_args"]["weight_decay"],
        )
        scheduler = {
            "scheduler": lr_scheduler_module.ExponentialLR(
                optimizer,
                gamma=self.optimizer_config["gamma"]
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
