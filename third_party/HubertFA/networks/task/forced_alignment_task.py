import lightning as pl
import textgrid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler_module

import networks.scheduler as scheduler_module
from networks.layer.backbone.unet import UNetBackbone
from networks.layer.block.resnet_block import AttentionResidualBasicBlock
from networks.layer.fusion.curves_fusion import PowerCurveEdgeFusion
from networks.layer.scaling.stride_conv import DownSampling, UpSampling
from networks.loss.GHMLoss import CTCGHMLoss, GHMLoss, MultiLabelGHMLoss
from networks.optimizer.muon import Muon_AdamW
from scripts.evaluate import remove_ignored_phonemes, quantize_tier
from tools.decoder import AlignmentDecoder
from tools.metrics import BoundaryEditRatio, BoundaryEditRatioWeighted, VlabelerEditRatio, CustomPointTier


class LitForcedAlignmentTask(pl.LightningModule):
    def __init__(
            self,
            vocab: dict,
            config: dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab: dict = vocab
        self.config: dict = config
        self.mel_spec_config: dict = config["mel_spec_config"]
        self.hubert_config: dict = config["hubert_config"]
        self.optimizer_config: dict = config["optimizer_config"]
        self.loss_config: dict = config["loss_config"]
        self.fa_arg: dict = config["fa_arg"]

        self.hop_size = self.mel_spec_config["hop_size"]
        self.window_size = self.mel_spec_config["window_size"]
        self.sample_rate = self.mel_spec_config["sample_rate"]
        self.frame_length = self.hop_size / self.sample_rate

        self.aug_num: int = config['aug_num']
        self.silent_phonemes: list = self.vocab["silent_phonemes"]
        self.ignored_phonemes: list = [x for x in self.silent_phonemes]
        self.language_prefix: bool = self.vocab.get("language_prefix", True)

        self.backbone = UNetBackbone(
            input_dims=self.hubert_config["channel"],
            output_dims=self.fa_arg["hidden_dims"],
            hidden_dims=self.fa_arg["hidden_dims"],
            block=AttentionResidualBasicBlock,
            down_sampling=DownSampling,
            up_sampling=UpSampling,
            down_sampling_factor=self.fa_arg["down_sampling_factor"],
            down_sampling_times=self.fa_arg["down_sampling_times"],
            channels_scaleup_factor=self.fa_arg["channels_scaleup_factor"],
            dropout=self.fa_arg["dropout"],
        )

        self.head = nn.Conv1d(
            in_channels=self.fa_arg["hidden_dims"],
            out_channels=self.vocab["vocab_size"] + 2,
            kernel_size=1
        )

        self.curves_edge_fusion = PowerCurveEdgeFusion(
            feature_dim=self.fa_arg["hidden_dims"],
            hidden_dim=64,
            dropout=self.fa_arg["curves_attention_dropout"]
        )

        self.losses_names = [
            "ph_frame_GHM_loss",
            "ph_edge_GHM_loss",
            "ph_edge_diff_loss",
            "ctc_GHM_loss",
            "consistency_loss",
            "total_loss",
        ]
        self.losses_weights = torch.tensor(self.loss_config["losses"]["weights"])

        self.losses_schedulers = []
        for enabled in self.loss_config["losses"]["enable_RampUpScheduler"]:
            if enabled:
                self.losses_schedulers.append(
                    scheduler_module.GaussianRampUpScheduler(
                        max_steps=self.optimizer_config["total_steps"] * 5
                    )
                )
            else:
                self.losses_schedulers.append(scheduler_module.NoneScheduler())

        # loss function
        self.ph_frame_GHM_loss_fn = GHMLoss(
            num_classes=self.vocab["vocab_size"],
            num_bins=self.loss_config["function"]["num_bins"],
            alpha=self.loss_config["function"]["alpha"],
            label_smoothing=self.loss_config["function"]["label_smoothing"],
        )
        self.ph_edge_GHM_loss_fn = MultiLabelGHMLoss(
            num_classes=1,
            num_bins=self.loss_config["function"]["num_bins"],
            alpha=self.loss_config["function"]["alpha"],
            label_smoothing=0.0,
        )
        self.ph_edge_diff_GHM_loss_fn = MultiLabelGHMLoss(
            num_classes=1,
            num_bins=self.loss_config["function"]["num_bins"],
            alpha=self.loss_config["function"]["alpha"],
            label_smoothing=0.0,
        )

        self.MSE_loss_fn = nn.MSELoss()
        self.CTC_GHM_loss_fn = CTCGHMLoss(alpha=1 - 1e-3)

        self.unitsEncoder = None
        self.decoder = AlignmentDecoder(self.vocab, self.sample_rate, self.hop_size)

        # validation_step_outputs
        self.validation_step_outputs = {"losses": [], "tiers-0": [], "tiers-1": []}

    def load_pretrained(self, pretrained_model):
        self.backbone = pretrained_model.backbone
        if self.vocab["vocab_size"] == pretrained_model.vocab["vocab_size"]:
            self.head = pretrained_model.head
        else:
            self.head = nn.Linear(
                self.backbone.output_dims, self.vocab["vocab_size"] + 2
            )

    def on_validation_start(self):
        self.on_train_start()

    def on_train_start(self):
        # resume loss schedulers
        for scheduler in self.losses_schedulers:
            scheduler.resume(self.global_step)
        self.losses_weights = self.losses_weights.to(self.device)

    def _losses_schedulers_step(self):
        for scheduler in self.losses_schedulers:
            scheduler.step()

    def _losses_schedulers_call(self):
        return torch.tensor([scheduler() for scheduler in self.losses_schedulers]).to(self.device)

    def _get_consistency_loss(
            self,
            ph_frame_logits,  # [B,vocab_size,T]
            ph_edge_logits,  # [B,T]
    ):
        def compute_consistency_for_frame_logits(logits):
            B, C, T = logits.shape

            if self.aug_num <= 1:
                return torch.tensor(0.0, device=self.device)

            assert B % self.aug_num == 0, f"batch size must be divisible by aug_num - {self.aug_num}."

            batch_size = B // self.aug_num
            if batch_size == 0:
                return torch.tensor(0.0, device=self.device)

            grouped_logits = logits.view(batch_size, self.aug_num, C, T)  # [B, aug_num, C, T]
            original_logits = grouped_logits[:, 0]  # [B, C, T]
            augmented_logits = grouped_logits[:, 1:]  # [B, aug_num-1, C, T]

            consistency_losses = []
            for i in range(augmented_logits.size(1)):
                aug_logit = augmented_logits[:, i]  # [B, C, T]

                original_probs = F.softmax(original_logits, dim=1)
                aug_probs = F.softmax(aug_logit, dim=1)

                original_probs_T = original_probs.transpose(1, 2)  # [B, T, C]
                aug_probs_T = aug_probs.transpose(1, 2)  # [B, T, C]

                kl_loss = F.kl_div(torch.log(aug_probs_T + 1e-8), original_probs_T, reduction='batchmean',
                                   log_target=False)
                mse_loss = F.mse_loss(aug_probs, original_probs)

                combined_loss = 0.7 * kl_loss + 0.3 * mse_loss
                consistency_losses.append(combined_loss)

            if consistency_losses:
                return torch.stack(consistency_losses).mean()
            else:
                return torch.tensor(0.0, device=self.device)

        def compute_consistency_for_edge_logits(logits):
            B, T = logits.shape

            if self.aug_num <= 1:
                return torch.tensor(0.0, device=self.device)

            assert B % self.aug_num == 0, f"batch size must be divisible by aug_num - {self.aug_num}."

            batch_size = B // self.aug_num
            if batch_size == 0:
                return torch.tensor(0.0, device=self.device)

            grouped_logits = logits.view(batch_size, self.aug_num, T)  # [B, aug_num, T]
            original_logits = grouped_logits[:, 0]  # [B, T]
            augmented_logits = grouped_logits[:, 1:]  # [B, aug_num-1, T]

            consistency_losses = []
            for i in range(augmented_logits.size(1)):
                aug_logit = augmented_logits[:, i]  # [B, T]

                original_probs = torch.sigmoid(original_logits)
                aug_probs = torch.sigmoid(aug_logit)

                mse_loss = F.mse_loss(aug_probs, original_probs)
                consistency_losses.append(mse_loss)

            if consistency_losses:
                return torch.stack(consistency_losses).mean()
            else:
                return torch.tensor(0.0, device=self.device)

        frame_consistency_loss = compute_consistency_for_frame_logits(ph_frame_logits)
        edge_consistency_loss = compute_consistency_for_edge_logits(ph_edge_logits)

        total_consistency_loss = (frame_consistency_loss + edge_consistency_loss) / 2

        return total_consistency_loss

    def _get_loss(
            self,
            ph_frame_logits,  # (B, C, T)
            ph_edge_logits,  # (B, T)
            ctc_logits,  # (B, C, T)
            ph_frame_gt,  # (B, T)
            ph_edge_gt,  # (B, T)
            ph_seq_gt,  # (B, S)
            ph_seq_lengths_gt,  # (B)
            ph_mask,  # (B, vocab_size)
            input_feature_lengths,  # (B)
            valid=False
    ):
        device = ph_frame_logits.device
        ZERO = torch.tensor(0.0, device=device, requires_grad=True)

        time_mask = torch.arange(ph_frame_logits.shape[2], device=device)[None, :] < input_feature_lengths[:, None]
        time_mask = time_mask.float()  # (B, T)

        edge_diff_gt = torch.diff(ph_edge_gt, dim=1)
        edge_diff_gt = (edge_diff_gt + 1) / 2

        ph_edge_sigmoid = torch.sigmoid(ph_edge_logits)
        edge_diff_pred = torch.diff(ph_edge_sigmoid, dim=1)
        edge_diff_pred = (edge_diff_pred + 1) / 2

        valid_diff_mask = time_mask[:, 1:] > 0
        ph_edge_diff_loss = self.ph_edge_diff_GHM_loss_fn(
            edge_diff_pred,  # (B,T-1)
            edge_diff_gt,  # (B,T-1)
            valid_diff_mask,
            valid
        ) if valid_diff_mask.any() else ZERO

        combined_mask = ph_mask.unsqueeze(-1) * time_mask.unsqueeze(1)  # (B, C, T)
        ph_frame_GHM_loss = self.ph_frame_GHM_loss_fn(
            ph_frame_logits,  # (B, C, T)
            ph_frame_gt,  # (B, T)
            combined_mask,  # (B, C, T)
            valid
        )

        ph_edge_GHM_loss = self.ph_edge_GHM_loss_fn(
            ph_edge_logits,  # (B, T)
            ph_edge_gt,  # (B, T)
            time_mask,  # (B, T)
            valid
        )

        ctc_log_probs = torch.log_softmax(ctc_logits, dim=1)
        ctc_log_probs = ctc_log_probs.permute(2, 0, 1)  # (T, B, C)

        ctc_GHM_loss = self.CTC_GHM_loss_fn(
            ctc_log_probs,
            ph_seq_gt,
            input_feature_lengths,
            ph_seq_lengths_gt,
            valid
        )

        consistency_loss = self._get_consistency_loss(ph_frame_logits, ph_edge_logits) if not valid \
            else torch.tensor(0.0, device=self.device)

        losses = [
            ph_frame_GHM_loss,
            ph_edge_GHM_loss,
            ph_edge_diff_loss,
            ctc_GHM_loss,
            consistency_loss
        ]

        return losses

    def forward(self,
                x,  # [B, C, T]
                curves  # [B, C, T]
                ):
        x = self.backbone(x)  # [B, hidden_dims, T]
        edge_enhancement = self.curves_edge_fusion(x, curves)  # [B, 1, T]

        logits = self.head(x)  # [B, vocab_size + 2, T]

        ph_frame_logits = logits[:, 2:, :]  # [B, vocab_size, T]
        ph_edge_logits = logits[:, 0, :] + edge_enhancement.squeeze(1)  # [B, T]

        ctc_logits = torch.cat([
            logits[:, [1], :],  # [B, 1, T]
            logits[:, 3:, :]  # [B, vocab_size-1, T]
        ], dim=1)  # [B, vocab_size, T]

        return ph_frame_logits, ph_edge_logits, ctc_logits

    def training_step(self, batch, batch_idx):
        (
            input_feature,  # (B, C, T)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_id_seq,  # (B S)
            ph_seq_lengths,  # (B)
            ph_edge,  # (B, T)
            ph_frame,  # (B, T)
            ph_mask,  # (B vocab_size)
            melspec,
            ph_time,
            name,
            ph_seq_raw,
            ph_time_raw,
            curves,  # [B, 1, T]
            wav_length,
        ) = batch

        (
            ph_frame_logits,  # (B, C, T)
            ph_edge_logits,  # (B, T)
            ctc_logits,  # (B, C, T)
        ) = self.forward(input_feature, curves)

        losses = self._get_loss(
            ph_frame_logits,
            ph_edge_logits,
            ctc_logits,
            ph_frame,
            ph_edge,
            ph_id_seq,
            ph_seq_lengths,
            ph_mask,
            input_feature_lengths,
            valid=False
        )

        schedule_weight = self._losses_schedulers_call()
        self._losses_schedulers_step()
        total_loss = (torch.stack(losses) * self.losses_weights * schedule_weight).sum()
        losses.append(total_loss)

        log_dict = {
            f"train/{k}": v
            for k, v in zip(self.losses_names, losses)
            if v != 0
        }
        log_dict["scheduler/lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
        log_dict.update(
            {
                f"scheduler/{k}": v
                for k, v in zip(self.losses_names, schedule_weight)
                if v != 1
            }
        )
        self.log_dict(log_dict)
        return total_loss

    def _get_evaluate_loss(self, tiers):
        metrics = {
            "BoundaryEditRatio": BoundaryEditRatio(),
            "BoundaryEditRatioWeighted": BoundaryEditRatioWeighted(),
            "VlabelerEditRatio_1-2frames": VlabelerEditRatio(move_min_frames=1, move_max_frames=2),
            "VlabelerEditRatio_3-5frames": VlabelerEditRatio(move_min_frames=3, move_max_frames=5),
            "VlabelerEditRatio_6-9frames": VlabelerEditRatio(move_min_frames=6, move_max_frames=9),
            "VlabelerEditRatio_10+frames": VlabelerEditRatio(move_min_frames=10, move_max_frames=10000)
        }

        if tiers:
            for pred_tier, target_tier in tiers:
                for metric in metrics.values():
                    pred_tier = remove_ignored_phonemes(pred_tier, self.silent_phonemes)
                    target_tier = remove_ignored_phonemes(target_tier, self.silent_phonemes)
                    metric.update(quantize_tier(pred_tier, self.frame_length),
                                  quantize_tier(target_tier, self.frame_length))

        result = {key: metric.compute() for key, metric in metrics.items()}

        vlabeler_loss = result["VlabelerEditRatio_1-2frames"] * 0.1 + result["VlabelerEditRatio_3-5frames"] * 0.2 + \
                        result["VlabelerEditRatio_6-9frames"] * 0.3 + result["VlabelerEditRatio_10+frames"] * 0.4
        result["vlabeler_loss"] = vlabeler_loss
        result["total"] = vlabeler_loss * 0.5 + result["BoundaryEditRatioWeighted"] * 0.5
        return result

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (
            input_feature,  # (B, C, T)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_seq_id,  # (B S)
            ph_seq_lengths,  # (B)
            ph_edge,  # (B, T)
            ph_frame,  # (B, T)
            ph_mask,  # (B vocab_size)
            melspec,
            ph_time,
            name,
            ph_seq_raw,
            ph_time_raw,
            curves,  # [B, 1, T]
            wav_length,
        ) = batch

        (
            ph_frame_logits,  # (B, C, T)
            ph_edge_logits,  # (B, T)
            ctc_logits,  # (B, C, T)
        ) = self.forward(input_feature, curves)

        ph_seq_g2p = []
        last_ph = ""
        for ph in ph_seq_raw[0]:
            temp_ph = ph if self.vocab["vocab"][ph] != 0 else "SP"
            if temp_ph == "SP" and last_ph == "SP":
                continue
            ph_seq_g2p.append(temp_ph)
            last_ph = temp_ph

        self.decoder.decode(
            ph_frame_logits.float().cpu().numpy(),  # (B, C, T)
            ph_edge_logits.float().cpu().numpy(),
            wav_length[0].float().cpu(), ph_seq_g2p, None, None, False
        )

        if dataloader_idx == 0 or self.config.get("get_evaluate_loss", False):
            losses = self._get_loss(
                ph_frame_logits,
                ph_edge_logits,
                ctc_logits,
                ph_frame,
                ph_edge,
                ph_seq_id,
                ph_seq_lengths,
                ph_mask,
                input_feature_lengths,
                valid=True
            )

            weights = self._losses_schedulers_call() * self.losses_weights
            total_loss = (torch.stack(losses) * weights).sum()
            losses.append(total_loss)
            losses = torch.stack(losses)
            self.validation_step_outputs["losses"].append(losses)

        pred_tier = CustomPointTier(name="phonemes")
        target_tier = CustomPointTier(name="phonemes")

        for mark, time in zip(ph_seq[0], ph_time[0].cpu().numpy()):
            target_tier.addPoint(textgrid.Point(float(time), mark))

        for mark, time in zip(self.decoder.ph_seq_pred, self.decoder.ph_intervals_pred):
            pred_tier.addPoint(textgrid.Point(float(time[0]), mark))
        self.validation_step_outputs[f"tiers-{dataloader_idx}"].append((pred_tier, target_tier))

        if ((dataloader_idx == 0 or self.config.get("draw_evaluate", False))
                and batch_idx < self.config.get("num_valid_plots", 20)):
            melspec = melspec[0].cpu().numpy().squeeze()  # [N_Mel,T]
            fig = self.decoder.plot(melspec, ph_time_gt=ph_time_raw[0])
            self.logger.experiment.add_figure(f"{'evaluate' if dataloader_idx > 0 else 'valid'}/plot_{name[0]}", fig,
                                              self.global_step)

    def on_validation_epoch_end(self):
        losses = torch.stack(self.validation_step_outputs["losses"], dim=0)
        losses = (losses / ((losses > 0).sum(dim=0, keepdim=True) + 1e-6)).sum(dim=0)
        self.log_dict(
            {f"valid/{k}": v for k, v in zip(self.losses_names, losses) if v != 0}
        )

        val_loss = self._get_evaluate_loss(self.validation_step_outputs.get(f"tiers-0", []))
        for metric_name, metric_value in val_loss.items():
            self.log_dict({f"valid_evaluate/{metric_name}": metric_value})
        self.validation_step_outputs[f"tiers-0"].clear()

        evaluate_loss = self._get_evaluate_loss(self.validation_step_outputs.get(f"tiers-1", []))
        for metric_name, metric_value in evaluate_loss.items():
            self.log_dict({f"unseen_evaluate/{metric_name}": metric_value})
        self.validation_step_outputs[f"tiers-1"].clear()

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
