# File: src/train.py
import yaml
import argparse
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import HubertForCTC, get_linear_schedule_with_warmup
import json

try:
    from src.dataset import PhonemeDataset, collate_fn
except ModuleNotFoundError:
    from dataset import PhonemeDataset, collate_fn

from tqdm import tqdm


def _edit_distance(tokens_a, tokens_b):
    """Levenshtein distance on token lists."""
    n, m = len(tokens_a), len(tokens_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if tokens_a[i - 1] == tokens_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def compute_per(predictions, references):
    """Compute phoneme error rate without external downloads."""
    total_err = 0
    total_ref = 0
    for p, r in zip(predictions, references):
        p_tokens = p.split() if p else []
        r_tokens = r.split() if r else []
        total_err += _edit_distance(p_tokens, r_tokens)
        total_ref += len(r_tokens)
    if total_ref == 0:
        return 0.0
    return total_err / total_ref


def get_optimizer_grouped_parameters(model, config, backbone_lr_override=None):
    lr_backbone = backbone_lr_override or float(config["train"].get("lr_backbone", 3e-5))
    lr_head = float(config["train"].get("lr_head", 3e-4))
    weight_decay = float(config["train"].get("weight_decay", 0.01))

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and "lm_head" not in n and p.requires_grad
            ],
            "weight_decay": weight_decay,
            "lr": lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and "lm_head" not in n and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and "lm_head" in n and p.requires_grad
            ],
            "weight_decay": weight_decay,
            "lr": lr_head,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and "lm_head" in n and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lr_head,
        },
    ]
    # Filter out empty groups
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
    return optimizer_grouped_parameters


def _freeze_backbone(model):
    """Freeze all backbone (encoder) parameters, only lm_head remains trainable."""
    for name, param in model.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = False


def _unfreeze_backbone(model):
    """Unfreeze backbone parameters."""
    for name, param in model.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = True


def _configure_specaugment(model, config):
    """Properly configure HuBERT's built-in SpecAugment from config."""
    sa_conf = config.get("augment", {}).get("specaug", {})
    if not sa_conf.get("enable", False):
        model.config.mask_time_prob = 0.0
        model.config.mask_feature_prob = 0.0
        return

    time_mask_num = sa_conf.get("time_mask_num", 2)
    time_mask_width = sa_conf.get("time_mask_width", 30)
    freq_mask_num = sa_conf.get("freq_mask_num", 2)
    freq_mask_width = sa_conf.get("freq_mask_width", 20)

    typical_seq_len = 600
    model.config.mask_time_prob = min(time_mask_num * time_mask_width / typical_seq_len, 0.5)
    model.config.mask_time_length = time_mask_width

    hidden_size = model.config.hidden_size
    model.config.mask_feature_prob = min(freq_mask_num * freq_mask_width / hidden_size, 0.5)
    model.config.mask_feature_length = freq_mask_width

    print(f"SpecAugment configured: "
          f"mask_time_prob={model.config.mask_time_prob:.4f}, "
          f"mask_time_length={model.config.mask_time_length}, "
          f"mask_feature_prob={model.config.mask_feature_prob:.4f}, "
          f"mask_feature_length={model.config.mask_feature_length}")


def _print_eval_samples(preds, refs, num_samples=5):
    """Print a few prediction vs reference samples for debugging."""
    print(f"\n--- Eval Samples (showing {min(num_samples, len(preds))} of {len(preds)}) ---")
    indices = list(range(len(preds)))
    if len(indices) > num_samples:
        random.seed(None)
        indices = random.sample(indices, num_samples)
    for idx in indices:
        print(f"  [{idx}] REF:  {refs[idx]}")
        print(f"  [{idx}] PRED: {preds[idx]}")
        print()
    print("--- End Samples ---\n")


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_manifest = config["data"]["train_manifest"]
    dev_manifest = config["data"].get("dev_manifest")
    if not dev_manifest or not Path(dev_manifest).exists():
        print(f"Warning: dev manifest not found at '{dev_manifest}'. Fallback to train manifest.")
        dev_manifest = train_manifest

    with open(config["data"]["vocab"], "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    id2phone = {int(v): k for k, v in vocab.items()}

    print(f"Vocab size: {vocab_size}")
    print(f"Train manifest: {train_manifest}")
    print(f"Dev manifest: {dev_manifest}")

    train_dataset = PhonemeDataset(train_manifest, config["data"]["vocab"], config["data"], is_train=True)
    dev_dataset = PhonemeDataset(dev_manifest, config["data"]["vocab"], config["data"], is_train=False)

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Dev dataset: {len(dev_dataset)} samples")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train"]["per_device_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config["train"]["per_device_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HubertForCTC.from_pretrained(config["model"]["pretrained"])

    # Check if vocab size matches or we need to reset head
    if model.config.vocab_size != vocab_size:
        print(f"Warning: model vocab size {model.config.vocab_size} != dataset vocab size {vocab_size}. Resetting head.")
        model.lm_head = torch.nn.Linear(model.config.hidden_size, vocab_size)
        model.config.vocab_size = vocab_size
    else:
        print(f"Vocab size matches checkpoint ({vocab_size}). Using Strategy A (full weight reuse).")

    # Freeze feature encoder (CNN layers) - always recommended
    if config["model"].get("freeze_feat", True):
        model.freeze_feature_encoder()

    # Freeze backbone for initial epochs
    freeze_backbone_epochs = config["model"].get("freeze_backbone_epochs", 0)
    backbone_frozen = False

    # Safety fix:
    # Disable gradient checkpointing to avoid conflict with feature extractor freeze/unfreeze.
    # (freeze_feature_encoder uses internal parameter freezing that can break checkpointing
    #  assumptions about grad flow through backbone inputs.)
    use_gradient_checkpointing = False
    if config["train"].get("fp16", True):
        print("Gradient checkpointing is disabled for training stability.")

    if freeze_backbone_epochs > 0:
        print(f"Freezing backbone for first {freeze_backbone_epochs} epochs (only lm_head trains)")
        _freeze_backbone(model)
        backbone_frozen = True
    else:
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

    # Configure SpecAugment
    _configure_specaugment(model, config)

    model.to(device)

    # Build optimizer
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, config)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    total_steps = len(train_dataloader) * config["train"]["epochs"] // config["train"]["grad_accum_steps"]
    warmup_steps = int(total_steps * config["train"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config["train"]["fp16"])

    out_dir = Path(config["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    log_dir = Path("logs") / out_dir.name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs: {log_dir}")

    best_per = float("inf")
    patience_counter = 0
    early_stop_patience = config["train"].get("early_stop_patience", 10)
    global_step = 0
    num_eval_samples = config["train"].get("num_eval_samples", 5)

    for epoch in range(1, config["train"]["epochs"] + 1):
        # Unfreeze backbone after freeze_backbone_epochs
        if backbone_frozen and epoch > freeze_backbone_epochs:
            print(f"\n=== Epoch {epoch}: Unfreezing backbone ===")
            _unfreeze_backbone(model)
            # Re-freeze feature encoder if configured
            if config["model"].get("freeze_feat", True):
                model.freeze_feature_encoder()
            backbone_frozen = False

            # Keep gradient checkpointing disabled for stability
            if use_gradient_checkpointing:
                model.gradient_checkpointing_enable()

            # Reset patience — new training phase
            patience_counter = 0
            print(f"  Patience counter reset. Best PER so far: {best_per:.4f}")

            # Rebuild optimizer with backbone at a SMALLER initial lr to avoid CTC collapse
            # Use 1/10 of target lr initially; the warmup will ramp it up
            unfreeze_backbone_lr = float(config["train"].get("lr_backbone", 3e-5))
            print(f"  Backbone lr: {unfreeze_backbone_lr:.1e}")
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(
                model, config, backbone_lr_override=unfreeze_backbone_lr
            )
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

            # Longer warmup after unfreeze to prevent catastrophic forgetting
            remaining_epochs = config["train"]["epochs"] - epoch + 1
            remaining_steps = len(train_dataloader) * remaining_epochs // config["train"]["grad_accum_steps"]
            # 10% warmup after unfreeze (critical to avoid CTC collapse)
            remaining_warmup = int(remaining_steps * 0.10)
            print(f"  Remaining steps: {remaining_steps}, warmup: {remaining_warmup}")
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=remaining_warmup, num_training_steps=remaining_steps
            )
            scaler = torch.amp.GradScaler("cuda", enabled=config["train"]["fp16"])

        model.train()
        train_loss = 0
        num_batches = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} Train")):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=config["train"]["fp16"]):
                outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            loss = loss / config["train"]["grad_accum_steps"]
            scaler.scale(loss).backward()

            train_loss += loss.item() * config["train"]["grad_accum_steps"]
            num_batches += 1

            if (step + 1) % config["train"]["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                writer.add_scalar("train/loss_step", loss.item() * config["train"]["grad_accum_steps"], global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        avg_train_loss = train_loss / max(num_batches, 1)
        print(f"Epoch {epoch} Avg Loss: {avg_train_loss:.4f}")
        writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)

        # Evaluation
        if epoch % config["train"]["eval_every_epoch"] == 0:
            model.eval()
            preds = []
            refs = []
            eval_loss = 0
            eval_batches = 0

            with torch.no_grad():
                for batch in tqdm(dev_dataloader, desc=f"Epoch {epoch} Eval"):
                    input_values = batch["input_values"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.amp.autocast("cuda", enabled=config["train"]["fp16"]):
                        outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                        eval_loss += outputs.loss.item()
                        eval_batches += 1
                        logits = outputs.logits

                    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
                    label_ids = batch["labels"].numpy()

                    for i in range(len(pred_ids)):
                        # CTC greedy decode: collapse repeats and remove blank (id=0)
                        pred_seq = []
                        prev = -1
                        for p in pred_ids[i]:
                            if p != prev and p != 0:
                                pred_seq.append(id2phone.get(p, "<unk>"))
                            prev = p
                        preds.append(" ".join(pred_seq))

                        ref_seq = [id2phone.get(int(l), "<unk>") for l in label_ids[i] if l != -100]
                        refs.append(" ".join(ref_seq))

            per = compute_per(predictions=preds, references=refs)
            avg_eval_loss = eval_loss / max(eval_batches, 1)
            print(f"Epoch {epoch} PER: {per:.4f} | Eval Loss: {avg_eval_loss:.4f}")

            # Print sample predictions
            _print_eval_samples(preds, refs, num_samples=num_eval_samples)

            writer.add_scalar("eval/per", per, epoch)
            writer.add_scalar("eval/loss", avg_eval_loss, epoch)

            if per < best_per:
                best_per = per
                patience_counter = 0
                model.save_pretrained(out_dir / "best")
                with open(out_dir / "best" / "phoneme_vocab.json", "w", encoding="utf-8") as f:
                    json.dump({k: v for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)
                print(f"Saved new best model with PER: {best_per:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best PER: {best_per:.4f}")
                break

    writer.close()
    print(f"\nTraining complete. Best PER: {best_per:.4f}")
    print(f"Best model saved to: {out_dir / 'best'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)