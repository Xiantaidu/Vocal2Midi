import os
import pathlib
import shutil

import click
import lightning as pl
import torch
from torch.utils.data import DataLoader

from networks.task.forced_alignment_task import LitForcedAlignmentTask
from networks.task.non_lexical_labeler_task import LitNonLexicalLabelerTask
from tools.config_utils import load_yaml
from tools.dataset import BinningAudioBatchSampler, ForcedAlignmentDataset, NonLexicalLabelerDataset, \
    non_lexical_labeler_collate_fn, forced_alignment_collate_fn
from tools.train_callbacks import StepProgressBar, RecentCheckpointsCallback, MonitorCheckpointsCallback


class BaseTrainer(object):
    def __init__(self, train_config_path):
        self.model = None

        self.collate_fn = None
        self.evaluate_dataset = None
        self.valid_dataset = None
        self.train_dataset = None

        self.train_config: dict = load_yaml(train_config_path)
        self.binary_folder = pathlib.Path(self.train_config["binary_folder"])
        self.vocab = load_yaml(self.binary_folder / "vocab.yaml")
        self.train_config.update(load_yaml(self.binary_folder / "datasets.yaml"))

        self.save_model_folder = pathlib.Path("ckpt") / self.train_config["model_name"]
        self.save_model_folder.mkdir(parents=True, exist_ok=True)

        shutil.copy(self.binary_folder / "vocab.yaml", self.save_model_folder)
        shutil.copy(self.binary_folder / "config.yaml", self.save_model_folder)
        shutil.copy(self.binary_folder / "datasets.yaml", self.save_model_folder)

        self.copy_source()
        self.evaluate_checkpoint = None

    def copy_source(self):
        pass

    def train(self, resume=False):
        torch.set_float32_matmul_precision(self.train_config["float32_matmul_precision"])
        pl.seed_everything(self.train_config["random_seed"], workers=True)

        # define dataset
        num_workers = self.train_config['dataloader_workers']
        train_sampler = BinningAudioBatchSampler(
            self.train_dataset.get_aug_wav_lengths(),
            self.train_config["batch_max_length"],
            self.train_config["binning_length"],
            self.train_config["drop_last"],
        )
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            prefetch_factor=(2 if num_workers > 0 else None),
        )

        valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        evaluate_dataloader = DataLoader(
            dataset=self.evaluate_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        recent_checkpoints_callback = RecentCheckpointsCallback(
            dirpath=self.save_model_folder,
            save_top_k=self.train_config["save_top_k"],
            save_every_steps=self.train_config["save_every_steps"],
        )

        stepProgressBar = StepProgressBar()

        # trainer
        trainer = pl.Trainer(
            accelerator=self.train_config["accelerator"],
            devices=self.train_config["devices"],
            precision=self.train_config["precision"],
            gradient_clip_val=self.train_config["gradient_clip_val"],
            gradient_clip_algorithm=self.train_config["gradient_clip_algorithm"],
            default_root_dir=self.save_model_folder,
            val_check_interval=self.train_config["val_check_interval"],
            check_val_every_n_epoch=None,
            max_epochs=-1,
            max_steps=self.train_config["optimizer_config"]["total_steps"],
            callbacks=[recent_checkpoints_callback, self.evaluate_checkpoint, stepProgressBar],
        )

        ckpt_path = None
        if resume:
            # resume training state
            ckpt_path_list = self.save_model_folder.glob("*.ckpt")
            ckpt_path_list = sorted(
                ckpt_path_list, key=lambda x: int(x.stem.split("step=")[-1]), reverse=True
            )
            ckpt_path = str(ckpt_path_list[0]) if len(ckpt_path_list) > 0 else None

        # start training
        trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=[valid_dataloader, evaluate_dataloader],
            ckpt_path=ckpt_path,
        )

        # Discard the optimizer and save
        trainer.save_checkpoint(
            str(pathlib.Path("ckpt") / self.train_config["model_name"]) + ".ckpt", weights_only=True
        )


class NonLexicalLabelerTrainer(BaseTrainer):
    def __init__(self, train_config_path):
        super().__init__(train_config_path)
        self.model = LitNonLexicalLabelerTask(self.vocab, self.train_config)

        self.collate_fn = non_lexical_labeler_collate_fn
        self.train_dataset = NonLexicalLabelerDataset(self.binary_folder, prefix="train")
        self.valid_dataset = NonLexicalLabelerDataset(self.binary_folder, prefix="valid")
        self.evaluate_dataset = NonLexicalLabelerDataset(self.binary_folder, prefix="evaluate")

        self.evaluate_checkpoint = MonitorCheckpointsCallback(
            dirpath=self.save_model_folder,
            monitor="unseen_evaluate/frame_error_rate",
            mode="min",
            save_top_k=5,
        )


class ForcedAlignmentTrainer(BaseTrainer):
    def __init__(self, train_config_path):
        super().__init__(train_config_path)
        self.model = LitForcedAlignmentTask(self.vocab, self.train_config)

        self.collate_fn = forced_alignment_collate_fn
        self.evaluate_dataset = ForcedAlignmentDataset(self.binary_folder, prefix="evaluate")
        self.train_dataset = ForcedAlignmentDataset(self.binary_folder, prefix="train")
        self.valid_dataset = ForcedAlignmentDataset(self.binary_folder, prefix="valid")

        self.evaluate_checkpoint = MonitorCheckpointsCallback(
            dirpath=self.save_model_folder,
            monitor="unseen_evaluate/total",
            mode="min",
            save_top_k=5,
        )

    def copy_source(self):
        for lang, dict_path in self.vocab["dictionaries"].items():
            shutil.copy(self.binary_folder / dict_path, self.save_model_folder)
            print(f'| Copied dictionary for language-{lang}-{dict_path} to {self.save_model_folder}.')


@click.command()
@click.option("--config", "-c", type=str, required=True, help="training config path")
@click.option("--model", "-m", type=str, required=True,
              help="model type: nll[non_lexical_labeler model, first step] fa[forced_alignment model, second step]")
@click.option("--resume", "-r", is_flag=True, default=False, show_default=True,
              help="resume training from checkpoint")
def main(config: str, model: str, resume: bool):
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Prevent unacceptable slowdowns when using 16 precision

    assert model in ['nll', 'fa'], "model type must in ['nll', 'fa'], please read help info or README.md."

    if model == 'nll':
        NonLexicalLabelerTrainer(config).train(resume=resume)
    elif model == 'fa':
        ForcedAlignmentTrainer(config).train(resume=resume)
    else:
        raise Exception(f"unknown model {model}")


if __name__ == "__main__":
    main()
