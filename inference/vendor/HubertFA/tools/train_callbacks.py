import os

from lightning.pytorch.callbacks import Callback, TQDMProgressBar


class StepProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self._global_step = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        self._global_step = trainer.global_step

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items["step"] = self._global_step
        return items


class RecentCheckpointsCallback(Callback):
    def __init__(self, dirpath, save_top_k=5, save_every_steps=5000):
        self.dirpath = dirpath
        self.save_top_k = save_top_k
        self.filename = "checkpoint-step={step}"
        self.saved_checkpoints = []
        self.save_every_steps = save_every_steps

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step % self.save_every_steps == 0:
            checkpoint_path = os.path.join(
                self.dirpath,
                self.filename.format(step=trainer.global_step) + ".ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)
            self.saved_checkpoints.append(checkpoint_path)

            if len(self.saved_checkpoints) > self.save_top_k:
                oldest_checkpoint = self.saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)


class MonitorCheckpointsCallback(Callback):
    def __init__(self, dirpath, monitor, save_top_k=5, mode='min', filename=None):
        self.dirpath = dirpath
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.mode = mode
        self.filename = filename if filename else "best-step={step}-{monitor}={value:.5f}"
        self.best_checkpoints = []

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step:
            current_value = trainer.callback_metrics.get(self.monitor)
            if current_value is None:
                return
            current_value = current_value.item()

            filename = self.filename.format(
                step=trainer.global_step,
                monitor=self.monitor.replace("/", "_"),
                value=current_value
            )
            checkpoint_path = os.path.join(self.dirpath, filename + ".ckpt")

            should_save = False
            if self.save_top_k <= 0:
                should_save = True
            else:
                if len(self.best_checkpoints) < self.save_top_k:
                    should_save = True
                else:
                    comparator = max if self.mode == 'min' else min
                    threshold = comparator([v for v, _ in self.best_checkpoints])
                    if (self.mode == 'min' and current_value < threshold) or (
                            self.mode == 'max' and current_value > threshold):
                        should_save = True

            if should_save:
                trainer.save_checkpoint(checkpoint_path)
                self.best_checkpoints.append((current_value, checkpoint_path))
                self.best_checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))
                if 0 < self.save_top_k < len(self.best_checkpoints):
                    removed = self.best_checkpoints[self.save_top_k:]
                    self.best_checkpoints = self.best_checkpoints[:self.save_top_k]
                    for _, path in removed:
                        if os.path.exists(path):
                            os.remove(path)
