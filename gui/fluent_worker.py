import logging
import pathlib
import sys
import traceback

from PySide6.QtCore import QThread, Signal

from application.config import PipelineConfig
from application.exceptions import CancellationError
from gui.i18n import tr


class _QtLogHandler(logging.Handler):
    """Forwards logging records from the pipeline into the GUI log window."""

    def __init__(self, signal):
        super().__init__(level=logging.INFO)
        self.signal = signal

    def emit(self, record):
        try:
            self.signal.emit(record.getMessage())
        except Exception:
            self.handleError()


try:
    from application.batch import run_auto_lyric_batch
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Hybrid pipeline not available. Error: {e}")
    HYBRID_AVAILABLE = False


def classify_batch_result(items):
    """Summarize a finished batch so the UI never reports success on failure.

    The batch runner isolates per-song failures instead of raising, so a clean
    return says nothing about the outcome; this turns the item list into one of
    ``("success" | "partial" | "all_failed", total, failed, first_error)``.
    """
    items = list(items or [])
    failed = [item for item in items if getattr(item, "failed_stage", None)]
    if items and len(failed) == len(items):
        return "all_failed", len(items), failed, failed[0].error
    if failed:
        return "partial", len(items), failed, None
    return "success", len(items), failed, None


class StreamRedirector:
    def __init__(self, stream, signal):
        self.stream = stream
        self.signal = signal

    def write(self, text):
        if text.strip():
            self.signal.emit(text.strip())
        self.stream.write(text)

    def flush(self):
        self.stream.flush()


class WorkerThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)
    progress_signal = Signal(int, int, str)  # (current file index, total files, filename)
    stage_progress_signal = Signal(str, int, int, str)  # (stage name, done, total, filename)

    def __init__(self, tasks: list[tuple[PipelineConfig, str]]):
        """Initialize the worker with a list of ``(config, filename)`` tasks.

        Each task carries its own PipelineConfig so files can be customized
        individually in batch mode.
        """
        super().__init__()
        self.tasks = tasks
        self._is_running = True

    def _cancel_checker(self) -> bool:
        return (not self._is_running) or self.isInterruptionRequested()

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamRedirector(sys.stdout, self.log_signal)
        sys.stderr = StreamRedirector(sys.stderr, self.log_signal)

        total = len(self.tasks)
        # Pipeline modules log instead of printing; capture those records for
        # the GUI log. The stdout redirector stays for third-party prints that
        # bypass logging.
        root_logger = logging.getLogger()
        old_root_level = root_logger.level
        log_handler = _QtLogHandler(self.log_signal)
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.INFO)
        try:
            save_dirs = {str(task_config.output_dir) for task_config, _ in self.tasks}
            save_dir = save_dirs.pop() if len(save_dirs) == 1 else "; ".join(sorted(save_dirs))

            for index, (config, filename) in enumerate(self.tasks, start=1):
                if not self._is_running:
                    break
                config.audio_path = str(pathlib.Path(filename))
                config.output_filename = filename
                config.cancel_checker = self._cancel_checker
                self.progress_signal.emit(index, total, filename)

            items = run_auto_lyric_batch(
                self.tasks,
                log=self.log_signal.emit,
                progress=self._on_stage_progress,
                cancel_checker=self._cancel_checker,
            )

            # The batch isolates per-song failures, so "no exception" does not
            # mean success: report what actually came out of the queue.
            outcome, total_items, failed, first_error = classify_batch_result(items)
            if not self._is_running:
                self.error_signal.emit(tr("worker_cancelled"))
            elif outcome == "all_failed":
                self.error_signal.emit(
                    tr("worker_all_failed", n=total_items, e=first_error)
                )
            elif outcome == "partial":
                self.finished_signal.emit(
                    tr("worker_partial", ok=total_items - len(failed),
                       n=total_items, bad=len(failed), d=save_dir)
                )
            else:
                self.finished_signal.emit(tr("worker_success", d=save_dir))

        except (InterruptedError, CancellationError):
            self.error_signal.emit(tr("worker_stopped"))
        except Exception:
            self.error_signal.emit(tr("worker_error", tb=traceback.format_exc()))
        finally:
            root_logger.removeHandler(log_handler)
            root_logger.setLevel(old_root_level)
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _on_stage_progress(self, stage: str, done: int, total: int, filename: str) -> None:
        """Batch stages sweep the queue, so report stage progress first."""
        self.stage_progress_signal.emit(stage, done, total, filename)
        self.log_signal.emit(
            tr("worker_stage", s=tr(f"stage_{stage}"), i=done, n=total, f=pathlib.Path(filename).name)
        )

    def stop(self):
        self._is_running = False
        self.requestInterruption()
