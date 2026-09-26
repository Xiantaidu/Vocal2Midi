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


# Import the hybrid pipeline
try:
    from application.pipeline import run_auto_lyric_job, open_asr_session
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Hybrid pipeline not available. Error: {e}")
    HYBRID_AVAILABLE = False


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

    def __init__(self, tasks: list[tuple[PipelineConfig, str]]):
        """Initialize the worker thread with a list of (config, filename) tasks.

        Each task carries its own PipelineConfig so every file can have
        custom settings in batch mode.

        Args:
            tasks: List of (PipelineConfig, audio filename) tuples.
        """
        super().__init__()
        self.tasks = tasks
        self._is_running = True

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamRedirector(sys.stdout, self.log_signal)
        sys.stderr = StreamRedirector(sys.stderr, self.log_signal)

        total = len(self.tasks)
        # One lazily-spawned ASR subprocess serves the whole batch; the
        # multi-GB Qwen model loads once instead of per file.
        session = open_asr_session() if HYBRID_AVAILABLE else None
        # Pipeline modules log instead of printing; capture those records for
        # the GUI log window while the worker runs. The stdout redirector
        # below stays for third-party prints that bypass logging.
        root_logger = logging.getLogger()
        old_root_level = root_logger.level
        log_handler = _QtLogHandler(self.log_signal)
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.INFO)
        try:
            save_dirs = {str(task_config.output_dir) for task_config, _ in self.tasks}
            save_dir = save_dirs.pop() if len(save_dirs) == 1 else "; ".join(sorted(save_dirs))

            for file_index, (config, filename) in enumerate(self.tasks, start=1):
                if not self._is_running:
                    break
                self.log_signal.emit(tr("worker_processing", f=filename))

                config.audio_path = str(pathlib.Path(filename))
                config.output_filename = filename
                config.asr_session = session
                config.cancel_checker = lambda: (
                    not self._is_running
                ) or self.isInterruptionRequested()

                self.progress_signal.emit(file_index, total, filename)
                run_auto_lyric_job(config)

            if self._is_running:
                self.finished_signal.emit(tr("worker_success", d=save_dir))
            else:
                self.error_signal.emit(tr("worker_cancelled"))

        except (InterruptedError, CancellationError):
            self.error_signal.emit(tr("worker_stopped"))
        except Exception:
            self.error_signal.emit(tr("worker_error", tb=traceback.format_exc()))
        finally:
            root_logger.removeHandler(log_handler)
            root_logger.setLevel(old_root_level)
            if session is not None:
                session.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def stop(self):
        self._is_running = False
        self.requestInterruption()
