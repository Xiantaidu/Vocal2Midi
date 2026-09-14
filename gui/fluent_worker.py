import pathlib
import sys
import traceback

from PySide6.QtCore import QThread, Signal

from application.config import PipelineConfig
from application.exceptions import CancellationError
from gui.i18n import tr


# Import the hybrid pipeline
try:
    from application.pipeline import run_auto_lyric_job
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

    def __init__(self, config: PipelineConfig, audio_files: list):
        """Initialize the worker thread with a PipelineConfig and audio file list.

        Args:
            config: PipelineConfig with all pipeline parameters.
            audio_files: List of audio file paths to process.
        """
        super().__init__()
        self.config = config
        self.audio_files = audio_files
        self._is_running = True

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamRedirector(sys.stdout, self.log_signal)
        sys.stderr = StreamRedirector(sys.stderr, self.log_signal)

        try:
            save_dir = self.config.output_dir

            for file_index, audio_path in enumerate(self.audio_files, start=1):
                if not self._is_running:
                    break
                original_path = pathlib.Path(audio_path)
                filename = original_path.name
                self.log_signal.emit(tr("worker_processing", f=filename))

                # Update per-file fields in config
                self.config.audio_path = str(original_path)
                self.config.output_filename = filename
                self.config.cancel_checker = lambda: (
                    not self._is_running
                ) or self.isInterruptionRequested()

                self.progress_signal.emit(file_index, len(self.audio_files), filename)
                run_auto_lyric_job(self.config)

            if self._is_running:
                self.finished_signal.emit(tr("worker_success", d=save_dir))
            else:
                self.error_signal.emit(tr("worker_cancelled"))

        except (InterruptedError, CancellationError):
            self.error_signal.emit(tr("worker_stopped"))
        except Exception:
            self.error_signal.emit(tr("worker_error", tb=traceback.format_exc()))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def stop(self):
        self._is_running = False
        self.requestInterruption()
