import pathlib
import sys
import traceback

from PyQt5.QtCore import QThread, pyqtSignal as Signal

from application.config import PipelineConfig


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

            for audio_path in self.audio_files:
                if not self._is_running:
                    break
                original_path = pathlib.Path(audio_path)
                filename = original_path.name
                self.log_signal.emit(f"========== 正在处理: {filename} ==========")

                # Update per-file fields in config
                self.config.audio_path = str(original_path)
                self.config.output_filename = filename
                self.config.cancel_checker = lambda: (
                    not self._is_running
                ) or self.isInterruptionRequested()
                self.config.debug_mode = True

                run_auto_lyric_job(self.config)

            if self._is_running:
                self.finished_signal.emit(f"提取成功！文件已保存至: {save_dir}")
            else:
                self.error_signal.emit("任务已被取消。")

        except InterruptedError:
            self.error_signal.emit("任务已被强制停止。")
        except Exception:
            self.error_signal.emit(f"发生错误:\n{traceback.format_exc()}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def stop(self):
        self._is_running = False
        self.requestInterruption()
