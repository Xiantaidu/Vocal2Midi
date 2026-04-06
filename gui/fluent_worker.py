import pathlib
import sys
import traceback

import torch
from PyQt5.QtCore import QThread, pyqtSignal as Signal

from gui.fluent_utils import t0_nstep_to_ts


# Import the hybrid pipeline
try:
    from inference.auto_lyric_hybrid import auto_lyric_hybrid_pipeline
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

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._is_running = True

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamRedirector(sys.stdout, self.log_signal)
        sys.stderr = StreamRedirector(sys.stderr, self.log_signal)

        try:
            audio_files = self.kwargs['audio_files']
            save_dir = pathlib.Path(self.kwargs['save_dir'])

            for audio_path in audio_files:
                if not self._is_running:
                    break
                original_path = pathlib.Path(audio_path)
                filename = original_path.name
                self.log_signal.emit(f"========== 正在处理: {filename} ==========")

                ts_list = t0_nstep_to_ts(self.kwargs['t0'], int(self.kwargs['nsteps']))
                ts_tensor = torch.tensor(ts_list, device=self.kwargs['device'])

                auto_lyric_hybrid_pipeline(
                    audio_path=str(original_path),
                    output_filename=filename,
                    game_model_dir=self.kwargs['game_model_path_str'],
                    device=self.kwargs['device'],
                    hfa_model_dir=self.kwargs['hfa_model_path_str'],
                    asr_model_path=self.kwargs['asr_model_path_str'],
                    ts=ts_tensor,
                    language=self.kwargs['language'],
                    original_lyrics=self.kwargs['original_lyrics'],
                    output_dir=save_dir,
                    output_formats=self.kwargs['output_formats'],
                    slicing_method=self.kwargs['slicing_method'],
                    tempo=self.kwargs['tempo'],
                    quantization_step=self.kwargs['quantization_step'],
                    pitch_format=self.kwargs['pitch_format'],
                    round_pitch=self.kwargs['round_pitch'],
                    seg_threshold=self.kwargs['seg_threshold'],
                    seg_radius=self.kwargs['seg_radius'],
                    est_threshold=self.kwargs['est_threshold'],
                    batch_size=self.kwargs['batch_size'],
                    asr_batch_size=self.kwargs['asr_batch_size'],
                    debug_mode=True,
                    cancel_checker=lambda: (not self._is_running) or self.isInterruptionRequested(),
                )

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
