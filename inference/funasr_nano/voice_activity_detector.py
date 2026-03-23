"""
Silero VAD (Voice Activity Detection) implementation in Python.

This code is converted from sherpa-onnx's C++ implementation.
The ONNX model (silero_vad.onnx) is provided by the sherpa-onnx project.

References:
  - sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
  - Model: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
"""

import numpy as np
import onnxruntime as ort
import soundfile as sf
import time

class SileroVAD:
    def __init__(self, model_path, threshold=0.5, min_silence_duration=0.5,
                 min_speech_duration=0.25, window_size=512,
                 max_speech_duration=20, neg_threshold=-1,
                 sample_rate=16000, num_threads=1):
        if sample_rate != 16000:
            raise ValueError(f"Expected sample rate 16000. Given: {sample_rate}")

        self.model_path = model_path
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration
        self.window_size = window_size
        self.max_speech_duration = max_speech_duration
        self.neg_threshold = neg_threshold
        self.sample_rate = sample_rate
        self.num_threads = num_threads

        self.min_silence_samples = int(sample_rate * min_silence_duration)
        self.min_speech_samples = int(sample_rate * min_speech_duration)
        self.max_speech_samples = int(sample_rate * max_speech_duration)

        self.original_threshold = threshold
        self.original_min_silence_duration = min_silence_duration
        self.original_min_silence_samples = self.min_silence_samples

        self.triggered = False
        self.current_sample = 0
        self.temp_start = 0
        self.temp_end = 0

        self._init_session()

    def _init_session(self):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = self.num_threads

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        is_k2fsa = self._is_exported_by_k2fsa()
        if (len(self.input_names) == 4 and len(self.output_names) == 3) or \
           is_k2fsa:
            self.is_v5 = False
            self.window_overlap = 0
        elif len(self.input_names) == 3 and len(self.output_names) == 2:
            self.is_v5 = True
            self.window_overlap = 64  # 64 for 16kHz, 32 for 8kHz

            if self.window_size != 512:
                raise ValueError(
                    f"For silero_vad v5, we require window_size to be 512 for 16kHz"
                )
        else:
            raise ValueError(f"Unsupported silero vad model: "
                           f"{len(self.input_names)} inputs, {len(self.output_names)} outputs")

        self.reset()

    def _is_exported_by_k2fsa(self):
        if len(self.input_names) == 3 and \
           self.input_names[0] == "x" and \
           self.input_names[1] == "h" and \
           self.input_names[2] == "c" and \
           len(self.output_names) == 3 and \
           self.output_names[0] == "prob" and \
           self.output_names[1] == "new_h" and \
           self.output_names[2] == "new_c":
            return True
        return False

    def reset(self):
        if self.is_v5:
            # 2 LSTM layers, batch=1, hidden_dim=128
            self.states = [np.zeros((2, 1, 128), dtype=np.float32)]
        else:
            # 2 LSTM layers, batch=1, hidden_dim=64
            self.states = [
                np.zeros((2, 1, 64), dtype=np.float32),  # h
                np.zeros((2, 1, 64), dtype=np.float32)   # c
            ]

        self.triggered = False
        self.current_sample = 0
        self.temp_start = 0
        self.temp_end = 0

        self.threshold = self.original_threshold
        self.min_silence_duration = self.original_min_silence_duration
        self.min_silence_samples = self.original_min_silence_samples

    def compute(self, samples):
        if self.is_v5:
            return self._run_v5(samples)
        else:
            return self._run_v4(samples)

    def _run_v5(self, samples):
        n = len(samples)

        x = np.array(samples, dtype=np.float32).reshape(1, n)
        sr = np.array([self.sample_rate], dtype=np.int64)

        inputs = {
            self.input_names[0]: x,
            self.input_names[1]: self.states[0],
            self.input_names[2]: sr
        }

        outputs = self.session.run(self.output_names, inputs)

        self.states[0] = outputs[1]

        prob_array = np.array(outputs[0])
        prob = float(prob_array.item() if prob_array.size == 1 else prob_array[0])
        return prob

    def _run_v4(self, samples):
        n = len(samples)

        x = np.array(samples, dtype=np.float32).reshape(1, n)
        sr = np.array([self.sample_rate], dtype=np.int64)

        inputs = {}
        inputs[self.input_names[0]] = x

        if len(self.input_names) == 4:
            # Standard V4 model: input, sr, h, c
            inputs[self.input_names[1]] = sr
            inputs[self.input_names[2]] = self.states[0]
            inputs[self.input_names[3]] = self.states[1]
        else:
            # k2-fsa exported model: x, h, c
            inputs[self.input_names[1]] = self.states[0]
            inputs[self.input_names[2]] = self.states[1]

        outputs = self.session.run(self.output_names, inputs)

        self.states[0] = outputs[1]
        self.states[1] = outputs[2]

        prob_array = np.array(outputs[0])
        prob = float(prob_array.item() if prob_array.size == 1 else prob_array[0])
        return prob

    def is_speech(self, samples):
        window_size = self.get_window_size()
        if len(samples) != window_size:
            raise ValueError(f"n: {len(samples)} != window_size: {window_size}")

        current_speech_length = 0
        if self.triggered and self.temp_start != 0:
            current_speech_length = self.current_sample - self.temp_start

        if current_speech_length > self.max_speech_samples:
            self.threshold = 0.9
            self.min_silence_duration = 0.1
            self.min_silence_samples = int(self.sample_rate * 0.1)
        else:
            self.threshold = self.original_threshold
            self.min_silence_duration = self.original_min_silence_duration
            self.min_silence_samples = self.original_min_silence_samples

        prob = self.compute(samples)

        self.current_sample += self.get_window_shift()

        if prob > self.threshold and self.temp_end != 0:
            self.temp_end = 0

        if prob > self.threshold and self.temp_start == 0:
            self.temp_start = self.current_sample
            return False

        if prob > self.threshold and self.temp_start != 0 and not self.triggered:
            if self.current_sample - self.temp_start < self.min_speech_samples:
                return False

            self.triggered = True
            return True

        if (prob < self.threshold) and not self.triggered:
            self.temp_start = 0
            self.temp_end = 0
            return False

        if self.neg_threshold < 0:
            neg_threshold = max(self.threshold - 0.15, 0.01)
        else:
            neg_threshold = max(self.neg_threshold, 0.01)

        if (prob > neg_threshold) and self.triggered:
            return True

        if (prob > self.threshold) and not self.triggered:
            self.triggered = True
            return True

        if (prob < self.threshold) and self.triggered:
            if self.temp_end == 0:
                self.temp_end = self.current_sample

            if self.current_sample - self.temp_end < self.min_silence_samples:
                return True

            self.temp_start = 0
            self.temp_end = 0
            self.triggered = False
            return False

        return False

    def get_window_size(self):
        return self.window_size + self.window_overlap

    def get_window_shift(self):
        return self.window_size

if __name__ == "__main__":
    import os

    vad_path = "models/silero_vad.onnx"
    audio_path = "examples/song.wav"

    if not os.path.exists(vad_path):
        print(f"Error: Model file not found: {vad_path}")
        exit(1)

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        exit(1)

    print(f"Loading VAD model from: {vad_path}")
    vad = SileroVAD(
        model_path=vad_path,
        threshold=0.5,
        min_silence_duration=0.5,
        min_speech_duration=0.25,
        window_size=512,
        sample_rate=16000
    )

    print(f"Loading audio file: {audio_path}")
    audio, sr = sf.read(audio_path)

    if len(audio.shape) > 1:
        audio = audio[:, 0]

    print(f"Audio info: shape={audio.shape}, sample_rate={sr}, duration={len(audio)/sr:.2f}s")

    if sr != 16000:
        print(f"Resampling from {sr}Hz to 16000Hz...")
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        except ImportError:
            print("Warning: librosa not installed. Cannot resample audio.")
            print("Please install librosa: pip install librosa")
            print("Or use audio with 16kHz sample rate.")
            exit(1)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val

    vad.reset()

    window_size_samples = vad.get_window_size()
    window_shift_samples = vad.get_window_shift()

    print(f"\nProcessing audio...")
    print(f"Window size: {window_size_samples} samples ({window_size_samples/sr:.3f}s)")
    print(f"Window shift: {window_shift_samples} samples ({window_shift_samples/sr:.3f}s)")
    print("-" * 80)

    speech_segments = []
    current_speech_start = None

    start_time = time.time()
    num_chunks = 0
    speech_chunks = 0

    for i in range(0, len(audio) - window_size_samples + 1, window_shift_samples):
        chunk = audio[i:i + window_size_samples]

        is_speech = vad.is_speech(chunk)

        num_chunks += 1
        if is_speech:
            speech_chunks += 1

        time_sec = i / sr
        if is_speech:
            if current_speech_start is None:
                current_speech_start = time_sec
        else:
            if current_speech_start is not None:
                speech_segments.append((current_speech_start, time_sec))
                current_speech_start = None

        if num_chunks % 100 == 0:
            print(f"Time: {time_sec:.2f}s, is_speech={is_speech}")

    if current_speech_start is not None:
        speech_segments.append((current_speech_start, len(audio) / sr))

    elapsed_time = time.time() - start_time
    audio_duration = len(audio) / sr

    print("-" * 80)
    print(f"\nProcessing complete!")
    print(f"Total chunks: {num_chunks}")
    print(f"Speech chunks: {speech_chunks} ({speech_chunks/num_chunks*100:.1f}%)")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Processing time: {elapsed_time:.2f}s")
    print(f"Real-time factor: {audio_duration/elapsed_time:.2f}x")
    print(f"\nDetected {len(speech_segments)} speech segment(s) (max_duration={vad.max_speech_duration}s):")
    for i, (start, end) in enumerate(speech_segments, 1):
        print(f"  Segment {i}: {start:.3f}s - {end:.3f}s (duration: {end-start:.3f}s)")
